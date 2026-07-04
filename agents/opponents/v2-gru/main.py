

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


USE_GRU_CONTROLLER = True
GRU_WEIGHTS_AVAILABLE = False
GRU_SEQUENCE_LEN = 12
GRU_FEATURE_DIM = 24
GRU_EMBEDDED_STATE_DICT = None
_GRU_CONTROLLER = None


class TinyGRUStrategyController(torch.nn.Module):
    def __init__(self, input_dim: int = GRU_FEATURE_DIM, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 4)

    def forward(self, seq: Tensor) -> Tensor:
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Reachable enemy planet mass proxy per planet."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    pid = int(player_id)
    H = max(float(horizon), 1e-6)

    # ----- planet-source term (unchanged) ---------------------------------
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt]
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * H).clamp(min=1e-6)                 # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)    # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye               # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib_planets = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib_planets.sum(dim=0)                                    # [P]


def _step_from_tensors(obs_tensors: dict) -> int:
    step = obs_tensors.get("step", 0)
    if torch.is_tensor(step):
        return int(step.reshape(-1)[0].item())
    return int(step)


def _owner_strengths(obs, prod: Tensor, player_count: int) -> tuple[Tensor, Tensor, Tensor]:
    dtype = obs.ships.dtype
    device = obs.device
    owner = obs.owner_abs.to(torch.long)
    ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
    prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
    for oid in range(int(player_count)):
        owned_by = obs.alive & (owner == oid)
        ships_by_owner[oid] = obs.ships.to(dtype)[owned_by].sum()
        prod_by_owner[oid] = prod.to(dtype)[owned_by].sum()
    strength = prod_by_owner + 0.025 * ships_by_owner
    return prod_by_owner, ships_by_owner, strength


def extract_global_features(
    obs,
    player_id: int,
    player_count: int,
    step: int,
    prod: Tensor,
    previous_stats: dict | None = None,
) -> Tensor:
    P = int(obs.P)
    dtype = torch.float32
    total_planets = max(P, 1)
    pid = int(player_id)
    pc = int(player_count)
    owner = obs.owner_abs.to(torch.long)
    alive = obs.alive
    prod_by_owner, ships_by_owner, strength = _owner_strengths(obs, prod, pc)

    my_mask = alive & (owner == pid)
    neutral_mask = alive & (owner < 0)
    my_planets = float(my_mask.sum().item())
    my_ships = float(ships_by_owner[pid].item()) if 0 <= pid < pc else 0.0
    my_prod = float(prod_by_owner[pid].item()) if 0 <= pid < pc else 0.0
    my_strength = float(strength[pid].item()) if 0 <= pid < pc else 0.0

    enemy_planet_counts = []
    enemy_ships = []
    enemy_prod = []
    enemy_strength = []
    for oid in range(pc):
        if oid == pid:
            continue
        oid_mask = alive & (owner == oid)
        enemy_planet_counts.append(float(oid_mask.sum().item()))
        enemy_ships.append(float(ships_by_owner[oid].item()))
        enemy_prod.append(float(prod_by_owner[oid].item()))
        enemy_strength.append(float(strength[oid].item()))
    enemy_max_planets = max(enemy_planet_counts, default=0.0)
    enemy_max_ships = max(enemy_ships, default=0.0)
    enemy_max_prod = max(enemy_prod, default=0.0)
    enemy_max_strength = max(enemy_strength, default=0.0)

    my_fleet_ships = 0.0
    if hasattr(obs, "f_alive") and hasattr(obs, "f_owner") and hasattr(obs, "f_ships"):
        f_mask = obs.f_alive & (obs.f_owner.to(torch.long) == pid)
        if bool(f_mask.any()):
            my_fleet_ships = float(obs.f_ships.to(obs.ships.dtype)[f_mask].sum().item())

    leader_prod_gap = max(enemy_max_prod - my_prod, 0.0)
    leader_ship_gap = max(enemy_max_ships - my_ships, 0.0)
    strength_ratio = my_strength / max(enemy_max_strength, 1e-6)
    clipped_ratio = max(0.0, min(strength_ratio, 3.0)) / 3.0

    prev = previous_stats or {}
    my_planet_delta = (my_planets - float(prev.get("my_planets", my_planets))) / float(total_planets)
    my_ship_delta = (my_ships - float(prev.get("my_ships", my_ships))) / 500.0
    enemy_prod_growth = (enemy_max_prod - float(prev.get("enemy_max_prod", enemy_max_prod))) / 100.0
    enemy_ship_growth = (enemy_max_ships - float(prev.get("enemy_max_ships", enemy_max_ships))) / 500.0

    values = [
        float(step) / 500.0,
        max(0.0, 500.0 - float(step)) / 500.0,
        float(pc) / 4.0,
        my_planets / float(total_planets),
        my_ships / 500.0,
        my_prod / 100.0,
        my_fleet_ships / 500.0,
        enemy_max_planets / float(total_planets),
        enemy_max_ships / 500.0,
        enemy_max_prod / 100.0,
        float(neutral_mask.sum().item()) / float(total_planets),
        my_strength / 100.0,
        enemy_max_strength / 100.0,
        clipped_ratio,
        leader_prod_gap / 100.0,
        leader_ship_gap / 500.0,
        1.0 if pc >= 4 else 0.0,
        1.0 if pc == 2 else 0.0,
        my_planet_delta,
        my_ship_delta,
        enemy_prod_growth,
        enemy_ship_growth,
        1.0 if int(step) > 400 else 0.0,
        1.0 if 150 < int(step) <= 400 else 0.0,
    ]
    return torch.tensor(values, dtype=dtype)


def _feature_stats(obs, prod: Tensor, player_id: int, player_count: int) -> dict:
    prod_by_owner, ships_by_owner, _ = _owner_strengths(obs, prod, int(player_count))
    owner = obs.owner_abs.to(torch.long)
    alive = obs.alive
    pid = int(player_id)
    enemy_prod = []
    enemy_ships = []
    for oid in range(int(player_count)):
        if oid == pid:
            continue
        enemy_prod.append(float(prod_by_owner[oid].item()))
        enemy_ships.append(float(ships_by_owner[oid].item()))
    return {
        "my_planets": float((alive & (owner == pid)).sum().item()),
        "my_ships": float(ships_by_owner[pid].item()) if 0 <= pid < int(player_count) else 0.0,
        "enemy_max_prod": max(enemy_prod, default=0.0),
        "enemy_max_ships": max(enemy_ships, default=0.0),
    }


def get_padded_history(feature_history: list[Tensor], seq_len: int = GRU_SEQUENCE_LEN) -> Tensor:
    if not feature_history:
        frame = torch.zeros(GRU_FEATURE_DIM, dtype=torch.float32)
        frames = [frame for _ in range(seq_len)]
    else:
        recent = [f.detach().to(dtype=torch.float32, device="cpu") for f in feature_history[-seq_len:]]
        pad = [recent[0].clone() for _ in range(max(0, seq_len - len(recent)))]
        frames = pad + recent
    return torch.stack(frames, dim=0).unsqueeze(0).cpu()


def _clamp_config_values(
    base_config: ProducerLiteConfig,
    *,
    delta_roi: float,
    delta_waves: float,
    ffa_multiplier: float,
) -> ProducerLiteConfig:
    roi = max(1.25, min(float(base_config.roi_threshold) + float(delta_roi), 1.85))
    waves = int(round(float(base_config.max_waves_per_turn) + float(delta_waves)))
    waves = max(3, min(waves, 7))
    ffa_mult = max(0.75, min(float(ffa_multiplier), 1.25))
    ffa_leader = max(0.0, min(float(base_config.ffa_leader_attack_bonus) * ffa_mult, 0.035))
    ffa_target = max(0.0, min(float(base_config.ffa_target_prod_bonus) * ffa_mult, 0.070))
    return dataclasses.replace(
        base_config,
        roi_threshold=roi,
        max_waves_per_turn=waves,
        ffa_leader_attack_bonus=ffa_leader,
        ffa_target_prod_bonus=ffa_target,
    )


def heuristic_strategy_controller(base_config: ProducerLiteConfig, seq: Tensor, player_count: int, step: int) -> ProducerLiteConfig:
    latest = seq[0, -1].detach().cpu()
    strength_ratio = float(latest[13].item()) * 3.0
    my_strength = float(latest[11].item()) * 100.0
    enemy_max_strength = float(latest[12].item()) * 100.0

    if strength_ratio >= 1.20:
        delta_roi = 0.05
        delta_waves = -1.0
        ffa_multiplier = 0.85
    elif strength_ratio >= 0.85:
        delta_roi = 0.0
        delta_waves = 0.0
        ffa_multiplier = 1.0
    else:
        delta_roi = -0.06
        delta_waves = 1.0
        ffa_multiplier = 1.10

    if int(player_count) >= 4 and enemy_max_strength > 1.25 * max(my_strength, 1e-6):
        ffa_multiplier = max(ffa_multiplier, 1.15)
    if int(step) > 430 and strength_ratio < 1.0:
        delta_roi -= 0.04
        delta_waves += 1.0
    return _clamp_config_values(
        base_config,
        delta_roi=delta_roi,
        delta_waves=delta_waves,
        ffa_multiplier=ffa_multiplier,
    )


def _load_gru_controller() -> TinyGRUStrategyController | None:
    global _GRU_CONTROLLER
    if _GRU_CONTROLLER is not None:
        return _GRU_CONTROLLER
    if not (USE_GRU_CONTROLLER and GRU_WEIGHTS_AVAILABLE):
        return None
    model = TinyGRUStrategyController(input_dim=GRU_FEATURE_DIM)
    try:
        if GRU_EMBEDDED_STATE_DICT is not None:
            state = GRU_EMBEDDED_STATE_DICT
        else:
            weights_path = os.path.join(_HERE, "gru_controller.pt")
            if not os.path.exists(weights_path):
                return None
            state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _GRU_CONTROLLER = model.cpu()
        return _GRU_CONTROLLER
    except Exception:
        return None


def apply_strategy_controller(
    base_config: ProducerLiteConfig,
    seq: Tensor,
    obs,
    player_id: int,
    player_count: int,
    step: int,
) -> ProducerLiteConfig:
    model = _load_gru_controller()
    if model is None:
        return heuristic_strategy_controller(base_config, seq, int(player_count), int(step))
    try:
        with torch.no_grad():
            out = model(seq.cpu()).reshape(-1).detach().cpu()
        delta_roi = max(-0.12, min(float(out[0].item()), 0.15))
        delta_waves = round(max(-1.0, min(float(out[1].item()), 1.0)))
        ffa_multiplier = max(0.75, min(float(out[2].item()), 1.25))
        return _clamp_config_values(
            base_config,
            delta_roi=delta_roi,
            delta_waves=delta_waves,
            ffa_multiplier=ffa_multiplier,
        )
    except Exception:
        return heuristic_strategy_controller(base_config, seq, int(player_count), int(step))


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair, lets the
    original scoring code rank them, and greedily fires the best waves.
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                            # [C]
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0 or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        strength = torch.zeros(int(player_count), dtype=dtype, device=device)
        for oid in range(int(player_count)):
            owned_by = obs.alive & (owner == oid)
            strength[oid] = prod[owned_by].to(dtype).sum() + 0.025 * obs.ships.to(dtype)[owned_by].sum()
        my_strength = strength[pid] if 0 <= pid < int(player_count) else torch.zeros((), dtype=dtype, device=device)
        target_owner = owner[cand_tgt_slot.clamp(0, P - 1)]
        enemy_target = (target_owner >= 0) & (target_owner != pid) & (~cand_is_def)
        safe_owner = target_owner.clamp(0, int(player_count) - 1)
        target_strength = strength[safe_owner]
        leader_delta = (target_strength - my_strength).clamp(min=0.0)
        target_prod = prod[cand_tgt_slot.clamp(0, P - 1)].to(dtype)
        ffa_bonus = (
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * target_prod
        )
        score = score + torch.where(enemy_target, ffa_bonus, torch.zeros_like(score))
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement, plan single-size waves, and emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    step = _step_from_tensors(obs_tensors)
    features = extract_global_features(
        obs, int(obs.player_id), int(player_count), step, movement.planet_prod,
        previous_stats=getattr(memory, "last_feature_stats", None),
    )
    memory.feature_history.append(features)
    if len(memory.feature_history) > GRU_SEQUENCE_LEN:
        memory.feature_history = memory.feature_history[-GRU_SEQUENCE_LEN:]
    seq = get_padded_history(memory.feature_history, seq_len=GRU_SEQUENCE_LEN)
    config = apply_strategy_controller(
        config, seq, obs, int(obs.player_id), int(player_count), step,
    )
    memory.last_feature_stats = _feature_stats(
        obs, movement.planet_prod, int(obs.player_id), int(player_count),
    )

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    ffa_leader_attack_bonus=0.015,
    ffa_target_prod_bonus=0.030,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.feature_history: list[Tensor] = []
        self.last_feature_stats: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.feature_history = []
        self.last_feature_stats = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.feature_history = []
            mem.last_feature_stats = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
