
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time as _time_mod
from dataclasses import dataclass

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
from orbit_lite.garrison_launch import LaunchSet


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs."""

    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    # schmeekler: prefer static (non-rotating) periphery planets
    static_target_bonus: float = 1.5
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    terminal_phase_turns: int = 40
    terminal_roi_threshold: float = 1.0
    terminal_max_waves_per_turn: int = 9
    terminal_enable_regroup: bool = False
    comet_evac_steps: int = 4


TOTAL_STEPS = 500


def _apply_phase_config(config: ProducerLiteConfig, step: int) -> ProducerLiteConfig:
    if int(step) >= TOTAL_STEPS - int(config.terminal_phase_turns):
        return dataclasses.replace(
            config,
            roi_threshold=float(config.terminal_roi_threshold),
            max_waves_per_turn=int(config.terminal_max_waves_per_turn),
            enable_regroup=bool(config.terminal_enable_regroup),
        )
    return config


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


def _build_opponent_launch_set(obs, cache, pid: int, player_count: int, config: ProducerLiteConfig,
                                H: int, device, dtype):
    """Cheap 1-ply opponent model: each opp planet fires ~60% ships to its nearest non-owned target.

    Returns a flat (non-batched) LaunchSet or None if the opponent has no valid moves.
    The LaunchSet.owner field identifies which player each launch belongs to.
    """
    srcs, tgts, ships_l, etas, owners, valids = [], [], [], [], [], []
    P = obs.P

    for opp_id in range(int(player_count)):
        if opp_id == pid:
            continue
        opp_mask = (obs.owner_abs == float(opp_id)) & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
        if not bool(opp_mask.any()):
            continue
        tgt_mask = obs.alive & (obs.owner_abs != float(opp_id))
        if not bool(tgt_mask.any()):
            continue

        s_idx = opp_mask.nonzero(as_tuple=False).flatten()   # [Sopp]
        t_idx = tgt_mask.nonzero(as_tuple=False).flatten()   # [Topp]
        Sopp = s_idx.shape[0]

        d = cache.cross_dist[0][s_idx][:, t_idx]             # [Sopp, Topp]
        near_local = d.argmin(dim=1)                          # [Sopp]
        near_tgt = t_idx[near_local]                          # [Sopp]

        src_ships = obs.ships[s_idx].to(dtype)
        send = (src_ships * 0.6).floor().clamp(min=float(config.min_ships_to_launch), max=src_ships)
        dist = cache.cross_dist[0][s_idx, near_tgt].to(dtype)
        speed = fleet_speed(send.clamp(min=1e-6))
        eta = (dist / speed.clamp(min=1e-6)).ceil().clamp(min=1.0, max=float(H)).to(dtype)
        valid = (send >= float(config.min_ships_to_launch)) & (s_idx != near_tgt)

        srcs.append(s_idx)
        tgts.append(near_tgt)
        ships_l.append(send)
        etas.append(eta)
        owners.append(torch.full((Sopp,), opp_id, dtype=torch.long, device=device))
        valids.append(valid)

    if not srcs:
        return None

    return LaunchSet(
        source_slots=torch.cat(srcs),
        target_slots=torch.cat(tgts),
        ships=torch.cat(ships_l),
        eta=torch.cat(etas),
        owner=torch.cat(owners),
        valid=torch.cat(valids),
    )


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
    """1-ply greedy planner + 2-ply opponent lookahead (comet_reaper_mcts).

    Standard schmeekler 1-ply flow-diff scoring is used as the base. For the
    top-K candidates, a cheap opponent reply (60%-drain → nearest non-owned) is
    combined into the same LaunchSet and re-scored with the exact flow scorer.
    The de-meaned correction is added to the 1-ply scores so the static bonus
    and ROI threshold remain intact while re-ranking by opponent-resistance.
    Falls back to 1-ply silently on any error or if time budget is exceeded.
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
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)

    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )
    K = int(floor.shape[-1])

    sizes = drain.view(S, 1).expand(S, T).floor()

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

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
    cand_is_def = target_is_mine[cand_tgt_short]

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
    )

    # schmeekler: nudge toward STATIC (non-rotating) periphery targets
    _sb = float(getattr(config, "static_target_bonus", 0.0))
    if _sb != 0.0:
        try:
            pl = obs_tensors["planets"]
            cx = pl[:, 2].to(dtype); cy = pl[:, 3].to(dtype); pr = pl[:, 4].to(dtype)
            is_static = (torch.sqrt((cx - 50.0) ** 2 + (cy - 50.0) ** 2) + pr) >= 50.0
            static_t = is_static[cand_tgt_slot.clamp(0, P - 1)]
            score = score + torch.where(static_t, torch.full_like(score, _sb), torch.zeros_like(score))
        except Exception:
            pass

    # ---- 2-ply opponent lookahead ----------------------------------------
    # For top-K candidates: build combined LaunchSet (our move + cheap opp reply)
    # and re-score with the exact flow scorer. The de-meaned correction is added
    # to the 1-ply+bonus scores to re-rank by opponent-resistance while keeping
    # the static bonus and ROI threshold intact.
    _search_depth = int(os.environ.get("SEARCH_DEPTH", "2"))
    if _search_depth >= 2:
        try:
            _t0 = _time_mod.time()
            _budget = float(os.environ.get("SEARCH_BUDGET_MS", "700")) / 1000.0
            _topk = int(os.environ.get("SEARCH_TOPK", "30"))

            opp_ls = _build_opponent_launch_set(obs, cache, pid, player_count, config, K_eta, device, dtype)

            valid_idx = cand_valid.nonzero(as_tuple=False).flatten()
            n_eval = min(_topk, int(valid_idx.numel()))

            if n_eval > 0 and (_time_mod.time() - _t0) < _budget:
                top_order = torch.topk(score[valid_idx], n_eval, largest=True).indices
                sel = valid_idx[top_order]   # [n_eval] — indices into C candidates

                our_src = cand_src[sel]                               # [n_eval, 1]
                our_tgt = cand_tgt_slot[sel].unsqueeze(-1)            # [n_eval, 1]
                our_ships = cand_send[sel]                            # [n_eval, 1]
                our_eta_t = cand_eta[sel]                             # [n_eval, 1]
                our_active = cand_active[sel]                         # [n_eval, 1]
                our_owner = torch.full_like(our_src, pid)             # [n_eval, 1]

                if opp_ls is not None and (_time_mod.time() - _t0) < _budget:
                    Lopp = int(opp_ls.source_slots.shape[0])
                    opp_src_e = opp_ls.source_slots.unsqueeze(0).expand(n_eval, Lopp)
                    opp_tgt_e = opp_ls.target_slots.unsqueeze(0).expand(n_eval, Lopp)
                    opp_shp_e = opp_ls.ships.unsqueeze(0).expand(n_eval, Lopp)
                    opp_eta_e = opp_ls.eta.unsqueeze(0).expand(n_eval, Lopp)
                    opp_own_e = opp_ls.owner.unsqueeze(0).expand(n_eval, Lopp)
                    opp_val_e = opp_ls.valid.unsqueeze(0).expand(n_eval, Lopp)

                    comb = LaunchSet(
                        source_slots=torch.cat([our_src, opp_src_e], dim=1),
                        target_slots=torch.cat([our_tgt, opp_tgt_e], dim=1),
                        ships=torch.cat([our_ships, opp_shp_e], dim=1),
                        eta=torch.cat([our_eta_t, opp_eta_e], dim=1),
                        owner=torch.cat([our_owner, opp_own_e], dim=1),
                        valid=torch.cat([our_active, opp_val_e], dim=1),
                    )
                else:
                    comb = LaunchSet(
                        source_slots=our_src, target_slots=our_tgt, ships=our_ships,
                        eta=our_eta_t, owner=our_owner, valid=our_active,
                    )

                if (_time_mod.time() - _t0) < _budget:
                    score_2ply = score_candidates(
                        garrison_status, prod=prod, alive_by_step=alive_by_step,
                        player_count=int(player_count), launches=comb, player_id=pid,
                    )  # [n_eval]

                    # De-meaned correction: adds the DIFFERENTIAL opponent-pressure signal
                    # without shifting the absolute score level (preserves ROI threshold).
                    correction = score_2ply - score_2ply.mean()
                    score = score.clone()
                    score[sel] = score[sel] + correction
        except Exception:
            pass  # silently fall back to 1-ply + static-bonus scores
    # ----------------------------------------------------------------------

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

    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
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


CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        base = _config_for(mem.cached_player_count)
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        config = _apply_phase_config(base, step)
        _sb = os.environ.get("SCHMEEKLER_STATIC_BONUS")
        if _sb is not None:
            config = dataclasses.replace(config, static_target_bonus=float(_sb))
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Comet evacuation
# ---------------------------------------------------------------------------
_SUN_X, _SUN_Y, _SUN_R = 50.0, 50.0, 10.0


def _oget(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_comet_remaining(obs) -> dict:
    out: dict = {}
    try:
        for g in (_oget(obs, "comets", None) or []):
            if isinstance(g, dict):
                pids = g.get("planet_ids") or []
                paths = g.get("paths") or []
                idx = int(g.get("path_index", 0) or 0)
            else:
                pids = getattr(g, "planet_ids", None) or []
                paths = getattr(g, "paths", None) or []
                idx = int(getattr(g, "path_index", 0) or 0)
            for i, cid in enumerate(pids):
                path = paths[i] if i < len(paths) else (paths[0] if len(paths) else None)
                if path is not None:
                    out[int(cid)] = max(0, int(len(path)) - 1 - idx)
    except Exception:
        return {}
    return out


def _segment_clears_sun(x0, y0, x1, y1, margin: float = 1.5) -> bool:
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(x0 - _SUN_X, y0 - _SUN_Y) > _SUN_R + margin
    t = max(0.0, min(1.0, ((_SUN_X - x0) * dx + (_SUN_Y - y0) * dy) / l2))
    cx, cy = x0 + t * dx, y0 + t * dy
    return math.hypot(cx - _SUN_X, cy - _SUN_Y) > _SUN_R + margin


def _is_static_planet(p) -> bool:
    return math.hypot(float(p[2]) - 50.0, float(p[3]) - 50.0) + float(p[4]) >= 49.999


def _comet_evac_moves(obs, player_id, moves, remaining, evac_steps):
    try:
        base = [list(m) for m in (moves or [])]
        if not remaining:
            return base
        planets = _oget(obs, "planets", None) or []
        comet_ids = set(int(c) for c in (_oget(obs, "comet_planet_ids", None) or []))
        by_id = {int(p[0]): p for p in planets}
        committed: dict = {}
        for m in base:
            committed[int(m[0])] = committed.get(int(m[0]), 0) + int(m[2])
        own_static = [p for p in planets if int(p[1]) == player_id and int(p[0]) not in comet_ids and _is_static_planet(p)]
        own_orbit = [p for p in planets if int(p[1]) == player_id and int(p[0]) not in comet_ids and not _is_static_planet(p)]
        others = [p for p in planets if int(p[0]) not in comet_ids and int(p[1]) != player_id]
        for cid, rem in remaining.items():
            if int(rem) > int(evac_steps):
                continue
            p = by_id.get(int(cid))
            if p is None or int(p[1]) != player_id:
                continue
            avail = int(p[5]) - committed.get(int(cid), 0)
            if avail < 1:
                continue
            px, py = float(p[2]), float(p[3])
            best = None
            pools = (
                sorted(own_static, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(own_orbit, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(others, key=lambda q: (float(q[5]) >= avail, (q[2] - px) ** 2 + (q[3] - py) ** 2)),
            )
            for pool in pools:
                for q in pool:
                    if int(q[0]) == int(cid):
                        continue
                    if _segment_clears_sun(px, py, float(q[2]), float(q[3])):
                        best = q
                        break
                if best is not None:
                    break
            if best is None:
                continue
            ang = math.atan2(float(best[3]) - py, float(best[2]) - px)
            base.append([int(cid), float(ang), int(avail)])
            committed[int(cid)] = committed.get(int(cid), 0) + int(avail)
        return base
    except Exception:
        return moves


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """comet_reaper_mcts: schmeekler + 2-ply opponent lookahead via exact flow scorer.

    The 2-ply search accounts for the opponent's likely response (cheap 1-ply
    safe-drain model) using the same exact flow-diff scorer as the base engine.
    This directly fixes comet_reaper's do-nothing blind spot without rollouts
    (the failed approach) or re-running the full orbital forward model.
    Falls back to pure schmeekler on any error or budget overflow.
    """
    try:
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        with torch.no_grad():
            sparse_row = _RUNTIME.tensor_action(obs_tensors)
        moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
        remaining = _parse_comet_remaining(obs)
        return _comet_evac_moves(obs, player_id, moves, remaining, ProducerLiteConfig().comet_evac_steps)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
