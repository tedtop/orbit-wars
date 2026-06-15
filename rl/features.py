#!/usr/bin/env python3
"""Feature extraction: Orbit Wars observation -> policy input tensors.

Produces the three feature groups the factored per-source policy consumes
(strategy/rl_strategy.md §1-2), INCLUDING the 4P opponent-asymmetry features the
whole orbit_lite family ignores (our intended edge):

  global_features   : [G]              board/phase/standings
  self_features     : [S, F_self]      per owned (source) planet
  cand_features     : [S, K, F_cand]   per source x candidate target
  cand_mask         : [S, K] bool      legal/reachable candidates
  cand_target_id    : [S, K] int       planet id of each candidate (for action decode)

Aim is intentionally NOT learned — at action time the chosen (source, target) is
turned into an angle by the physics lead-aim solver. Here we only featurize.

Check:  python rl/features.py [replay.json] [player_id] [step]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CX = CY = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
K_CANDIDATES = 8


def fleet_speed(ships: float) -> float:
    n = max(1.0, ships)
    return min(1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5, MAX_SPEED)


def _orbital_radius(x, y):
    return math.hypot(x - CX, y - CY)


def _owner_strength(obs, n_players):
    """Per-player strength = production + 0.025*ships (matches i-m-stronger's metric)."""
    s = [0.0] * n_players
    for p in obs.get("planets", []):
        o = p[1]
        if 0 <= o < n_players:
            s[o] += p[6] + 0.025 * p[5]
    for f in obs.get("fleets", []):
        o = f[1]
        if 0 <= o < n_players:
            s[o] += 0.025 * f[5]
    return s


def build_features(obs: dict, player_id: int, n_players: int, k: int = K_CANDIDATES):
    planets = obs.get("planets", [])
    step = obs.get("step", 0)
    strength = _owner_strength(obs, n_players)
    others = [i for i in range(n_players) if i != player_id]
    my_strength = strength[player_id]
    max_opp_strength = max((strength[i] for i in others), default=0.0)
    tot_strength = sum(strength) + 1e-6

    mine = [p for p in planets if p[1] == player_id]
    targets = [p for p in planets if p[1] != player_id]  # enemy + neutral

    # ---- global features ----
    n_alive = sum(1 for i in range(n_players) if strength[i] > 0)
    my_planets = len(mine)
    rank = 1 + sum(1 for i in others if strength[i] > my_strength)
    phase = step / 500.0
    global_features = np.array([
        phase,
        n_alive / 4.0,
        rank / max(1, n_players),
        my_strength / tot_strength,
        (my_strength - max_opp_strength) / tot_strength,
        my_planets / max(1, len(planets)),
        1.0 if n_players == 2 else 0.0,
        1.0 if n_players >= 4 else 0.0,
    ], dtype=np.float32)

    F_SELF, F_CAND = 8, 12
    S = len(mine)
    self_features = np.zeros((S, F_SELF), dtype=np.float32)
    cand_features = np.zeros((S, k, F_CAND), dtype=np.float32)
    cand_mask = np.zeros((S, k), dtype=bool)
    cand_target_id = np.full((S, k), -1, dtype=np.int64)

    for si, sp in enumerate(mine):
        _, _, sx, sy, sr, sships, sprod = sp[:7]
        nearest_enemy = min(
            (math.hypot(sx - t[2], sy - t[3]) for t in targets if t[1] != -1),
            default=100.0,
        )
        self_features[si] = [
            min(sships / 100.0, 5.0),
            sprod / 5.0,
            _orbital_radius(sx, sy) / 50.0,
            (math.hypot(sx - CX, sy - CY) - SUN_R) / 50.0,
            nearest_enemy / 100.0,
            1.0 if _orbital_radius(sx, sy) + sr < 50.0 else 0.0,  # rotating?
            min(sships / 20.0, 1.0),                              # launchable proxy
            sr / 5.0,
        ]
        # K nearest targets as candidates
        ranked = sorted(targets, key=lambda t: math.hypot(sx - t[2], sy - t[3]))[:k]
        for ci, t in enumerate(ranked):
            tid, towner, tx, ty, tr, tships, tprod = t[:7]
            dist = math.hypot(sx - tx, sy - ty)
            send = max(1.0, sships)
            eta = dist / fleet_speed(send)
            t_strength = strength[towner] if 0 <= towner < n_players else 0.0
            cand_target_id[si, ci] = tid
            cand_mask[si, ci] = sships >= 1.0 and dist > 1e-3
            cand_features[si, ci] = [
                1.0 if towner == -1 else 0.0,           # neutral
                1.0 if towner in others else 0.0,        # enemy
                min(tships / 100.0, 5.0),
                tprod / 5.0,
                dist / 100.0,
                min(eta / 50.0, 2.0),
                # 4P opponent-asymmetry features (our edge):
                (t_strength - my_strength) / tot_strength,    # leader-ness of target's owner
                (my_strength - t_strength) / tot_strength,    # weakness of target's owner
                1.0 if (towner in others and t_strength < 4.0) else 0.0,  # near-dead enemy
                min(tships / max(1.0, send), 2.0),            # defender/attacker ratio
                _orbital_radius(tx, ty) / 50.0,
                1.0 if tid in obs.get("comet_planet_ids", []) else 0.0,
            ]

    return {
        "global": global_features, "self": self_features,
        "cand": cand_features, "cand_mask": cand_mask, "cand_target_id": cand_target_id,
    }


def _check(replay_path: str, player_id: int = 0, step: int = 40):
    d = json.loads(Path(replay_path).read_text())
    n = len(d.get("info", {}).get("TeamNames", []))
    step = min(step, len(d["steps"]) - 1)
    obs = d["steps"][step][0]["observation"]
    f = build_features(obs, player_id, n)
    print(f"{Path(replay_path).name} | {n}p | player {player_id} | step {step}")
    print(f"  global {f['global'].shape}: {np.round(f['global'], 3)}")
    print(f"  self   {f['self'].shape}  (sources={f['self'].shape[0]})")
    print(f"  cand   {f['cand'].shape}  legal={int(f['cand_mask'].sum())}/{f['cand_mask'].size}")
    if f["self"].shape[0]:
        print(f"  sample self[0]:  {np.round(f['self'][0], 3)}")
        print(f"  sample cand[0,0]: {np.round(f['cand'][0,0], 3)}  -> target id {f['cand_target_id'][0,0]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args else next(iter(Path("replays").rglob("*.json")), None)
    if path is None:
        print("no replay found"); sys.exit(1)
    _check(str(path),
           int(args[1]) if len(args) > 1 else 0,
           int(args[2]) if len(args) > 2 else 40)
