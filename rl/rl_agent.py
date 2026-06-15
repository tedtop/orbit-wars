#!/usr/bin/env python3
"""Inference wrapper: a trained PlanetPolicy -> a Kaggle Orbit Wars agent.

Loads `training/bc_policy.pt`, and each turn: builds features, runs the policy
deterministically, and decodes per-source (target, ship-bucket) choices into
`[from_planet, angle, ships]` launches. Aim here is a simple aim-at-current-
position; swap in orbit_lite's `intercept_angle` lead-aim for moving targets.

This closes the BC loop: data -> bc_train.py -> bc_policy.pt -> rl_agent.agent.
(The local-data checkpoint plays weakly — it just proves the deployment path.)

Run check (1 game vs random in the gym):  python rl/rl_agent.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features   # noqa: E402
from policy import PlanetPolicy, SHIP_BUCKETS   # noqa: E402
from aim import lead_aim   # noqa: E402

_POLICY = None
_WEIGHTS = os.environ.get("RL_POLICY", str(Path(__file__).resolve().parent.parent / "training" / "bc_policy.pt"))


def _load():
    global _POLICY
    if _POLICY is None:
        _POLICY = PlanetPolicy()
        if Path(_WEIGHTS).exists():
            _POLICY.load_state_dict(torch.load(_WEIGHTS, map_location="cpu"))
        _POLICY.eval()
    return _POLICY


def _n_players(obs, pid):
    mx = pid
    for p in obs.get("planets", []):
        mx = max(mx, p[1])
    for f in obs.get("fleets", []):
        mx = max(mx, f[1])
    return 4 if mx >= 2 else 2


def agent(obs, config=None):
    try:
        pid = int(obs["player"] if isinstance(obs, dict) else obs.player)
        obs = dict(obs) if not isinstance(obs, dict) else obs
        n = _n_players(obs, pid)
        f = build_features(obs, pid, n)
        S = f["self"].shape[0]
        if S == 0:
            return []
        pol = _load()
        t = lambda a: torch.tensor(a)
        out = pol.forward(t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"]))
        # Launch-gate decode: argmax over {no-op, K targets} is biased toward no-op
        # because launch probability is split across K candidates. Instead launch when
        # the TOTAL target probability beats a threshold, then take the best target.
        tprob = torch.softmax(out.target_logits, dim=-1)          # [S, K+1]
        ship_choice = out.ship_logits.argmax(-1)                  # [S]
        thresh = float(os.environ.get("RL_LAUNCH_THRESH", "0.5"))
        omega = float(obs.get("angular_velocity", 0.0))
        mine = [p for p in obs["planets"] if p[1] == pid]
        id_to_planet = {p[0]: p for p in obs["planets"]}
        moves = []
        for si in range(S):
            p_launch = float(1.0 - tprob[si, 0])
            if p_launch < thresh:
                continue
            ci = int(tprob[si, 1:].argmax())                       # best candidate
            cand_id = int(f["cand_target_id"][si, ci])
            if cand_id < 0 or cand_id not in id_to_planet:
                continue
            src, tp = mine[si], id_to_planet[cand_id]
            ships = int(max(1, math.floor(src[5] * SHIP_BUCKETS[int(ship_choice[si])])))
            if ships < 1:
                continue
            # lead-aim the (orbiting) target so the fleet actually connects; skip if
            # the only line of fire crosses the sun (the fleet would be destroyed).
            angle, _ax, _ay, _eta, sun = lead_aim(
                src[2], src[3], tp[2], tp[3], tp[4], ships, omega)
            if sun:
                continue
            moves.append([src[0], float(angle), ships])
        return moves
    except Exception:
        return []


def _check():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gym"))
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": 80})
    env.run([agent, "random"])
    rw = [s.reward for s in env.steps[-1]]
    acts = sum(1 for st in env.steps if st[0].get("action"))
    print(f"rl_agent (BC policy) vs random | rewards={rw} | action-turns={acts}/{len(env.steps)}")
    print("deployment path OK" if acts > 0 else "WARNING: emitted no actions")


if __name__ == "__main__":
    _check()
