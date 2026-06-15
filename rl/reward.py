#!/usr/bin/env python3
"""Shaped reward for Orbit Wars RL (see strategy/rl_strategy.md §3).

reward'(t) = [terminal reward at final step] + gamma * Phi(s_{t+1}) - Phi(s_t)

- Terminal: placement-shaped (1st=+1, 2nd=+0.3, 3rd=-0.3, 4th=-1.0; 2P: +1/-1).
- Phi: a production-weighted, normalized dominance potential. Potential-based
  shaping (Ng et al. 1999) telescopes over the episode, so it provides a dense
  per-step gradient WITHOUT changing the optimal policy (no reward hacking).

Usage as a check:
    python rl/reward.py path/to/replay.json [player_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Phi weights — production leads (it compounds; mirrors orbit_lite's 535-replay
# early-term calibration that weighs production ~5:1 over ships in 2P).
W_PROD, W_SHIPS, W_PLANETS = 0.6, 0.2, 0.2
EPS = 1e-6

# 4P placement -> terminal reward (punish last hardest; encodes the OpenSkill
# variance-management objective directly).
PLACE_REWARD_4P = {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}


def _owner_aggregates(obs: dict, n_players: int):
    """Return per-player (production, ships, planets) from one observation."""
    prod = [0.0] * n_players
    ships = [0.0] * n_players
    planets = [0] * n_players
    for p in obs.get("planets", []):
        owner = p[1]
        if 0 <= owner < n_players:
            ships[owner] += p[5]
            prod[owner] += p[6]
            planets[owner] += 1
    for f in obs.get("fleets", []):
        owner = f[1]
        if 0 <= owner < n_players:
            ships[owner] += f[5]
    return prod, ships, planets


def state_potential(obs: dict, player_id: int, n_players: int) -> float:
    """Phi(s): normalized production-weighted dominance for player_id in [-~1, ~1]."""
    prod, ships, planets = _owner_aggregates(obs, n_players)
    others = [i for i in range(n_players) if i != player_id]
    if not others:
        return 0.0
    tot_p = sum(prod) + EPS
    tot_s = sum(ships) + EPS
    tot_c = sum(planets) + EPS
    mean_opp_prod = sum(prod[i] for i in others) / len(others)
    best_opp_ships = max(ships[i] for i in others)
    mean_opp_planets = sum(planets[i] for i in others) / len(others)
    return (
        W_PROD * (prod[player_id] - mean_opp_prod) / tot_p
        + W_SHIPS * (ships[player_id] - best_opp_ships) / tot_s
        + W_PLANETS * (planets[player_id] - mean_opp_planets) / tot_c
    )


def terminal_reward(placement: int, n_players: int) -> float:
    if n_players <= 2:
        return 1.0 if placement == 1 else -1.0
    return PLACE_REWARD_4P.get(placement, -1.0)


def shaped_rewards(obs_seq: list[dict], player_id: int, n_players: int,
                   placement: int, gamma: float = 0.999) -> list[float]:
    """Per-step shaped reward over an episode (len = len(obs_seq) - 1)."""
    phis = [state_potential(o, player_id, n_players) for o in obs_seq]
    rewards = []
    T = len(obs_seq) - 1
    for t in range(T):
        shaping = gamma * phis[t + 1] - phis[t]
        r = shaping + (terminal_reward(placement, n_players) if t == T - 1 else 0.0)
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
def _placement_from_replay(d: dict, n_players: int) -> list[int]:
    """Final placement per player (1=best) from rewards + total-ship tiebreak."""
    rewards = d.get("rewards", [0] * n_players)
    last_obs = d["steps"][-1][0]["observation"]
    _, ships, planets = _owner_aggregates(last_obs, n_players)
    order = sorted(range(n_players), key=lambda i: (-(rewards[i] or 0), -ships[i]))
    place = [0] * n_players
    for rank, slot in enumerate(order):
        place[slot] = rank + 1
    return place


def _check(replay_path: str, player_id: int = 0):
    d = json.loads(Path(replay_path).read_text())
    teams = d.get("info", {}).get("TeamNames", [])
    n = len(teams)
    obs_seq = [step[0]["observation"] for step in d["steps"]]
    place = _placement_from_replay(d, n)
    rs = shaped_rewards(obs_seq, player_id, n, place[player_id])
    phi0 = state_potential(obs_seq[0], player_id, n)
    phiT = state_potential(obs_seq[-1], player_id, n)
    print(f"replay {Path(replay_path).name} | {n}p | player {player_id} = {teams[player_id]!r}")
    print(f"  placement: {place[player_id]}/{n}  terminal_reward: {terminal_reward(place[player_id], n):+.2f}")
    print(f"  Phi: start={phi0:+.3f} end={phiT:+.3f}  (rises if player dominates over the game)")
    print(f"  steps: {len(rs)}  sum(shaped)={sum(rs):+.3f}  mean|shaped|={sum(abs(x) for x in rs)/max(1,len(rs)):.4f}")
    print(f"  first 5 shaped rewards: {[round(x,4) for x in rs[:5]]}")
    print(f"  last 5 shaped rewards:  {[round(x,4) for x in rs[-5:]]} (last includes terminal)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default to any local replay
        cands = list(Path("replays").rglob("*.json"))
        if not cands:
            print("usage: python rl/reward.py <replay.json> [player_id]"); sys.exit(1)
        _check(str(cands[0]), int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    else:
        _check(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
