#!/usr/bin/env python3
"""Self-play PPO for Orbit Wars (strategy/rl_strategy.md §5 phase 2).

Drives orbit_wars games manually with the learning policy in seat 0 and opponents
(frozen policy snapshots) in the other seats, collects seat-0 trajectories, applies
the potential-based shaped reward (rl/reward.py), and does a clipped-PPO update with
GAE. A snapshot of the policy is periodically added to the opponent pool (a light
PFSP league) to keep training non-stationary and avoid cycles.

This is a CPU prototype to validate the loop; for real training warm-start from a
BC checkpoint (rl/bc_train.py) and scale games/iters (and ideally move the hot loop
into orbit_lite's batched sim — §6).

    python rl/selfplay.py --iters 3 --games 4 --steps 80   # quick loop check
    python rl/selfplay.py --init training/bc_policy.pt --iters 200 --games 16 --players 4
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
_s = os.dup(1); os.dup2(2, 1)
try:
    from kaggle_environments import make
finally:
    sys.stdout.flush(); os.dup2(_s, 1); os.close(_s)

from features import build_features          # noqa: E402
from policy import PlanetPolicy, SHIP_BUCKETS  # noqa: E402
from reward import shaped_rewards             # noqa: E402
from aim import lead_aim                      # noqa: E402

GAMMA, LAM, CLIP, VF, ENT = 0.997, 0.95, 0.2, 0.5, 0.01


def decide(policy, obs, pid, n, deterministic=False):
    """Run policy on one obs; return (moves, record) where record has training tensors."""
    f = build_features(obs, pid, n)
    S = f["self"].shape[0]
    if S == 0:
        return [], None
    t = lambda a: torch.tensor(a)
    sf, cf, gf, cm = t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"])
    a = policy.act(sf, cf, gf, cm, deterministic=deterministic)
    mine = [p for p in obs["planets"] if p[1] == pid]
    id_to_p = {p[0]: p for p in obs["planets"]}
    moves = []
    for si in range(S):
        tgt = int(a["target"][si])
        if tgt == 0:
            continue
        cid = int(f["cand_target_id"][si, tgt - 1])
        if cid < 0 or cid not in id_to_p:
            continue
        src, tp = mine[si], id_to_p[cid]
        ships = int(max(1, math.floor(src[5] * SHIP_BUCKETS[int(a["ship"][si])])))
        ang, _ax, _ay, _eta, sun = lead_aim(src[2], src[3], tp[2], tp[3], tp[4],
                                            ships, float(obs.get("angular_velocity", 0.0)))
        if sun:
            continue
        moves.append([src[0], float(ang), ships])
    rec = {"sf": sf, "cf": cf, "gf": gf, "cm": cm,
           "tgt": a["target"], "ship": a["ship"],
           "logp": a["logp"].sum().detach(), "value": a["value"].detach()}
    return moves, rec


def play_episode(policy, opponents, n, steps, seed):
    """Manual game: learner=seat0, opponents fill the rest. Return (records, obs_seq, placement)."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": steps}, debug=False)
    env.reset(n)
    records, obs_seq = [], []
    agents = [policy] + [opponents[i % len(opponents)] for i in range(n - 1)]
    while not env.done:
        actions = []
        for i in range(n):
            obs_i = dict(env.state[i]["observation"])
            mv, rec = decide(agents[i], obs_i, i, n, deterministic=(i != 0))
            actions.append(mv)
            if i == 0:
                obs_seq.append(obs_i)
                if rec is not None:
                    records.append(rec)
        env.step(actions)
    final = env.state[0]["observation"]
    obs_seq.append(dict(final))
    # placement of seat 0
    rewards = [s["reward"] for s in env.state]
    ships = []
    for p in range(n):
        ships.append(sum(x[5] for x in final["planets"] if x[1] == p)
                     + sum(x[5] for x in final["fleets"] if x[1] == p))
    order = sorted(range(n), key=lambda i: (-(rewards[i] or 0), -ships[i]))
    placement = order.index(0) + 1
    return records, obs_seq, placement


def compute_gae(records, rewards):
    """Attach advantage + return to each record (record count may be < reward count
    when some steps had no owned sources; align to the last len(records) rewards)."""
    R = rewards[-len(records):] if records else []
    vals = [r["value"].item() for r in records] + [0.0]
    adv, gae = [0.0] * len(records), 0.0
    for t in reversed(range(len(records))):
        delta = R[t] + GAMMA * vals[t + 1] - vals[t]
        gae = delta + GAMMA * LAM * gae
        adv[t] = gae
    ret = [adv[t] + vals[t] for t in range(len(records))]
    return adv, ret


def ppo_update(policy, opt, batch, epochs=4):
    advs = torch.tensor([b["adv"] for b in batch])
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    rets = torch.tensor([b["ret"] for b in batch])
    old_logp = torch.stack([b["logp"] for b in batch])
    last = 0.0
    for _ in range(epochs):
        idx = torch.randperm(len(batch))
        for j in idx:
            b = batch[int(j)]
            logp, value, ent = policy.evaluate(b["sf"], b["cf"], b["gf"], b["cm"], b["tgt"], b["ship"])
            logp = logp.sum()
            ratio = torch.exp(logp - old_logp[j])
            a = advs[j]
            s1, s2 = ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a
            pol_loss = -torch.min(s1, s2)
            val_loss = F.mse_loss(value, rets[j])
            loss = pol_loss + VF * val_loss - ENT * ent
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.detach())
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="warm-start policy .pt (e.g. BC checkpoint)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(ROOT / "training" / "selfplay_policy.pt"))
    args = ap.parse_args()

    policy = PlanetPolicy()
    if args.init and Path(args.init).exists():
        policy.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    pool = [copy.deepcopy(policy)]  # opponent league (frozen snapshots)

    for it in range(args.iters):
        batch, wins, places = [], 0, []
        for g in range(args.games):
            opp = [pool[np.random.randint(len(pool))] for _ in range(args.players - 1)]
            recs, obs_seq, place = play_episode(policy, opp, args.players, args.steps, seed=1000 + it * 100 + g)
            rs = shaped_rewards(obs_seq, 0, args.players, place, gamma=GAMMA)
            adv, ret = compute_gae(recs, rs)
            for k, r in enumerate(recs):
                r["adv"], r["ret"] = adv[k], ret[k]
            batch += recs
            places.append(place); wins += (place == 1)
        loss = ppo_update(policy, opt, batch) if batch else float("nan")
        print(f"iter {it:>3}: games={args.games} avg_place={np.mean(places):.2f} "
              f"win%={100*wins/args.games:.0f} steps_collected={len(batch)} loss={loss:.4f}", flush=True)
        if (it + 1) % 10 == 0:           # grow the league
            pool.append(copy.deepcopy(policy))
            if len(pool) > 6:
                pool.pop(0)
    torch.save(policy.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
