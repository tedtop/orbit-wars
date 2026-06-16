#!/usr/bin/env python3
# ============================================================================
# ⚠️  PARTIAL DEAD END — the BC-clone league (--clones) is abandoned (2026-06-15).
#     The base self-play loop is fine infra; the clone-opponent path is not.
#
# --clones loads training/clone_*.pt — neural policies behavior-cloned from the
# top public players (see rl/bc_train.py) — as fixed PPO league opponents. The
# whole neural track hit a BC ceiling and lost 0–16 to comet_reaper (the engine),
# so training against those clones cannot produce an engine-beating bot.
# Verdict: abandoned. See archive/experiments/comet_reaper_forks/README.md.
# The DEAD-END clone code below is fenced with `--- BC-clone league ---` markers.
# ============================================================================
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


LAUNCH_THRESH = 0.5  # opponent launch-gate (matches rl_agent deploy decode)


def _emit_moves(targets, ships, f, obs, pid):
    """Turn per-source (target-index, ship-bucket) into lead-aimed [planet,angle,ships]."""
    mine = [p for p in obs["planets"] if p[1] == pid]
    id_to_p = {p[0]: p for p in obs["planets"]}
    omega = float(obs.get("angular_velocity", 0.0))
    moves = []
    for si in range(len(mine)):
        tgt = int(targets[si])
        if tgt == 0:                       # 0 = no-op
            continue
        cid = int(f["cand_target_id"][si, tgt - 1])
        if cid < 0 or cid not in id_to_p:
            continue
        src, tp = mine[si], id_to_p[cid]
        n_ships = int(max(1, math.floor(src[5] * SHIP_BUCKETS[int(ships[si])])))
        ang, _x, _y, _e, sun = lead_aim(src[2], src[3], tp[2], tp[3], tp[4], n_ships, omega)
        if not sun:
            moves.append([src[0], float(ang), n_ships])
    return moves


def decide(policy, obs, pid, n, deterministic=False):
    """Return (moves, record).

    - opponent (deterministic=True): play ACTIVELY via the launch-gate decode
      (gate on total launch probability, take the best target) — same as rl_agent.
      argmax-over-{noop,K} would collapse to no-op and make opponents passive.
    - learner (deterministic=False): SAMPLE the policy (for a valid PPO log-prob).
    """
    f = build_features(obs, pid, n)
    S = f["self"].shape[0]
    if S == 0:
        return [], None
    t = lambda a: torch.tensor(a)
    sf, cf, gf, cm = t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"])
    out = policy.forward(sf, cf, gf, cm)

    if deterministic:                       # opponent: active launch-gate, no training record
        tprob = torch.softmax(out.target_logits, dim=-1)        # [S, K+1]
        ship_choice = out.ship_logits.argmax(-1)
        targets = [(int(tprob[si, 1:].argmax()) + 1) if float(1 - tprob[si, 0]) >= LAUNCH_THRESH else 0
                   for si in range(S)]
        return _emit_moves(targets, ship_choice, f, obs, pid), None

    # learner: sample for PPO
    tdist = torch.distributions.Categorical(logits=out.target_logits)
    sdist = torch.distributions.Categorical(logits=out.ship_logits)
    tgt = tdist.sample(); ship = sdist.sample()
    logp = (tdist.log_prob(tgt) + sdist.log_prob(ship)).sum().detach()
    moves = _emit_moves([int(x) for x in tgt], ship, f, obs, pid)
    rec = {"sf": sf, "cf": cf, "gf": gf, "cm": cm,
           "tgt": tgt, "ship": ship, "logp": logp, "value": out.value.detach()}
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
    ap.add_argument("--clones", action="store_true",
                    help="[DEAD END] add training/clone_*.pt (BC'd top bots) as fixed league "
                         "opponents — the neural track lost 0–16 to the engine; see module header")
    ap.add_argument("--out", default=str(ROOT / "training" / "selfplay_policy.pt"))
    args = ap.parse_args()

    policy = PlanetPolicy()
    if args.init and Path(args.init).exists():
        policy.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # --- BC-clone league (DEAD END — see module header) -----------------------
    # Fixed opponents: BC'd clones of real top bots (Jake Will, Xiangyu Liu, ...).
    # Same architecture -> load in-process, no subprocess. These never change.
    # Abandoned: the cloned policies lost 0–16 to comet_reaper, so this league
    # never pushed the learner toward engine strength. Off unless --clones.
    import glob
    clones = []
    if args.clones:
        for cp in sorted(glob.glob(str(ROOT / "training" / "clone_*.pt"))):
            c = PlanetPolicy()
            c.load_state_dict(torch.load(cp, map_location="cpu"))
            c.eval()
            clones.append(c)
            print(f"league opponent: {Path(cp).stem}")
    # --- end BC-clone league --------------------------------------------------
    self_snaps = [copy.deepcopy(policy)]   # frozen past selves (capped)

    for it in range(args.iters):
        batch, wins, places = [], 0, []
        pool = clones + self_snaps         # learner trains vs clones + its own past
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
        if (it + 1) % 10 == 0:           # add a fresh self-snapshot (clones stay fixed)
            self_snaps.append(copy.deepcopy(policy))
            if len(self_snaps) > 4:
                self_snaps.pop(0)
    torch.save(policy.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
