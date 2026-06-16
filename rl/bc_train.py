#!/usr/bin/env python3
# ============================================================================
# ⚰️  DEAD END — neural BC / self-play track (2026-06-15). Kept as infra; do not
#     expect it to beat the engine.
#
# This is "behavior cloning": train a neural PlanetPolicy to imitate the top
# public players (their recorded moves from prize-zone episodes). The resulting
# clones (training/clone_*.pt) were then used as PPO self-play opponents in
# rl/selfplay.py --clones. The cloned-from-humans policy hit a *BC ceiling* and
# lost 0–16 to comet_reaper (the orbit_lite engine). The engine is simply
# stronger than anything this neural track reaches.
#
# Verdict: abandoned. See archive/experiments/comet_reaper_forks/README.md.
# Left in place as a reusable BC loop / cautionary trail — re-run only if you
# have a genuinely new idea for clearing the BC ceiling.
# ============================================================================
"""Behavior cloning: train PlanetPolicy to imitate recorded moves.

Bridges replay data -> policy labels. For each (obs, action) example:
  - build features (rl/features.py) -> K candidates per owned source planet,
  - for each owned source, the label is which candidate it launched at (recovered
    by matching the recorded launch angle to the candidate's direction) + a ship
    bucket; sources that didn't launch are labelled no-op.
Then supervised cross-entropy on the target + ship heads.

This is BC phase 1 of strategy/rl_strategy.md §5. Point it at a prize-zone dataset
(`--min-rating 1500`) once `pipeline/pull_topbot_episodes.py` has run; it also works
on the local data to validate the loop end-to-end.

    python rl/bc_train.py --data training/moves_local.jsonl.gz --steps 300
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, K_CANDIDATES, fleet_speed   # noqa: E402
from policy import PlanetPolicy, SHIP_BUCKETS                     # noqa: E402

ANGLE_TOL = 0.35  # rad — match a launch angle to a candidate's direction


def _n_players(obs, player_id: int) -> int:
    mx = player_id  # a player can own nothing at a given step, so include their id
    for p in obs.get("planets", []):
        mx = max(mx, p[1])
    for f in obs.get("fleets", []):
        mx = max(mx, f[1])
    return 4 if mx >= 2 else 2


def label_example(obs: dict, action, player_id: int):
    """Return (features, target_labels[S], ship_labels[S], launched[S], matched, launches)."""
    n = _n_players(obs, player_id)
    f = build_features(obs, player_id, n)
    S = f["self"].shape[0]
    mine = [p for p in obs.get("planets", []) if p[1] == player_id]
    id_to_si = {p[0]: i for i, p in enumerate(mine)}
    tgt = np.zeros(S, dtype=np.int64)        # 0 = no-op
    ship = np.zeros(S, dtype=np.int64)
    launched = np.zeros(S, dtype=bool)
    matched = 0
    launches = 0
    for mv in (action or []):
        if not mv or len(mv) < 3:
            continue
        src_id, angle, ships = int(mv[0]), float(mv[1]), float(mv[2])
        launches += 1
        si = id_to_si.get(src_id)
        if si is None:
            continue
        sp = mine[si]
        sx, sy, sgar = sp[2], sp[3], sp[5]
        # match to the candidate whose current-position direction is closest to the angle
        best_ci, best_d = None, ANGLE_TOL
        for ci in range(K_CANDIDATES):
            tid = f["cand_target_id"][si, ci]
            if tid < 0 or not f["cand_mask"][si, ci]:
                continue
            tp = next((p for p in obs["planets"] if p[0] == tid), None)
            if tp is None:
                continue
            dir_to = math.atan2(tp[3] - sy, tp[2] - sx)
            dd = abs((dir_to - angle + math.pi) % (2 * math.pi) - math.pi)
            if dd < best_d:
                best_d, best_ci = dd, ci
        if best_ci is None:
            continue
        matched += 1
        tgt[si] = best_ci + 1   # +1 because index 0 is no-op
        launched[si] = True
        frac = ships / max(1.0, sgar)
        ship[si] = int(np.argmin([abs(frac - b) for b in SHIP_BUCKETS]))
    return f, tgt, ship, launched, matched, launches


def load_examples(path: str, min_rating: float, limit: int, team: str | None = None):
    out = []
    team_l = team.lower() if team else None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if team_l is not None and (r.get("team") or "").lower() != team_l:
                continue
            if min_rating > 0 and (r.get("rating") is None or r["rating"] < min_rating):
                continue
            if "obs" not in r:
                continue
            out.append((r["obs"], r.get("action") or [], int(r["player"])))
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "training" / "moves_local.jsonl.gz"))
    ap.add_argument("--min-rating", type=float, default=0.0)
    ap.add_argument("--team", default=None, help="clone ONE team (e.g. 'Jake Will') — filter to its moves")
    ap.add_argument("--limit", type=int, default=4000, help="max examples to load")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--noop-weight", type=float, default=0.25,
                    help="class weight for the no-op target (down-weight the dominant 'hold' "
                         "label so the policy learns to launch, not collapse to all-no-op)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "training" / "bc_policy.pt"))
    args = ap.parse_args()

    print(f"Loading examples from {args.data} (min_rating={args.min_rating}, team={args.team}) …")
    raw = load_examples(args.data, args.min_rating, args.limit, args.team)
    # Pre-label (skip examples with no owned sources)
    data, tot_match, tot_launch = [], 0, 0
    for obs, action, pid in raw:
        f, tgt, ship, launched, m, l = label_example(obs, action, pid)
        tot_match += m; tot_launch += l
        if f["self"].shape[0] > 0:
            data.append((f, tgt, ship, launched))
    print(f"  {len(data)} usable decisions | launch→candidate match rate: "
          f"{tot_match}/{tot_launch} ({100*tot_match/max(1,tot_launch):.0f}%)")
    if not data:
        print("no usable data"); return

    pol = PlanetPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    t = lambda a: torch.tensor(a)
    rng = np.random.default_rng(0)
    # Down-weight the dominant no-op class (index 0) so BC doesn't collapse to "hold".
    tgt_weight = torch.tensor([args.noop_weight] + [1.0] * pol.K)

    losses = []
    for step in range(args.steps):
        f, tgt, ship, launched = data[rng.integers(len(data))]
        out = pol.forward(t(f["self"]), t(f["cand"]), t(f["global"]), t(f["cand_mask"]))
        # target loss over all sources (no-op included); ship loss only where launched
        tl = F.cross_entropy(out.target_logits, t(tgt), weight=tgt_weight)
        if launched.any():
            sl = F.cross_entropy(out.ship_logits[t(launched)], t(ship[launched]))
        else:
            sl = torch.tensor(0.0)
        loss = tl + sl
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % max(1, args.steps // 6) == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"  step {step+1:>4}: loss={recent:.4f}", flush=True)

    torch.save(pol.state_dict(), args.out)
    print(f"\nstart loss ~{sum(losses[:20])/20:.3f} -> end loss ~{sum(losses[-20:])/20:.3f}")
    print(f"Saved policy -> {args.out}")
    print("(local data is low-rating; this only proves the BC loop. Re-run on "
          "--data training/moves_prizezone.jsonl.gz --min-rating 1500 once pulled.)")


if __name__ == "__main__":
    main()
