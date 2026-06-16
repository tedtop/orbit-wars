#!/usr/bin/env python3
"""Objective recovery (strategy idea #5, Track A): recover what the top teams value.

Naive behavior cloning caps at the teacher. Instead of cloning *actions*, we recover
their *objective*: each prize-zone launch is a revealed preference — the team chose
one target over the other legal candidates from that source. We fit a linear scorer
``w·cand_features`` (conditional logit over candidates per source) so the team's ACTUAL
target ranks top. The recovered weights say what the top teams optimize for in a target
(production? proximity? hitting the leader? near-dead enemies?), which can then bias the
engine's target shortlist/scoring — and, because our planner can optimize that objective
better than they do, exceed the teacher.

Reuses rl/features.py candidate features (12-d, interpretable) and rl/bc_train.py's
launch->candidate angle-matching for labels. Compares recovered top-1 target-prediction
accuracy to nearest-target and random baselines.

    .venv/bin/python rl/objective_recovery.py --data training/moves_prizezone.jsonl.gz --limit 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_train import label_example, load_examples   # noqa: E402

FEATURE_NAMES = [
    "neutral", "enemy", "tgt_ships", "tgt_prod", "dist", "eta",
    "tgt_leaderness", "tgt_weakness", "near_dead_enemy", "def_atk_ratio",
    "tgt_orbit_r", "comet",
]
NF = len(FEATURE_NAMES)


def build_ranking_examples(raw):
    """For each launched source: (cand_feats [K,NF], legal_mask [K], chosen_idx)."""
    ex = []
    for obs, action, pid in raw:
        f, tgt, ship, launched, matched, launches = label_example(obs, action, pid)
        if f["self"].shape[0] == 0:
            continue
        cand = f["cand"]            # [S, K, NF]
        mask = f["cand_mask"]       # [S, K]
        S, K, _ = cand.shape
        for si in range(S):
            if not launched[si]:
                continue
            chosen = int(tgt[si]) - 1            # tgt 0 = no-op; chosen candidate index
            if chosen < 0 or chosen >= K or not mask[si, chosen]:
                continue
            if int(mask[si].sum()) < 2:          # need a real choice to learn from
                continue
            ex.append((cand[si].astype(np.float32), mask[si].astype(bool), chosen))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "training" / "moves_prizezone.jsonl.gz"))
    ap.add_argument("--min-rating", type=float, default=1500.0)
    ap.add_argument("--limit", type=int, default=20000, help="max raw decisions to load")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "training" / "objective_weights.json"))
    args = ap.parse_args()

    print(f"Loading {args.data} (min_rating={args.min_rating}, limit={args.limit}) …", flush=True)
    raw = load_examples(args.data, args.min_rating, args.limit)
    ex = build_ranking_examples(raw)
    print(f"  ranking examples (launched sources w/ >=2 legal targets): {len(ex)}", flush=True)
    if len(ex) < 50:
        print("not enough data"); return

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ex))
    cut = int(0.85 * len(ex))
    tr, te = idx[:cut], idx[cut:]

    # z-score features over the training pool (for interpretable weight magnitudes)
    allfeat = np.concatenate([ex[i][0][ex[i][1]] for i in tr], axis=0)   # [N_legal, NF]
    mu = allfeat.mean(0); sd = allfeat.std(0) + 1e-6

    def pack(i):
        cand, mask, chosen = ex[i]
        z = (cand - mu) / sd
        return torch.tensor(z), torch.tensor(mask), int(chosen)

    w = torch.zeros(NF, requires_grad=True)
    opt = torch.optim.Adam([w], lr=args.lr)
    NEG = torch.finfo(torch.float32).min
    for step in range(args.steps):
        i = int(tr[rng.integers(len(tr))])
        z, mask, chosen = pack(i)
        logits = (z @ w).masked_fill(~mask, NEG)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([chosen]))
        opt.zero_grad(); loss.backward(); opt.step()

    # evaluation
    def top1(split):
        ok = chosen0 = n = 0
        with torch.no_grad():
            for i in split:
                z, mask, chosen = pack(int(i))
                pred = int((z @ w).masked_fill(~mask, NEG).argmax())
                ok += (pred == chosen); chosen0 += (chosen == 0); n += 1
        return ok / max(1, n), chosen0 / max(1, n), n

    acc_tr, _, _ = top1(tr)
    acc_te, near_te, n_te = top1(te)            # near_te = "always nearest (cand 0)" accuracy
    rand_te = float(np.mean([1.0 / int(ex[int(i)][1].sum()) for i in te]))

    wv = w.detach().numpy()
    order = np.argsort(-np.abs(wv))
    print("\nRecovered target-preference weights (z-scored; + = top teams prefer MORE of this):")
    for j in order:
        print(f"  {FEATURE_NAMES[j]:16s} {wv[j]:+.3f}")
    print(f"\nTop-1 target-prediction accuracy:")
    print(f"  recovered scorer : train {acc_tr*100:.1f}%   test {acc_te*100:.1f}%  (n_test={n_te})")
    print(f"  nearest-target   : test {near_te*100:.1f}%")
    print(f"  random           : test {rand_te*100:.1f}%")
    verdict = "LEARNED real preferences" if acc_te > near_te + 0.03 else "≈ nearest-target (little signal beyond proximity)"
    print(f"  -> {verdict}")

    Path(args.out).write_text(json.dumps({
        "feature_names": FEATURE_NAMES, "weights": wv.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
        "acc_test": acc_te, "acc_nearest": near_te, "acc_random": rand_te, "n_test": n_te,
    }, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
