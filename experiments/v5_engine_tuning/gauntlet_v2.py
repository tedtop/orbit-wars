#!/usr/bin/env python3
"""Gauntlet v2 — a RATING-STYLE evaluator that fixes gauntlet v1's gym↔live mismatches.

v1 was: pairwise win% vs 5 STRONG bots, mostly 2P, binary win/loss. Replays showed that
overstates (schmeekler beat comet_reaper 72% in v1 but is field-PARITY live). The three fixes:
  - FIELD:  a DIVERSE pool spanning skill (strong opponents + weak/mid archived bots), not just 5 strong ones.
  - FORMAT: MIXED 2P/4P pods (~40% 4P, ≈ live), seats randomized.
  - METRIC: OpenSkill rating from PLACEMENT (ships+planets tiebreak), not binary win-rate.

Calibration: we have 4 of our own bots with LIVE scores — comet_reaper 1248, schmeekler 1080,
markowitz 566, coordinated_strike 524. A trustworthy v2 must reproduce that ordering
({comet_reaper ≈ schmeekler} ≫ {markowitz, coordinated}). Run all participants in ONE tournament
and check the v2 rating order vs live.

    .venv/bin/python experiments/v5_engine_tuning/gauntlet_v2.py --games 120
    .venv/bin/python experiments/v5_engine_tuning/gauntlet_v2.py --games 200 --p4 0.4
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import arena  # reuse play_one / discover_bots / resolve / _placements_4p / PlackettLuce
from openskill.models import PlackettLuce

# Diverse field spanning the skill range. Anchors carry known live scores for calibration.
ANCHORS = {  # bot name -> live publicScore
    "comet_reaper": 1248.4,
    "schmeekler": 1079.6,
    "markowitz_portfolio_optimization": 565.5,
    "coordinated_strike_interceptor": 523.5,
}
# Strong opponents live in nested folders — give explicit paths (as evaluate.py does).
STRONG = {
    "the-producer-v2": "agents/opponents/the-producer-v2/main.py",
    "i-m-stronger": "agents/opponents/orbit-wars-i-m-stronger/main.py",
    "floor-matched": "agents/opponents/floor-matched-fleets-target-veto-evacuation/main.py",
    "1266-elo": "agents/opponents/1266-elo-the-v44-agent-100-self-contained/main.py",
}
MIDWEAK = ["the_vulture", "greedy_lead_interceptor", "artificial_potential_fields",
           "bayesian_wave_function_collapse", "minimax_fleet_allocation",
           "susceptible_infected_recovered_model", "stigmergic_pheromone_routing"]


def safe_resolve(name, bots):
    """Resolve a bot to a runnable spec; explicit path for strong opponents; None if missing."""
    if name in STRONG:
        p = ROOT / STRONG[name]
        return str(p) if p.exists() else None
    try:
        return arena.resolve(name, bots)
    except Exception:
        return None


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--p4", type=float, default=0.4, help="fraction of pods that are 4P (≈ live)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    bots = arena.discover_bots()
    names = list(ANCHORS) + list(STRONG) + MIDWEAK
    spec = {b: safe_resolve(b, bots) for b in names}
    field = [b for b in names if spec[b]]
    missing = [b for b in names if not spec[b]]
    if missing:
        print(f"  [skip missing bots]: {missing}")
    model = PlackettLuce()
    R = {b: model.rating(name=b) for b in field}

    played = errored = 0
    for g in range(args.games):
        k = 4 if random.random() < args.p4 else 2
        if len(field) < k:
            continue
        roster = random.sample(field, k)
        specs = [spec[b] for b in roster]
        res = arena.play_one(specs, seed=args.seed * 100000 + g)
        if not res:
            errored += 1
            continue
        rewards, env = res
        placements = arena._placements_4p(rewards, env)  # rank per slot (1=winner)
        teams = [[R[roster[i]]] for i in range(k)]
        ranks = [placements[i] for i in range(k)]
        rated = model.rate(teams, ranks=ranks)
        for i in range(k):
            R[roster[i]] = rated[i][0]
        played += 1
        if played % 25 == 0:
            print(f"  ...{played} games")

    print(f"\n=== gauntlet v2: {played} games ({errored} engine errors), 4P fraction {args.p4} ===")
    ordered = sorted(field, key=lambda b: -R[b].ordinal())
    for b in ordered:
        tag = f"   <-- LIVE {ANCHORS[b]:.0f}" if b in ANCHORS else ""
        print(f"  {R[b].ordinal():6.2f}  (mu {R[b].mu:5.1f})  {b}{tag}")

    # calibration: v2 ordinal vs live score for the 4 anchors
    av = [b for b in ANCHORS if b in R]
    v2 = [R[b].ordinal() for b in av]
    live = [ANCHORS[b] for b in av]
    rho = spearman(v2, live)
    print(f"\n  CALIBRATION (anchors): Spearman(v2 rating, live score) = {rho:+.2f}  "
          f"({'GOOD — tracks live' if rho > 0.8 else 'WEAK — v2 not yet trustworthy'})")
    print("  anchors by v2:", [b for b in ordered if b in ANCHORS])
    print("  anchors by live:", sorted(av, key=lambda b: -ANCHORS[b]))


if __name__ == "__main__":
    main()
