#!/usr/bin/env python3
"""Gauntlet v2 — a RATING-STYLE evaluator that fixes gauntlet v1's gym↔live mismatches.

v1 was: pairwise win% vs 5 STRONG bots, mostly 2P, binary win/loss. Replays showed that
overstates (schmeekler beat comet_reaper 72% in v1 but is field-PARITY live). The three fixes:
  - FIELD:  a DIVERSE pool spanning skill, not just 5 strong ones.
  - FORMAT: MIXED 2P/4P pods (~40% 4P, ≈ live), seats randomized.
  - METRIC: OpenSkill rating from PLACEMENT (ships+planets tiebreak), not binary win-rate.

CALIBRATION: bots with KNOWN live scores are the anchors. Many opponents encode their live score
in their NAME (1266-elo, lb-max-1224, lyonel-1200, heuristic-1110, lb-1000, lb-958) — plus
the-producer-v2 (1259), comet_reaper (1248), markowitz (566), coordinated_strike (524). A trustworthy
v2 must reproduce that ordering (Spearman → 1). schmeekler is only WATCHED (its live rating is still
converging, so it's a moving target — excluded from the Spearman).

    .venv/bin/python experiments/v5_engine_tuning/gauntlet_v2.py --games 250 --p4 0.4
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import arena
from openskill.models import PlackettLuce

OPP = "agents/opponents"
# name -> (spec_or_None, live_score).  spec None => resolve by short name via arena.
ANCHORS = {
    "1266-elo":                         (f"{OPP}/1266-elo-the-v44-agent-100-self-contained/main.py", 1266),
    "the-producer-v2":                  (f"{OPP}/the-producer-v2/main.py", 1259),
    "comet_reaper":                     (None, 1248),
    "lyonel-1200":                      (f"{OPP}/agent-lyonel-1200lb/main.py", 1200),
    # DROPPED — these 4 emit ZERO actions locally (74/74 empty, no-op losses); they don't run
    # in our harness (pulled from Kaggle notebooks) and poisoned the v2.1 calibration:
    #   orbit-star-wars-lb-max-1224, lb-highest-1000-search-learned-value-function,
    #   orbit-wars-heuristic-lb-1110, lb-958-1-orbit-wars-2026-reinforce
    "markowitz_portfolio_optimization": (None, 566),
    "coordinated_strike_interceptor":   (None, 524),
}
WATCH = {"schmeekler": (None, 1085)}  # moving target — ranked but NOT in the Spearman
# extra weak/mid archived bots purely to thicken the lower field
EXTRA = ["the_vulture", "susceptible_infected_recovered_model", "artificial_potential_fields",
         "minimax_fleet_allocation", "stigmergic_pheromone_routing"]
# OFFICIAL kaggle_environments baselines (orbit_wars v1.0.9) — real low-end live opponents; passed
# straight to env.run as built-in agent names (not file bots). 'starter' ≈ what new submissions run.
BUILTINS = ["random", "starter"]


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for rank, i in enumerate(order): r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys); n = len(xs)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--p4", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    bots = arena.discover_bots()

    def resolve(name, spec):
        if spec:
            p = ROOT / spec
            return str(p) if p.exists() else None
        try:
            return arena.resolve(name, bots)
        except Exception:
            return None

    spec = {}
    for n, (s, _) in {**ANCHORS, **WATCH}.items():
        spec[n] = resolve(n, s)
    for n in EXTRA:
        spec[n] = resolve(n, None)
    for n in BUILTINS:
        spec[n] = n  # built-in agent name passed straight to env.run
    field = [n for n, s in spec.items() if s]
    missing = [n for n, s in spec.items() if not s]
    if missing:
        print(f"  [skip missing]: {missing}")

    model = PlackettLuce()
    R = {n: model.rating(name=n) for n in field}
    played = errored = 0
    for g in range(args.games):
        k = 4 if random.random() < args.p4 else 2
        if len(field) < k:
            continue
        roster = random.sample(field, k)
        res = arena.play_one([spec[b] for b in roster], seed=args.seed * 100000 + g)
        if not res:
            errored += 1; continue
        rewards, env = res
        placements = arena._placements_4p(rewards, env)
        rated = model.rate([[R[roster[i]]] for i in range(k)], ranks=[placements[i] for i in range(k)])
        for i in range(k):
            R[roster[i]] = rated[i][0]
        played += 1
        if played % 50 == 0:
            print(f"  ...{played} games")

    print(f"\n=== gauntlet v2.1: {played} games ({errored} errors), 4P frac {args.p4} ===")
    ordered = sorted(field, key=lambda b: -R[b].ordinal())
    allscore = {**ANCHORS, **WATCH}
    for b in ordered:
        live = allscore[b][1] if b in allscore else None
        tag = (f"   <-- LIVE {live}" + (" (WATCH/moving)" if b in WATCH else "")) if live else ""
        print(f"  {R[b].ordinal():6.2f}  {b}{tag}")

    av = [b for b in ANCHORS if b in R]
    rho = spearman([R[b].ordinal() for b in av], [ANCHORS[b][1] for b in av])
    print(f"\n  CALIBRATION over {len(av)} stable anchors: Spearman(v2, live) = {rho:+.2f}  "
          f"({'GOOD — trustworthy' if rho > 0.8 else 'WEAK — iterate field/format/metric'})")
    print("  v2 order :", [b for b in ordered if b in ANCHORS])
    print("  live order:", sorted(av, key=lambda b: -ANCHORS[b][1]))
    if "schmeekler" in R:
        sr = sorted(field, key=lambda b: -R[b].ordinal()).index("schmeekler")
        print(f"  WATCH schmeekler: v2 rank {sr+1}/{len(field)} (ordinal {R['schmeekler'].ordinal():.2f})")


if __name__ == "__main__":
    main()
