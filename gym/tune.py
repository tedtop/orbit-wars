#!/usr/bin/env python3
"""Config-search harness for the v4 bot (strategy #1: automated tuning).

Evaluates v4 with candidate `OW_V4_CONFIG` overrides against a fixed panel of
strong opponents, using the gym's isolated subprocess games. Random search by
default; objective = v4's average placement (lower=better) over 4P games vs
random panel triples, with win-rate as a tiebreak.

Examples:
    # validate the asymmetry kit: v4-with-kit vs v4-base head to head in 4P
    python gym/tune.py --ab --games 24
    # random-search the 4P kit knobs (slow; run when the gym is free)
    python gym/tune.py --trials 20 --games 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gym"))
from proxy import ProxyAgent          # noqa: E402
from tournament import play_game, resolve_bot  # noqa: E402

V4 = str(ROOT / "agents" / "v4_producer")
PANEL = ["producer-v2", "v44", "lb-1224", "i-m-stronger", "exp50", "floor-matched"]

# Search space for the 4P opponent-asymmetry kit + a couple of core knobs.
# Widened after diagnostics showed the small default values barely shift the
# greedy selection; effects appear at larger magnitudes (capped to avoid the
# kit swamping the exact competitive scorer).
SPACE = {
    "ffa_leader_attack_bonus": [0.0, 0.05, 0.1, 0.2, 0.4],
    "ffa_weakest_attack_bonus": [0.0, 0.05, 0.1, 0.2, 0.4],
    "ffa_elimination_bonus": [0.0, 1.0, 3.0, 6.0, 12.0],
    "ffa_elimination_strength": [4.0, 6.0, 10.0],
    "ffa_target_prod_bonus": [0.0, 0.1, 0.2, 0.4],
    "roi_threshold": [1.45, 1.5, 1.55, 1.6],
}


def v4_proxy(name, cfg: dict | None):
    extra = {"OW_V4_CONFIG": json.dumps(cfg)} if cfg else {}
    return ProxyAgent(V4, name, env_extra=extra)


def eval_config(cfg: dict | None, panel_paths, games, steps, seed0):
    """Run `games` 4P matches: v4(cfg) vs 3 random panel bots. Return (avg_place, winrate)."""
    v4 = v4_proxy("v4", cfg)
    opp = {n: ProxyAgent(p, n) for n, p in panel_paths.items()}
    places, wins, n = [], 0, 0
    rng = random.Random(seed0)
    for g in range(games):
        trio = rng.sample(list(opp), 3)
        group = [v4] + [opp[t] for t in trio]
        order = list(range(4)); rng.shuffle(order)
        seats = [group[i] for i in order]
        pl = play_game(seats, seed0 + g, steps)
        if pl is None:
            continue
        v4_seat = order.index(0)
        place = pl[v4_seat]
        places.append(place); n += 1
        if place == 1:
            wins += 1
    v4.close()
    for o in opp.values():
        o.close()
    avg = sum(places) / n if n else 9.9
    return avg, (wins / n if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="A/B: current v4 kit vs kit-off base")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()

    panel_paths = {n: resolve_bot(n)[1] for n in PANEL}

    if args.ab:
        base = {"ffa_weakest_attack_bonus": 0.0, "ffa_elimination_bonus": 0.0,
                "ffa_leader_attack_bonus": 0.035, "ffa_target_prod_bonus": 0.08}
        kit = None  # v4 default (kit enabled)
        print("A/B in 4P vs panel triples (lower avg_place = better):")
        for label, cfg in [("v4-KIT (default)", kit), ("v4-base (kit off)", base)]:
            avg, wr, n = eval_config(cfg, panel_paths, args.games, args.steps, args.seed)
            print(f"  {label:22} avg_place={avg:.3f}  win%={wr*100:.0f}  ({n} games)")
        return

    best = (9.9, -1.0, None)
    rng = random.Random(args.seed)
    print(f"Random search: {args.trials} trials x {args.games} games vs {PANEL}")
    for t in range(args.trials):
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        avg, wr, n = eval_config(cfg, panel_paths, args.games, args.steps, args.seed + t * 100)
        flag = ""
        if (avg, wr) < (best[0], best[1]) or avg < best[0]:
            best = (avg, wr, cfg); flag = "  <-- best"
        print(f"  trial {t:>2}: avg_place={avg:.3f} win%={wr*100:.0f}  {json.dumps(cfg)}{flag}", flush=True)
    print(f"\nBEST avg_place={best[0]:.3f} win%={best[1]*100:.0f}\n{json.dumps(best[2], indent=2)}")
    out = ROOT / "strategy" / "v4_tuned_config.json"
    out.write_text(json.dumps({"avg_place": best[0], "winrate": best[1], "config": best[2]}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
