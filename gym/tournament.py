#!/usr/bin/env python3
"""Local tournament gym — rank bots by playing them against each other.

Opponent bots run as subprocess-isolated proxies (see proxy.py), so the whole
downloaded field plus our own agents can share one tournament despite colliding
module names. Ratings via OpenSkill PlackettLuce (same model as arena.py).

Examples:
    # quick sanity: 5 bots, 2P, short games
    python gym/tournament.py --bots producer-v2,v44,lb-1224,heuristic-1110,ow-proto \
        --mode 2p --games 1 --steps 150

    # full field, 2P round-robin, real-length games (slow; run in background)
    python gym/tournament.py --bots all --mode 2p --games 2

    # 4P free-for-all over random quads
    python gym/tournament.py --bots all --mode 4p --games 40

Bots are referenced by a short alias (see ALIASES) or any unique substring of an
opponent folder name, or an explicit path to a folder/.py file (e.g. our agents/).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPP = ROOT / "agents" / "opponents"
sys.path.insert(0, str(ROOT / "gym"))

# Quiet kaggle_environments import in THIS process too.
import logging
logging.disable(logging.INFO)  # silence kaggle_environments' env-load chatter
# Fully mute the noisy env registration (cabt dlopen error + OpenSpiel list) by
# sending both stdout and stderr to /dev/null during the import.
_o, _e = os.dup(1), os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1); os.dup2(_devnull, 2)
try:
    from kaggle_environments import make
finally:
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_o, 1); os.dup2(_e, 2)
    os.close(_o); os.close(_e); os.close(_devnull)
from openskill.models import PlackettLuce

from proxy import ProxyAgent

# Non-playable folders (reference/training notebooks).
SKIP = {
    "producer-orbit-wars-utils", "orbit-wars-complete-game-mechanics-deep-dive",
    "orbit-wars-physics-helper-module", "reverse-engineering-agents-replay-analysis-tooli",
    "orbit-wars-reinforcement-learning-tutorial", "simplified-orbit-wars-agent",
}
# Friendly aliases for the common bots.
ALIASES = {
    "producer-v2": "the-producer-v2",
    "v44": "1266-elo-the-v44-agent-100-self-contained",
    "lb-1224": "orbit-star-wars-lb-max-1224",
    "i-m-stronger": "orbit-wars-i-m-stronger",
    "heuristic-1110": "orbit-wars-heuristic-lb-1110",
    "ow-proto": "orbit-wars-agent-ow-proto-passed-1-000",
    "search-value": "lb-highest-1000-search-learned-value-function",
    "reinforce": "lb-958-1-orbit-wars-2026-reinforce",
    "rule-ml": "orbit-wars-rule-base-ml-shot-validator-hybrid",
    "lyonel": "agent-lyonel-1200lb",
    "exp50": "orbit-wars-exp50",
    "floor-matched": "floor-matched-fleets-target-veto-evacuation",
    "gru": "v2-gru",
    "advanced-1608": "orbit-wars-advanced-agent-target-1608-6",
}


def playable_opponents() -> list[str]:
    return sorted(
        p.name for p in OPP.iterdir()
        if p.is_dir() and p.name not in SKIP and (p / "main.py").exists()
    )


def resolve_bot(token: str) -> tuple[str, str]:
    """Return (display_name, path) for a bot token."""
    if token in ALIASES:
        name = ALIASES[token]
        return name, str(OPP / name)
    p = Path(token)
    if p.exists():  # explicit path to folder or .py (e.g. our agents/foo.py)
        return p.stem if p.is_file() else p.name, str(p.resolve())
    if (OPP / token).is_dir():
        return token, str(OPP / token)
    # our own bots in agents/: folder bot (agents/<token>/main.py) or single file (agents/<token>.py)
    agents = ROOT / "agents"
    if (agents / token / "main.py").exists():
        return token, str(agents / token)
    if (agents / f"{token}.py").exists():
        return token, str(agents / f"{token}.py")
    matches = [n for n in playable_opponents() if token in n]
    if len(matches) == 1:
        return matches[0], str(OPP / matches[0])
    raise SystemExit(f"Cannot resolve bot '{token}'. Matches: {matches or 'none'}")


# ---------------------------------------------------------------------------
# Game running
# ---------------------------------------------------------------------------
def total_ships(obs, pid: int) -> float:
    return (sum(p[5] for p in obs["planets"] if p[1] == pid) +
            sum(f[5] for f in obs["fleets"] if f[1] == pid))


def play_game(proxies: list[ProxyAgent], seed: int, steps: int):
    """Run one game; return placements[slot] = rank (1=best), or None on engine error."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": steps}, debug=False)
    _s = os.dup(1); os.dup2(2, 1)
    try:
        env.run(list(proxies))
    except Exception:
        return None
    finally:
        sys.stdout.flush(); os.dup2(_s, 1); os.close(_s)
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None
    obs = env.steps[-1][0].observation
    n = len(rewards)
    ships = [total_ships(obs, i) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-rewards[i], -ships[i]))
    placements = [0] * n
    for rank, slot in enumerate(order):
        placements[slot] = rank + 1
    return placements


# ---------------------------------------------------------------------------
# Tournament drivers
# ---------------------------------------------------------------------------
def run(bot_tokens, mode, games, steps, seed0, out_path):
    names, paths = [], {}
    for tok in bot_tokens:
        nm, pth = resolve_bot(tok)
        if nm not in paths:
            names.append(nm); paths[nm] = pth
    print(f"Tournament: {len(names)} bots | mode={mode} | games={games} | steps={steps}")
    for n in names:
        print(f"  - {n}")

    model = PlackettLuce()
    ratings = {n: model.rating(name=n) for n in names}
    proxies = {n: ProxyAgent(paths[n], n) for n in names}
    record = {n: {"games": 0, "wins": 0, "place_sum": 0} for n in names}
    t0 = time.time()
    gi = 0

    def settle(group, placements):
        nonlocal ratings
        teams = [[ratings[g]] for g in group]
        ranks = [placements[i] for i in range(len(group))]
        new = model.rate(teams, ranks=ranks)
        for i, g in enumerate(group):
            ratings[g] = new[i][0]
            record[g]["games"] += 1
            record[g]["place_sum"] += placements[i]
            if placements[i] == 1:
                record[g]["wins"] += 1

    if mode == "2p":
        pairs = list(itertools.combinations(names, 2))
        for a, b in pairs:
            for g in range(games):
                # alternate sides to cancel position bias
                group = [a, b] if g % 2 == 0 else [b, a]
                pl = play_game([proxies[x] for x in group], seed0 + gi, steps)
                gi += 1
                if pl is None:
                    continue
                settle(group, pl)
            print(f"  [{gi}] {a} vs {b}  ({time.time()-t0:.0f}s)", flush=True)
    else:  # 4p
        quads = list(itertools.combinations(names, 4)) if len(names) >= 4 else []
        random.Random(seed0).shuffle(quads)
        per_quad = max(1, games // max(1, len(quads))) if games >= len(quads) else 1
        targets = quads if games >= len(quads) else quads[:games]
        for quad in targets:
            for g in range(per_quad):
                group = list(quad)
                random.Random(seed0 + gi).shuffle(group)
                pl = play_game([proxies[x] for x in group], seed0 + gi, steps)
                gi += 1
                if pl is None:
                    continue
                settle(group, pl)
            print(f"  [{gi}] {' '.join(s[:10] for s in quad)}  ({time.time()-t0:.0f}s)", flush=True)

    for p in proxies.values():
        p.close()

    table = []
    for n in names:
        r = ratings[n]; rec = record[n]
        gp = rec["games"]
        table.append({
            "bot": n, "rating": round(r.ordinal(), 2), "mu": round(r.mu, 2),
            "sigma": round(r.sigma, 2), "games": gp,
            "winrate": round(rec["wins"] / gp, 3) if gp else 0.0,
            "avg_place": round(rec["place_sum"] / gp, 2) if gp else 0.0,
        })
    table.sort(key=lambda d: -d["rating"])
    print(f"\n=== RANKING ({mode}, {gi} games, {time.time()-t0:.0f}s) ===")
    print(f"{'#':>2}  {'bot':40} {'rating':>7} {'win%':>6} {'avg_pl':>6} {'games':>5}")
    for i, d in enumerate(table, 1):
        print(f"{i:>2}  {d['bot']:40} {d['rating']:>7.2f} {d['winrate']*100:>5.0f}% {d['avg_place']:>6.2f} {d['games']:>5}")

    out = {"mode": mode, "games": gi, "steps": steps, "seconds": round(time.time()-t0, 1),
           "ranking": table}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="all", help="'all' or comma list of aliases/paths")
    ap.add_argument("--mode", choices=["2p", "4p"], default="2p")
    ap.add_argument("--games", type=int, default=2, help="2p: games per pair; 4p: total quads (or per-quad if < #quads)")
    ap.add_argument("--steps", type=int, default=500, help="episodeSteps (use <500 for fast smoke runs)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--with-ours", action="store_true", help="also include our agents/*.py bots")
    ap.add_argument("--out", default=str(ROOT / "strategy" / "gym_results.json"))
    args = ap.parse_args()

    if args.bots == "all":
        tokens = playable_opponents()
    else:
        tokens = [t.strip() for t in args.bots.split(",") if t.strip()]
    if args.with_ours:
        ours = [str(p) for p in sorted((ROOT / "agents").glob("*.py")) if p.stem != "__init__"]
        tokens = ours + tokens
    run(tokens, args.mode, args.games, args.steps, args.seed, args.out)


if __name__ == "__main__":
    main()
