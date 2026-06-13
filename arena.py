"""
Orbit Wars arena — pit versioned bots against each other.

Bots live in agents/*.py (each self-contained; filename stem is the version
name). The engine's built-in "random" and "starter" are always available as
fixed baselines. A game is 2 or 4 players (engine hard limit), so you fill each
slot with a bot of your choice.

Interactive "startup screen" (no args):
    .venv/bin/python arena.py

Non-interactive:
    .venv/bin/python arena.py --players graceful_sloth_v2,starter [--games N] [--seed-offset K]
    .venv/bin/python arena.py --players v2,v1,starter,random          # 4-player game
    .venv/bin/python arena.py --round-robin [--games N]               # ladder of all bots (2p)
    .venv/bin/python arena.py --list
    .venv/bin/python arena.py --promote graceful_sloth_v2             # copy bot -> main.py (submission)
"""
import argparse
import glob
import logging
import os
import sys

os.environ["KAGGLE_ENV_LOG_LEVEL"] = "ERROR"
logging.disable(logging.INFO)
# Silence the heavy kaggle-environments import (jax/open_spiel chatter).
_devnull = open(os.devnull, "w")
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = _devnull
_saved_fd1, _saved_fd2 = os.dup(1), os.dup(2)
os.dup2(_devnull.fileno(), 1)
os.dup2(_devnull.fileno(), 2)
try:
    from kaggle_environments import make
finally:
    os.dup2(_saved_fd1, 1); os.close(_saved_fd1)
    os.dup2(_saved_fd2, 2); os.close(_saved_fd2)
    sys.stdout, sys.stderr = _old_stdout, _old_stderr
    _devnull.close()
    logging.disable(logging.NOTSET)

ROOT = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT, "agents")
BUILTINS = ["random", "starter"]  # engine-provided baselines


def discover_bots():
    """Return {name: spec} where spec is a file path (our bots) or a builtin name."""
    bots = {}
    for path in sorted(glob.glob(os.path.join(AGENTS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "__init__":
            continue
        bots[name] = path
    for b in BUILTINS:
        bots[b] = b
    return bots


def resolve(name, bots):
    """Map a user-typed name to an engine agent spec, tolerating short aliases."""
    if name in bots:
        return bots[name]
    # allow shorthand: "v2" -> "graceful_sloth_v2", unique suffix match
    matches = [n for n in bots if n.endswith(name) or n.endswith("_" + name)]
    if len(matches) == 1:
        return bots[matches[0]]
    raise SystemExit(f"Unknown bot '{name}'. Available: {', '.join(bots)}")


def play_one(specs, seed):
    """Run one game; return list of rewards per slot, or None on engine error."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run(list(specs))
    except Exception as exc:  # noqa: BLE001 — report and skip the seed
        print(f"  [ERROR seed={seed}] {exc}", file=sys.stderr)
        return None
    rewards = [s.reward for s in env.steps[-1]]
    # None reward => an agent errored/timed out; skip the game rather than
    # miscounting it (a None would otherwise look like a draw to _winners).
    if any(r is None for r in rewards):
        print(f"  [SKIP seed={seed}] non-terminal rewards={rewards}", file=sys.stderr)
        return None
    return rewards


def _winners(rewards):
    """Slot indices that strictly win (engine gives reward 1 to every top scorer;
    a shared top is a draw)."""
    top = [i for i, r in enumerate(rewards) if r == 1]
    return top if len(top) == 1 else []  # [] => draw


def run_match(names, bots, games, seed_offset, swap_sides=True):
    """Play `games` seeds among the named slots; print a per-bot tally.

    For 2-player matches, each seed is also replayed with sides swapped to
    cancel first-mover bias (unless swap_sides=False)."""
    specs = [resolve(n, bots) for n in names]
    n = len(names)
    wins = {i: 0 for i in range(n)}
    draws = 0
    played = 0

    orders = [list(range(n))]
    if swap_sides and n == 2:
        orders.append([1, 0])

    for i in range(games):
        seed = seed_offset + i
        for order in orders:
            ordered_specs = [specs[o] for o in order]
            rewards = play_one(ordered_specs, seed)
            if rewards is None:
                continue
            played += 1
            w = _winners(rewards)
            if not w:
                draws += 1
            else:
                # map winning ordered-slot back to original slot index
                wins[order[w[0]]] += 1

    print(f"\nResults over {played} games "
          f"({games} seeds{' x2 sides' if len(orders) > 1 else ''}):")
    ranked = sorted(range(n), key=lambda s: -wins[s])
    for s in ranked:
        pct = wins[s] / played * 100 if played else 0
        print(f"  slot {s}: {names[s]:<22} {wins[s]:>3} wins  ({pct:4.0f}%)")
    print(f"  draws: {draws}  ({draws / played * 100 if played else 0:.0f}%)")
    return wins, draws


def run_round_robin(bots, games, seed_offset):
    """2-player ladder: every pair of bots plays, side-swapped. Prints a matrix."""
    names = [n for n in bots if bots[n] != n] + BUILTINS  # our bots first
    names = list(dict.fromkeys(names))
    score = {n: 0 for n in names}
    print(f"Round-robin: {len(names)} bots, {games} seeds x2 sides per pair\n")
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            specs = [resolve(a, bots), resolve(b, bots)]
            aw = bw = dr = 0
            for i in range(games):
                for order in ([0, 1], [1, 0]):
                    rewards = play_one([specs[order[0]], specs[order[1]]],
                                       seed_offset + i)
                    if rewards is None:
                        continue
                    w = _winners(rewards)
                    if not w:
                        dr += 1
                    elif order[w[0]] == 0:
                        aw += 1
                    else:
                        bw += 1
            score[a] += aw
            score[b] += bw
            print(f"  {a:<22} {aw:>3} - {bw:<3} {b:<22} (draws {dr})")
    print("\nLeaderboard (total wins across all pairings):")
    for n in sorted(names, key=lambda x: -score[x]):
        print(f"  {n:<22} {score[n]}")


PROMOTE_HEADER = '''# =============================================================================
# Orbit Wars SUBMISSION — promoted from bot: {name}
# Generated by arena.py --promote {name}. Do not edit here; edit agents/{name}.py.
# =============================================================================
'''


def promote(name, bots):
    """Copy agents/<name>.py to main.py with a submission header tagging the bot."""
    spec = bots.get(name)
    if not spec or spec in BUILTINS:
        raise SystemExit(f"Can only promote a file-based bot. Got '{name}'.")
    body = open(spec).read()
    out = PROMOTE_HEADER.format(name=name) + "\n" + body
    with open(os.path.join(ROOT, "main.py"), "w") as f:
        f.write(out)
    print(f"Promoted '{name}' -> main.py (tagged for submission).")
    print(f"Submit with:  kaggle competitions submit orbit-wars -f main.py -m '{name}'")


def startup_screen(bots):
    """Interactive menu: choose slots, then run a series."""
    names = list(bots)
    print("\n=== Orbit Wars Arena ===\nAvailable bots:")
    for i, n in enumerate(names):
        kind = "builtin" if bots[n] in BUILTINS else "bot"
        print(f"  [{i}] {n}  ({kind})")

    def pick(prompt):
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) < len(names):
            return names[int(raw)]
        return raw  # allow typing a name/alias

    while True:
        p = input("\nNumber of players (2 or 4) [2]: ").strip() or "2"
        if p in ("2", "4"):
            nplayers = int(p)
            break
        print("  Please enter 2 or 4.")

    slots = []
    for s in range(nplayers):
        slots.append(pick(f"  slot {s} bot (number or name): "))

    games = input("Seeds to play [10]: ").strip() or "10"
    seed_off = input("Seed offset [0]: ").strip() or "0"
    run_match(slots, bots, int(games), int(seed_off))

    if input("\nPromote a bot to main.py for submission? (bot name / blank): ").strip():
        # re-prompt cleanly
        pass


def main():
    bots = discover_bots()
    parser = argparse.ArgumentParser(description="Orbit Wars arena")
    parser.add_argument("--players", help="comma-separated bot names for 2 or 4 slots")
    parser.add_argument("--games", type=int, default=10, help="seeds to play")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--round-robin", action="store_true",
                        help="play every pair of bots (2p) and rank them")
    parser.add_argument("--no-swap", action="store_true",
                        help="don't replay 2p matches with sides swapped")
    parser.add_argument("--list", action="store_true", help="list discovered bots")
    parser.add_argument("--promote", metavar="BOT",
                        help="copy agents/BOT.py to main.py for submission")
    args = parser.parse_args()

    if args.list:
        for n in bots:
            kind = "builtin" if bots[n] in BUILTINS else bots[n]
            print(f"  {n:<24} {kind}")
        return
    if args.promote:
        promote(args.promote, bots)
        return
    if args.round_robin:
        run_round_robin(bots, args.games, args.seed_offset)
        return
    if args.players:
        names = [x.strip() for x in args.players.split(",") if x.strip()]
        if len(names) not in (2, 4):
            raise SystemExit("A game needs 2 or 4 players (engine limit).")
        run_match(names, bots, args.games, args.seed_offset,
                  swap_sides=not args.no_swap)
        return

    startup_screen(bots)


if __name__ == "__main__":
    main()
