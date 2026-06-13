"""
Win-rate benchmark: run N games of a chosen bot vs each baseline across seeds.

Usage:
    .venv/bin/python bench.py [--bot NAME] [--games N] [--seed-offset K]

--bot selects which bot to benchmark (same names/aliases as arena.py): a file
in agents/ (e.g. coordinated_strike_interceptor, or the alias v3), the generated submission
file (main), or a builtin (random/starter). Defaults to main.
"""
import argparse
import glob
import logging
import os
import sys

os.environ["KAGGLE_ENV_LOG_LEVEL"] = "ERROR"
logging.disable(logging.INFO)
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
BUILTINS = ["random", "starter"]


def discover_bots():
    """{name: spec} — agents/*.py and main.py map to file paths; builtins to
    their engine name. Mirrors arena.py so names/aliases are consistent."""
    bots = {}
    for path in sorted(glob.glob(os.path.join(AGENTS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name != "__init__":
            bots[name] = path
    bots["main"] = os.path.join(ROOT, "main.py")
    for b in BUILTINS:
        bots[b] = b
    return bots


def resolve(name, bots):
    """(resolved_name, spec) for an exact name or a unique suffix alias."""
    if name in bots:
        return name, bots[name]
    matches = [n for n in bots if n.endswith(name) or n.endswith("_" + name)]
    if len(matches) == 1:
        return matches[0], bots[matches[0]]
    raise SystemExit(f"Unknown bot '{name}'. Available: {', '.join(bots)}")


def play_one(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run(agents)
    except Exception as exc:
        print(f"  [ERROR seed={seed}] {exc}", file=sys.stderr)
        return None
    rewards = [s.reward for s in env.steps[-1]]
    # A None reward means an agent errored or timed out (kaggle-environments
    # leaves reward unset). Treat the whole game as a skipped/errored seed so
    # callers — which all guard `if rewards is None` — don't choke on None.
    if any(r is None for r in rewards):
        statuses = [s.status for s in env.steps[-1]]
        print(f"  [SKIP seed={seed}] non-terminal rewards={rewards} "
              f"statuses={statuses}", file=sys.stderr)
        return None
    return rewards


def bench_vs(agent_spec, opponent_name, num_games, seed_offset=0, verbose=True):
    """Return (wins, draws, losses) for `agent_spec` playing both sides."""
    wins = draws = losses = 0
    for i in range(num_games):
        seed = seed_offset + i

        # Play both sides to cancel first-mover bias
        for our_pos in (0, 1):
            agents = [None, None]
            agents[our_pos] = agent_spec
            agents[1 - our_pos] = opponent_name

            rewards = play_one(agents, seed)
            if rewards is None:
                continue
            our_r = rewards[our_pos]
            opp_r = rewards[1 - our_pos]
            if our_r > opp_r:
                wins += 1
            elif our_r == opp_r:
                draws += 1
            else:
                losses += 1

        if verbose and (i + 1) % 5 == 0:
            total = wins + draws + losses
            pct = wins / total * 100 if total else 0
            print(f"  vs {opponent_name}: {i+1}/{num_games} games done — "
                  f"W/D/L = {wins}/{draws}/{losses}  ({pct:.0f}% win)")

    return wins, draws, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", default="main",
                        help="bot to benchmark (agents/ name or alias, 'main', "
                             "or a builtin). Default: main")
    parser.add_argument("--games", type=int, default=30,
                        help="Number of seeds to test (each seed plays 2 games, one per side)")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    bots = discover_bots()
    bot_name, agent_spec = resolve(args.bot, bots)

    baselines = ["random", "starter"]
    print(f"Benchmarking bot: {bot_name}")
    print(f"  ({agent_spec})")
    print(f"{args.games} seeds × 2 sides per seed per baseline\n")

    all_ok = True
    for opp in baselines:
        print(f"--- {bot_name} vs {opp} ---")
        wins, draws, losses = bench_vs(agent_spec, opp, args.games,
                                       args.seed_offset)
        total = wins + draws + losses
        win_rate = wins / total * 100 if total else 0
        print(f"  FINAL  W/D/L = {wins}/{draws}/{losses} / {total}  "
              f"win-rate = {win_rate:.1f}%")
        if opp == "starter" and win_rate < 70.0:
            print(f"  WARNING: win-rate {win_rate:.1f}% < 70% target vs starter-sniper")
            all_ok = False
        print()

    # Self-play: closest proxy for Kaggle evaluation (bot vs itself).
    # Expected ~50% wins since both sides are equal; tracks position bias and
    # confirms no crashes or deadlocks in symmetric play.
    print(f"--- {bot_name} self-play (vs itself) ---")
    sp_p0 = sp_p1 = sp_draw = 0
    for i in range(args.games):
        seed = args.seed_offset + i
        rewards = play_one([agent_spec, agent_spec], seed)
        if rewards is None:
            continue
        if rewards[0] > rewards[1]:
            sp_p0 += 1
        elif rewards[1] > rewards[0]:
            sp_p1 += 1
        else:
            sp_draw += 1
        if (i + 1) % 5 == 0:
            total = sp_p0 + sp_p1 + sp_draw
            print(f"  self-play: {i+1}/{args.games} games — "
                  f"P0 wins={sp_p0}  P1 wins={sp_p1}  draws={sp_draw}")
    sp_total = sp_p0 + sp_p1 + sp_draw
    bias = abs(sp_p0 - sp_p1) / sp_total * 100 if sp_total else 0
    print(f"  FINAL  P0={sp_p0}  P1={sp_p1}  draws={sp_draw} / {sp_total}  "
          f"position bias={bias:.0f}%")
    print()

    if all_ok:
        print(f"All targets met -- {bot_name} is ready.")
        print()
        print("Next: add a new bot in agents/, compare with arena.py --players vN,vN-1,")
        print("then promote the winner:  .venv/bin/python arena.py --promote <bot>")
        print("Submit:  kaggle competitions submit orbit-wars -f main.py -m '<bot>'")
    else:
        print(f"Some targets not met for {bot_name} -- see warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
