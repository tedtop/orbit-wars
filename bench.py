"""
Win-rate benchmark: run N games vs each baseline across different seeds.

Usage:
    .venv/bin/python bench.py [--games N] [--seed-offset K]
"""
import argparse
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

AGENT = os.path.join(os.path.dirname(__file__), "main.py")


def play_one(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run(agents)
    except Exception as exc:
        print(f"  [ERROR seed={seed}] {exc}", file=sys.stderr)
        return None
    rewards = [s.reward for s in env.steps[-1]]
    return rewards


def bench_vs(opponent_name, num_games, seed_offset=0, verbose=True):
    """Return (wins, draws, losses) for our agent playing both sides."""
    wins = draws = losses = 0
    for i in range(num_games):
        seed = seed_offset + i

        # Play both sides to cancel first-mover bias
        for our_pos in (0, 1):
            agents = [None, None]
            agents[our_pos] = AGENT
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
    parser.add_argument("--games", type=int, default=30,
                        help="Number of seeds to test (each seed plays 2 games, one per side)")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    baselines = ["random", "starter"]
    print(f"Benchmarking {args.games} seeds × 2 sides per seed per baseline\n")

    all_ok = True
    for opp in baselines:
        print(f"--- vs {opp} ---")
        wins, draws, losses = bench_vs(opp, args.games, args.seed_offset)
        total = wins + draws + losses
        win_rate = wins / total * 100 if total else 0
        print(f"  FINAL  W/D/L = {wins}/{draws}/{losses} / {total}  "
              f"win-rate = {win_rate:.1f}%")
        if opp == "starter" and win_rate < 70.0:
            print(f"  WARNING: win-rate {win_rate:.1f}% < 70% target vs starter-sniper")
            all_ok = False
        print()

    if all_ok:
        print("All targets met -- agent is ready.")
        print()
        print("Next: iterate on main.py to improve win-rate, then re-run this benchmark.")
        print("When ready to submit: kaggle competitions submit orbit-wars -f main.py -m 'v1'")
    else:
        print("Some targets not met -- see warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
