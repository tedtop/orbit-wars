"""Single-game local test harness for the Orbit Wars agent."""
import logging
import os
import sys

# Suppress kaggle-environments startup noise during import:
#   - cabt "Loading environment failed" is a plain print() → stdout
#   - open_spiel INFO lines go through Python logging → stderr
# Silence both Python-level streams and their underlying fds.
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
DEBUG = "--debug" in sys.argv or "-d" in sys.argv


def run_game(agents, seed=42, label=None):
    env = make("orbit_wars", configuration={"seed": seed}, debug=DEBUG)
    env.run(agents)
    results = [(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])]
    tag = label or " vs ".join(
        a if isinstance(a, str) else a.__name__ for a in agents
    )
    print(f"\n=== {tag} (seed={seed}) ===")
    for player_id, reward, status in results:
        print(f"  Player {player_id}: reward={reward:+d}  status={status}")
    return results


def main():
    print("Running Orbit Wars local test games...")
    run_game([AGENT, "random"],   seed=42,  label="agent vs random")
    run_game([AGENT, "starter"],  seed=42,  label="agent vs starter-sniper")
    run_game(["random", AGENT],   seed=99,  label="random vs agent (sides swapped)")
    run_game(["starter", AGENT],  seed=99,  label="starter-sniper vs agent (sides swapped)")
    # 4-player game: agent vs 3 random opponents
    run_game([AGENT, "random", "random", "random"], seed=7, label="4p: agent vs 3×random")
    print("\nAll games complete.")
    print()
    print("Next: run the win-rate benchmark (30 seeds, ~2-3 min)")
    print("  .venv/bin/python bench.py --games 30")


if __name__ == "__main__":
    main()
