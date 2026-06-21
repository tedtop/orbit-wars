#!/usr/bin/env python3
"""v5 seat-fair eval worker.

Runs N games of a focal bot vs an opponent (default comet_reaper), seat-rotated in 2P,
and prints ``winrate <wins> <games>``. Reads ``CRT_CONFIG`` (+ any env knobs) from the
environment, so the Optuna study sets the focal's config per trial. Single-core and
self-contained (a real file → the study launches many concurrently as subprocesses).

    .venv/bin/python eval_worker.py <focal_main.py> <2p|4p> <n_games> <seed0> [opponent_main.py]
"""
import contextlib
import os
import sys

ROOT = "/Users/Ted/src/orbit_wars"
sys.path.insert(0, os.path.join(ROOT, "gym"))
_NULL = open(os.devnull, "w")


def _ships(o, p):
    return (sum(x[5] for x in o["planets"] if x[1] == p)
            + sum(f[5] for f in o["fleets"] if f[1] == p))


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main():
    focal = _abs(sys.argv[1]); mode = sys.argv[2]; n = int(sys.argv[3]); seed0 = int(sys.argv[4])
    opp = _abs(sys.argv[5]) if len(sys.argv) > 5 else os.path.join(ROOT, "agents/comet_reaper/main.py")
    with contextlib.redirect_stdout(_NULL), contextlib.redirect_stderr(_NULL):
        from kaggle_environments import make
    wins = games = 0
    for i in range(n):
        if mode == "2p":
            fseat = i % 2
            specs = [focal, opp] if fseat == 0 else [opp, focal]
        else:
            fseat = 0
            specs = [focal, opp, opp, opp]
        try:
            with contextlib.redirect_stdout(_NULL), contextlib.redirect_stderr(_NULL):
                env = make("orbit_wars", configuration={"seed": seed0 + i}, debug=False)
                env.run(specs)
            rw = [s.reward for s in env.steps[-1]]
            if any(r is None for r in rw):
                continue
            o = env.steps[-1][0].observation
            sh = [_ships(o, k) for k in range(len(rw))]
            order = sorted(range(len(rw)), key=lambda k: (-(rw[k] or 0), -sh[k]))
            wins += 1 if order[0] == fseat else 0
            games += 1
        except Exception:
            continue
    print(f"winrate {wins} {games}")


if __name__ == "__main__":
    main()
