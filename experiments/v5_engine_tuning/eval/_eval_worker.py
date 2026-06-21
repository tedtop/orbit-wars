#!/usr/bin/env python3
"""Fitness-eval worker for pipeline/overnight.py.

Runs N games with the candidate bot in seat 0 vs comet_reaper (2p: 1 control;
4p: 3 controls) and prints `winrate <wins> <games>`. Serial / single-core and
self-contained (a real file, so the driver can launch many concurrently as
subprocesses — no multiprocessing spawn/pickle pitfalls). Knob overrides are read
from the inherited environment (the driver sets e.g. PRECOG_OPP_STRENGTH).

    .venv/bin/python pipeline/_eval_worker.py <bot_main.py> <2p|4p> <n_games> <seed0>
"""
import contextlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gym"))
CONTROL = os.path.join(ROOT, "agents", "comet_reaper", "main.py")
_NULL = open(os.devnull, "w")


def _ships(obs, pid):
    return (sum(p[5] for p in obs["planets"] if p[1] == pid)
            + sum(f[5] for f in obs["fleets"] if f[1] == pid))


def main():
    bot, mode, n, seed0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with contextlib.redirect_stdout(_NULL), contextlib.redirect_stderr(_NULL):
        from kaggle_environments import make
    wins = games = 0
    for i in range(n):
        # alternate the focal's seat in 2P (Orbit Wars has a seat-0 advantage) so the
        # win rate isn't biased; 4P keeps focal in seat 0.
        if mode == "2p":
            fseat = i % 2
            specs = [bot, CONTROL] if fseat == 0 else [CONTROL, bot]
        else:
            fseat = 0
            specs = [bot, CONTROL, CONTROL, CONTROL]
        try:
            with contextlib.redirect_stdout(_NULL), contextlib.redirect_stderr(_NULL):
                env = make("orbit_wars", configuration={"seed": seed0 + i}, debug=False)
                env.run(specs)
            rw = [s.reward for s in env.steps[-1]]
            if any(r is None for r in rw):
                continue
            obs = env.steps[-1][0].observation
            sh = [_ships(obs, k) for k in range(len(rw))]
            order = sorted(range(len(rw)), key=lambda k: (-(rw[k] or 0), -sh[k]))
            wins += 1 if order[0] == fseat else 0
            games += 1
        except Exception:
            continue
    print(f"winrate {wins} {games}")


if __name__ == "__main__":
    main()
