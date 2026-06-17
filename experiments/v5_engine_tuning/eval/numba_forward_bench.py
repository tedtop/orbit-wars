#!/usr/bin/env python3
"""Feasibility benchmark: a flat-array Numba forward model for MCTS rollouts.

Drops the continuous geometry (orbital motion, sun, comets) and keeps only the
discrete ship economics (production accrues, fleets land, combat resolves). If this
runs at ~µs scale we can afford thousands of rollouts per turn inside the 1 s budget
→ real MCTS / deep beam search is viable. If not, fall back to beam search over the
engine's own projection.

    .venv/bin/python experiments/v5_engine_tuning/eval/numba_forward_bench.py
"""
import time

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def fast_forward(ships, prod, owner, f_eta, f_ships, f_owner, f_target, n_turns, pid):
    """Roll the board forward n_turns; return my net ships at the end. Pure economics."""
    ships = ships.copy(); owner = owner.copy(); f_eta = f_eta.copy()
    P = ships.shape[0]; F = f_eta.shape[0]
    for _ in range(n_turns):
        for i in range(P):                       # production
            if owner[i] >= 0:
                ships[i] += prod[i]
        for k in range(F):                       # fleet arrivals + combat
            if f_eta[k] > 0:
                f_eta[k] -= 1
                if f_eta[k] == 0:
                    tg = f_target[k]
                    if owner[tg] == f_owner[k]:
                        ships[tg] += f_ships[k]
                    else:
                        ships[tg] -= f_ships[k]
                        if ships[tg] < 0.0:
                            owner[tg] = f_owner[k]
                            ships[tg] = -ships[tg]
    tot = 0.0
    for i in range(P):
        if owner[i] == pid:
            tot += ships[i]
    return tot


def main():
    rng = np.random.default_rng(0)
    P, F = 30, 40
    ships = (rng.random(P) * 50).astype(np.float64)
    prod = (rng.random(P) * 5).astype(np.float64)
    owner = rng.integers(-1, 4, P).astype(np.int64)
    f_eta = rng.integers(1, 15, F).astype(np.int64)
    f_ships = (rng.random(F) * 30).astype(np.float64)
    f_owner = rng.integers(0, 4, F).astype(np.int64)
    f_target = rng.integers(0, P, F).astype(np.int64)

    t0 = time.time()
    fast_forward(ships, prod, owner, f_eta, f_ships, f_owner, f_target, 20, 0)   # compile
    print(f"compile + first call: {time.time() - t0:.2f}s  (one-time on turn 0)")

    for nturns in (20, 60):
        N = 20000
        t0 = time.time()
        for _ in range(N):
            fast_forward(ships, prod, owner, f_eta, f_ships, f_owner, f_target, nturns, 0)
        dt = time.time() - t0
        us = 1e6 * dt / N
        rps = N / dt
        print(f"{nturns:>2}-turn rollout (P={P},F={F}): {us:6.1f} µs/rollout | "
              f"{int(rps):,} rollouts/s | ~{int(0.8 * rps):,} rollouts in an 800ms turn budget")


if __name__ == "__main__":
    main()
