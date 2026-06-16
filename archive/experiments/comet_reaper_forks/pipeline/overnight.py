#!/usr/bin/env python3
"""Self-looping overnight improvement driver for Orbit Wars.

Runs detached and keeps the machine busy improving the bots while you sleep:

  1. Self-play: keeps a `rl/selfplay.py` chunk alive (warm-started from the latest
     checkpoint) so the neural PlanetPolicy keeps training; relaunches when a chunk
     finishes.
  2. Knob tuning: for each engine-fork bot, sweeps its env knob against the
     comet_reaper control and records win rate per value, so the best non-regressing
     setting is found empirically. (The engine is a tight local optimum — naive
     defaults regress — so we *search*; the sweet spot is often ~0 = comet_reaper.)
  3. Appends a timestamped readout to overnight/RESULTS.md.

Parallelism is by SUBPROCESS (pipeline/_eval_worker.py, many concurrent) — no
multiprocessing spawn/pickle pitfalls. arena.py remains the canonical harness; this
is only the tuning/orchestration loop (in the plan as the Track-B tuner). Crash-safe:
every step is guarded so one failure never kills the loop.

    nohup .venv/bin/python pipeline/overnight.py > overnight/driver.log 2>&1 &
"""
from __future__ import annotations

import contextlib
import datetime as dt
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
WORKER = os.path.join(ROOT, "pipeline", "_eval_worker.py")
RESULTS = os.path.join(ROOT, "overnight", "RESULTS.md")

# bot -> (env_var, grid, mode)
SWEEPS = {
    "precog":    ("PRECOG_OPP_STRENGTH",     [0.0, 0.1, 0.25, 0.5], "2p"),
    "maestro":   ("MAESTRO_GAIN",            [0.0, 0.1, 0.25, 0.5], "2p"),
    "helmsman":  ("HELMSMAN_HORIZON_MULT",   [1.0, 1.15, 1.3, 1.5], "2p"),
    "kingmaker": ("KINGMAKER_LEADER_WEIGHT", [0.0, 0.1, 0.25, 0.5], "4p"),
    "oracle":    ("ORACLE_BIAS",             [0.0, 2.0, 5.0, 10.0], "2p"),
}
GAMES_PER_WORKER = 14
WORKERS_PER_VALUE = 2          # split each grid value across 2 workers (uses ~8 cores total)


def _spec(bot: str) -> str:
    return os.path.join(ROOT, "agents", bot, "main.py")


def _log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with contextlib.suppress(Exception):
        with open(RESULTS, "a") as f:
            f.write(line + "\n")


def sweep(bot: str, env_var: str, grid: list, mode: str, seed0: int):
    """Launch all (value × WORKERS_PER_VALUE) workers concurrently; collect win rates."""
    procs = []   # (value, Popen)
    for vi, val in enumerate(grid):
        for w in range(WORKERS_PER_VALUE):
            env = dict(os.environ)
            env[env_var] = str(val)
            seed = seed0 + vi * 1000 + w * GAMES_PER_WORKER
            p = subprocess.Popen(
                [PYTHON, WORKER, _spec(bot), mode, str(GAMES_PER_WORKER), str(seed)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=ROOT, text=True,
            )
            procs.append((val, p))
    agg: dict = {v: [0, 0] for v in grid}      # value -> [wins, games]
    for val, p in procs:
        try:
            out, _ = p.communicate(timeout=1800)
            parts = out.split()
            if len(parts) >= 3 and parts[0] == "winrate":
                agg[val][0] += int(parts[1]); agg[val][1] += int(parts[2])
        except Exception:
            with contextlib.suppress(Exception):
                p.kill()
    return [(v, (agg[v][0] / agg[v][1] if agg[v][1] else 0.0), agg[v][1]) for v in grid]


def selfplay_alive() -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", "rl/selfplay.py"], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def launch_selfplay() -> None:
    latest = os.path.join(ROOT, "training", "selfplay_overnight.pt")
    init = latest if os.path.exists(latest) else os.path.join(ROOT, "training", "bc_prizezone_v2.pt")
    log = os.path.join(ROOT, "overnight", "selfplay.log")
    cmd = [PYTHON, "-u", os.path.join(ROOT, "rl", "selfplay.py"),
           "--init", init, "--clones", "--iters", "30", "--games", "12", "--steps", "500",
           "--players", "4", "--out", latest]
    with open(log, "a") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=ROOT)
    _log(f"self-play: launched chunk (init={os.path.basename(init)})")


def main():
    os.makedirs(os.path.join(ROOT, "overnight"), exist_ok=True)
    _log("=== overnight driver START ===")
    bots = list(SWEEPS)
    best_seen: dict = {}
    cycle = 0
    while cycle < 1000:
        cycle += 1
        try:
            if not selfplay_alive():
                launch_selfplay()
        except Exception as e:
            _log(f"self-play mgmt error: {e}")

        bot = bots[cycle % len(bots)]
        env_var, grid, mode = SWEEPS[bot]
        try:
            res = sweep(bot, env_var, grid, mode, seed0=1000 + cycle * 100)
            res_sorted = sorted(res, key=lambda r: -r[1])
            bv, bwr, bn = res_sorted[0]
            tbl = "  ".join(f"{v}={wr*100:.0f}%" for v, wr, _ in res)
            tag = "2P vs comet_reaper" if mode == "2p" else "focal-4P vs 3x comet_reaper"
            star = "  <-- BEATS control" if bwr > 0.53 else ""
            _log(f"{bot:9s} {env_var} [{tag}] {tbl} | best={bv} ({bwr*100:.0f}%, n={bn}){star}")
            prev = best_seen.get(bot)
            if prev is None or bwr > prev[1]:
                best_seen[bot] = (bv, bwr)
        except Exception as e:
            _log(f"sweep error ({bot}): {e}")
        time.sleep(2)
    _log("=== overnight driver STOP ===")


if __name__ == "__main__":
    main()
