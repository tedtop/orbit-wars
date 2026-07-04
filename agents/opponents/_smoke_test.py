#!/usr/bin/env python3
"""Driver: smoke-test every playable extracted bot in an isolated subprocess.

Each bot is run with cwd=its folder so its local `orbit_lite` / `submission`
modules resolve correctly and don't collide across bots. Prints a PASS/FAIL table.

Usage:  .venv/bin/python agents/opponents/_smoke_test.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Reference/training notebooks (no playable main.py) — excluded.
SKIP = {
    "producer-orbit-wars-utils",
    "orbit-wars-complete-game-mechanics-deep-dive",
    "orbit-wars-physics-helper-module",
    "reverse-engineering-agents-replay-analysis-tooli",
    "orbit-wars-reinforcement-learning-tutorial",
    "simplified-orbit-wars-agent",  # not reconstructable (private model artifact)
}

TIMEOUT = 120  # seconds per bot


def main():
    folders = sorted(
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name not in SKIP and (p / "main.py").exists()
    )
    results = []
    for folder in folders:
        name = folder.name
        try:
            proc = subprocess.run(
                [PY, str(ROOT / "_smoke_one.py"), "2", "8"],
                cwd=folder, capture_output=True, text=True, timeout=TIMEOUT,
            )
            out = (proc.stdout + proc.stderr).strip().splitlines()
            verdict = out[-1] if out else f"NO OUTPUT (rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            verdict = f"TIMEOUT (>{TIMEOUT}s)"
        results.append((name, verdict))
        print(f"{'PASS' if verdict.startswith('PASS') else 'FAIL':4}  {name:55}  {verdict}")

    n_pass = sum(1 for _, v in results if v.startswith("PASS"))
    print(f"\n{n_pass}/{len(results)} bots passed smoke test.")


if __name__ == "__main__":
    main()
