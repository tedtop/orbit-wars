#!/usr/bin/env python3
"""Canonical gauntlet evaluator — the autoresearch fixed yardstick.

Runs <bot> seat-swapped (arena handles seat rotation) vs comet_reaper + the public panel and prints win%
per opponent + overall. Same metric every iteration so the ratchet's keep/discard decisions are trustworthy.

    .venv/bin/python experiments/v5_engine_tuning/autoresearch/evaluate.py <bot_name> [N] [ENV=VAL ...]
    e.g.  ... evaluate.py schmeekler 50 SCHMEEKLER_STATIC_BONUS=1.5
"""
import os
import re
import subprocess
import sys

# Repo root resolved relative to THIS file, so it works inside a git worktree
# (each Track session runs in its own worktree — must test its OWN agents/, not the main checkout).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PANEL = {
    "comet_reaper": "comet_reaper",
    "the-producer-v2": "agents/opponents/the-producer-v2/main.py",
    "i-m-stronger": "agents/opponents/orbit-wars-i-m-stronger/main.py",
    "floor-matched": "agents/opponents/floor-matched-fleets-target-veto-evacuation/main.py",
    "1266-elo": "agents/opponents/1266-elo-the-v44-agent-100-self-contained/main.py",
}


def run(bot, opp, n, env):
    out = subprocess.run(
        [f"{ROOT}/.venv/bin/python", "arena.py", "--players", f"{bot},{opp}", "--games", str(n), "--no-color"],
        cwd=ROOT, env=env, capture_output=True, text=True).stdout
    m = re.search(r"Results:\s+\S+\s+(\d+)[–-](\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    bot = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 40
    env = dict(os.environ)
    for a in sys.argv[3:]:
        if "=" in a:
            k, v = a.split("=", 1); env[k] = v
    print(f"=== gauntlet: {bot} (n={n}/opp, seat-swapped) ===")
    tw = tg = 0
    for name, opp in PANEL.items():
        r = run(bot, opp, n, env)
        if r:
            w, l = r; tw += w; tg += w + l
            print(f"  vs {name:16s} {w:3d}-{l:<3d} ({100 * w / max(1, w + l):3.0f}%)")
    print(f"  OVERALL: {tw}/{tg} = {100 * tw / max(1, tg):.0f}%  (>50% beats the field)")


if __name__ == "__main__":
    main()
