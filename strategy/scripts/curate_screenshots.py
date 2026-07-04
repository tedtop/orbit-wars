#!/usr/bin/env python3
"""
Pick the best screenshots for the portfolio website and copy them to public/screenshots/.
Also writes public/data/screenshots_index.json for the site to consume.

Selection strategy: ~20 images that cover each story beat without redundancy.
"""

import json
import shutil
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
OCR_JSON        = Path(__file__).parent / "08-SCREENSHOT_OCR_FULL.json"
PUBLIC_DIR      = Path(__file__).parent.parent / "website/public/screenshots"
INDEX_OUT       = Path(__file__).parent.parent / "website/public/data/screenshots_index.json"

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

with open(OCR_JSON) as f:
    all_shots = {d["filename"]: d for d in json.load(f)}

# ---------------------------------------------------------------------------
# Hand-curated picks — filename : caption for the website
# ---------------------------------------------------------------------------
# Ordered chronologically to tell the story.
# Numbers extracted from OCR output above; descriptions checked against
# SCREENSHOT_ANALYSIS.md narrative.
# ---------------------------------------------------------------------------

PICKS = [
    # --- Night 0: 23 Scientists arena ---
    ("001-dashboard-winrate-chart.png",
     "The 23-scientist arena dashboard after 10h10m of round-robin play"),

    # --- Day 1-2: First Kaggle submissions ---
    ("005-leaderboard-rank-480.png",
     "First Kaggle leaderboard appearance — rank ~480"),
    ("009-kaggle-submission.png",
     "First bot submission confirmation"),

    # --- Day 2: Reverse-engineering orbit_lite ---
    ("010-dashboard-winrate-chart.png",
     "Local eval dashboard comparing our bots vs orbit_lite clone"),
    ("013-leaderboard-rank-578.png",
     "Score climbing after adopting the orbit_lite engine — ~578 Elo"),

    # --- Day 3-4: comet_reaper breakthrough ---
    ("026-leaderboard-rank-1235.png",
     "comet_reaper hits 1235 Elo — our first submission above 1200"),
    ("038-kaggle-submission.png",
     "schmeekler submission at rank #144 — our best live leaderboard position"),
    ("045-game-board-state.png",
     "Live Orbit Wars game: planet capture in progress"),

    # --- Day 4: Autonomous overnight session (Jun 17) ---
    ("059-leaderboard-rank-105.png",
     "Rank #105 — peak position during the overnight autoresearch run"),
    ("062-leaderboard-rank-154.png",
     "Leaderboard showing our bot competing among the top 200"),
    ("064-autoresearch-session.png",
     "Orchestrator agent planning the next experiment batch"),
    ("079-autoresearch-session.png",
     "Autoresearch log: hypothesis → gate → eval → verdict cycle in action"),
    ("082-autoresearch-session.png",
     "Auditor agent catching a collapsed training run the orchestrator missed"),

    # --- Day 5: RL fleet launch ---
    ("002-tmux-fleet-monitor.png",
     "tmux fleet monitor: 9 Jetstream2 instances running PPO seeds in parallel"),
    ("028-rl-training-step-127.png",
     "PPO training metrics — entropy decay and clip_frac across seeds"),
    ("036-rl-training-step-41877.png",
     "RL training at step 41 877 — policy plateauing at the 20% ceiling"),
    ("100-tmux-fleet-monitor.png",
     "Fleet monitor mid-run showing per-seed step counts and eval win rates"),

    # --- Day 5-6: Diagnosing the ceiling ---
    ("084-kaggle-submission.png",
     "Submission comparison: comet_reaper 1234.7 vs RL agent 1083 — gap is real"),
    ("090-autoresearch-session.png",
     "Autoresearch post-mortem: RL ceiling diagnosis, 20% greedy win rate"),

    # --- Final standing ---
    ("095-leaderboard-rank-1234.png",
     "Final leaderboard position: 1234.7 Elo, top ~200 out of 1248 teams"),
]

# ---------------------------------------------------------------------------
# Copy and build index
# ---------------------------------------------------------------------------

index = []
missing = []

for filename, caption in PICKS:
    src = SCREENSHOTS_DIR / filename
    if not src.exists():
        missing.append(filename)
        continue

    dst = PUBLIC_DIR / filename
    shutil.copy2(src, dst)

    ocr = all_shots.get(filename, {})
    index.append({
        "filename": filename,
        "caption": caption,
        "category": ocr.get("category", "unknown"),
        "key_numbers": ocr.get("key_numbers", ""),
        "ocr_summary": ocr.get("summary", "")[:300],
    })
    print(f"  ✓  {filename}")

if missing:
    print(f"\n  ⚠  Missing (skipped): {missing}")

with open(INDEX_OUT, "w") as f:
    json.dump(index, f, indent=2)

print(f"\nCopied {len(index)} screenshots → {PUBLIC_DIR}")
print(f"Index  → {INDEX_OUT}")
