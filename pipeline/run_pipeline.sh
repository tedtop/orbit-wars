#!/usr/bin/env bash
# Run the full data pipeline: pull episodes → download replays → leaderboard snapshot.
# Designed to be cron-able (e.g. every 30 minutes).
#
# Usage:
#   bash pipeline/run_pipeline.sh
#   # or as a cron job (adjust path):
#   */30 * * * * cd /path/to/orbit_wars && bash pipeline/run_pipeline.sh >> /tmp/orbit_pipeline.log 2>&1

set -e
cd "$(dirname "$0")/.."

echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') — orbit-wars pipeline ==="
.venv/bin/python pipeline/pull_episodes.py
.venv/bin/python pipeline/download_replays.py --limit 30
.venv/bin/python pipeline/leaderboard_snapshot.py
echo "=== done ==="
