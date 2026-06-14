#!/usr/bin/env bash
# Orbit Wars — launch dashboard + run pipeline every 15 minutes.
# Ctrl-C kills both.
#
# Usage:  bash start.sh

set -e
cd "$(dirname "$0")"

echo "🪐 Orbit Wars"
echo "  Dashboard → http://localhost:8501"
echo "  Pipeline  → every 15 minutes"
echo "  Stop      → Ctrl-C"
echo ""

.venv/bin/streamlit run dashboard/app.py --server.headless true &
DASH_PID=$!
trap "echo ''; echo 'Stopping…'; kill $DASH_PID 2>/dev/null; wait $DASH_PID 2>/dev/null; exit 0" INT TERM

# Run pipeline immediately on start, then every 15 minutes
while true; do
    echo "=== $(date '+%H:%M:%S') — pipeline ==="
    bash pipeline/run_pipeline.sh || echo "[pipeline error, will retry next cycle]"
    echo "Next pipeline run in 15 min…"
    sleep 900
done
