#!/usr/bin/env bash
# Orbit Wars — launch dashboard + pipeline every 15 min + ngrok tunnel.
# Ctrl-C kills everything.
#
# Usage:  bash start.sh

set -e
cd "$(dirname "$0")"

VERBOSE=0
for arg in "$@"; do
    [[ "$arg" == "--verbose" ]] && VERBOSE=1
done
STREAMLIT_STDERR=$([[ $VERBOSE -eq 1 ]] && echo "/dev/stderr" || echo "/dev/null")

INTERVAL=900        # seconds between pipeline runs (15 min)
RL_SYNC_INTERVAL=600 # seconds between RL checkpoint syncs (10 min)
LOCK_FILE="/tmp/orbit_wars_pipeline.last"
RL_LOCK_FILE="/tmp/orbit_wars_rl_sync.last"

# ---------------------------------------------------------------------------
# Rate-limit helper: skip pipeline if it ran within the last $INTERVAL seconds.
# Writes a timestamp to LOCK_FILE so the check survives script restarts.
# ---------------------------------------------------------------------------
run_pipeline_if_due() {
    local now
    now=$(date +%s)

    if [[ -f "$LOCK_FILE" ]]; then
        local last
        last=$(cat "$LOCK_FILE")
        local age=$(( now - last ))
        if (( age < INTERVAL )); then
            local wait=$(( INTERVAL - age ))
            echo "  Pipeline ran ${age}s ago — skipping (next run in ${wait}s)"
            return
        fi
    fi

    echo "=== $(date '+%H:%M:%S') — pipeline ==="
    bash pipeline/run_pipeline.sh || echo "  [pipeline error, will retry next cycle]"
    date +%s > "$LOCK_FILE"
}

# ---------------------------------------------------------------------------
# RL sync helper: pull metrics.jsonl + best_model.pt from fleet every 10 min
# ---------------------------------------------------------------------------
sync_rl_if_due() {
    local now
    now=$(date +%s)
    if [[ -f "$RL_LOCK_FILE" ]]; then
        local last age
        last=$(cat "$RL_LOCK_FILE")
        age=$(( now - last ))
        if (( age < RL_SYNC_INTERVAL )); then
            return
        fi
    fi
    if [[ -f "agents/rl_ppo_cpu/sync_checkpoints.sh" ]]; then
        echo "=== $(date '+%H:%M:%S') — RL sync ==="
        bash agents/rl_ppo_cpu/sync_checkpoints.sh 2>/dev/null | grep -E 'synced|unreachable|Best' || true
        date +%s > "$RL_LOCK_FILE"
    fi
}

# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
.venv/bin/streamlit run dashboard/app.py --server.headless true >/dev/null 2>"$STREAMLIT_STDERR" &
DASH_PID=$!
sleep 2   # give Streamlit a moment to bind to its port

# ---------------------------------------------------------------------------
# ngrok — expose dashboard publicly (optional, skipped if ngrok not found)
# ---------------------------------------------------------------------------
NGROK_PID=""
NGROK_URL=""
if command -v ngrok &>/dev/null; then
    # --log=stdout suppresses the interactive TUI; logs go to file instead
    ngrok http 8501 --log=stdout > /tmp/orbit_wars_ngrok.log 2>&1 &
    NGROK_PID=$!
    # Poll ngrok's local API until the tunnel is up (usually <3s)
    for i in $(seq 1 10); do
        sleep 1
        NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
            | python3 -c "
import sys, json
try:
    tunnels = json.load(sys.stdin).get('tunnels', [])
    print(next((t['public_url'] for t in tunnels if t['public_url'].startswith('https')), ''))
except Exception:
    pass
" 2>/dev/null)
        [[ -n "$NGROK_URL" ]] && break
    done
    [[ -n "$NGROK_URL" ]] && echo "$NGROK_URL" > /tmp/orbit_wars_ngrok_url.txt
else
    echo "  ngrok not found — dashboard is local only (install with: brew install ngrok)"
fi

echo ""
echo "🪐 Orbit Wars"
echo "  Dashboard → http://localhost:8501"
if [[ -n "$NGROK_URL" ]]; then
    echo "  Public    → $NGROK_URL"
elif command -v ngrok &>/dev/null; then
    echo "  Public    → check dashboard.ngrok.com for your URL"
fi
echo "  Pipeline  → every 15 min (rate-limited across restarts)"
echo "  Stop      → Ctrl-C"
echo ""

trap '
    echo ""
    echo "Stopping…"
    kill "$DASH_PID" 2>/dev/null
    [[ -n "$NGROK_PID" ]] && kill "$NGROK_PID" 2>/dev/null
    wait 2>/dev/null
    exit 0
' INT TERM

# ---------------------------------------------------------------------------
# Pipeline loop
# ---------------------------------------------------------------------------
while true; do
    run_pipeline_if_due
    sync_rl_if_due
    sleep 60   # check every minute; actual run only happens after INTERVAL seconds
done
