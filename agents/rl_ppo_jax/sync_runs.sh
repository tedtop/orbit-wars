#!/usr/bin/env bash
# Sync GPU instance runs/ → local Mac for unified Tensorboard.
# Usage:
#   ./sync_runs.sh                  # one-shot sync
#   watch -n 300 ./sync_runs.sh     # every 5 min

set -euo pipefail

SSH_KEY="$HOME/.ssh/id_rsa"
LOCAL_RUNS="$(dirname "$0")/runs"
mkdir -p "$LOCAL_RUNS"

# Add IP addresses of all active GPU instances here (one per line)
GPU_IPS=(
  "149.165.172.114"
  "149.165.171.71"
)

for IP in "${GPU_IPS[@]}"; do
  rsync -az --timeout=15 \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
    "exouser@$IP:~/orbit_wars/agents/rl_ppo_jax/runs/" \
    "$LOCAL_RUNS/" 2>/dev/null && echo "$(date -u +%H:%M:%S) synced $IP" \
    || echo "$(date -u +%H:%M:%S) SKIP $IP (unreachable)"
done

echo "→ tensorboard --logdir $LOCAL_RUNS"
