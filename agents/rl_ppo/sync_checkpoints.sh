#!/usr/bin/env bash
# Pull best_model.pt from all instances to local runs/remote/
# Usage: bash agents/rl_ppo/sync_checkpoints.sh
set -e

HOSTS=(
    # m3.quad — 4 CPU (monitor/dashboard only, no training jobs)
    "149.165.175.182"
    # m3.2xl — 64 CPU each
    "149.165.174.18"
    "149.165.174.133"
    "149.165.171.142"
    "149.165.170.73"
    "149.165.171.248"
    # m3.xl — 32 CPU each
    "149.165.175.105"
    "149.165.170.84"
    "149.165.175.177"
)
USER="exouser"
REPO_DIR="/home/${USER}/orbit_wars"
LOCAL_OUT="agents/rl_ppo/runs/remote"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

mkdir -p "$LOCAL_OUT"

echo "=== Syncing checkpoints from ${#HOSTS[@]} hosts ==="

for HOST in "${HOSTS[@]}"; do
    HOST_DIR="${LOCAL_OUT}/${HOST}"
    mkdir -p "$HOST_DIR"
    # Pull all best_model.pt files from all run subdirs
    rsync -avz \
        --include='*/' --include='best_model.pt' --include='*.log' \
        --include='metrics.jsonl' --exclude='*' \
        -e "ssh $SSH_OPTS" \
        "${USER}@${HOST}:${REPO_DIR}/agents/rl_ppo/runs/" \
        "$HOST_DIR/" 2>/dev/null && echo "✓ $HOST synced" || echo "✗ $HOST unreachable"
done

echo ""
echo "Checkpoints in: $LOCAL_OUT"
echo "Best models:"
find "$LOCAL_OUT" -name "best_model.pt" -exec sh -c \
    'echo "  $(dirname $1 | xargs basename): $(python3 -c "import torch; c=torch.load(\"$1\",map_location=\"cpu\"); print(f\"U{c.get(chr(117)+chr(112)+chr(100)+chr(97)+chr(116)+chr(101),\"?\")}, S{c.get(chr(115)+chr(116)+chr(101)+chr(112)+chr(115),\"?\"):,}\")")"' \
    _ {} \; 2>/dev/null
