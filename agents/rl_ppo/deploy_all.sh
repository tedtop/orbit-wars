#!/usr/bin/env bash
# Deploy repo to all Jetstream2 instances and start training.
# Usage: bash agents/rl_ppo/deploy_all.sh
# Edit the HOSTS array with your actual IPs before running.
set -e

HOSTS=(
    # m3.2xl — 64 CPU each (original fleet)
    "149.165.174.18"
    "149.165.174.133"
    "149.165.171.142"
    "149.165.170.73"
    "149.165.171.248"
    # m3.xl — 32 CPU each (bonus fleet, added 2026-06-18)
    "149.165.175.105"
    "149.165.170.84"
    "149.165.175.177"
)
USER="exouser"
REPO_DIR="/home/${USER}/orbit_wars"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "=== Deploying to ${#HOSTS[@]} hosts ==="

for HOST in "${HOSTS[@]}"; do
    echo ""
    echo "--- $HOST ---"

    # Sync repo (exclude venvs and cache)
    rsync -avz --progress \
        --exclude='.venv' --exclude='.venv_gpu' \
        --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='archive/' --exclude='replays/' \
        --exclude='runs/' --exclude='episodes/' \
        --exclude='training/*.pt' \
        -e "ssh $SSH_OPTS" \
        "$LOCAL_DIR/" "${USER}@${HOST}:${REPO_DIR}/" &
done

wait
echo ""
echo "=== Sync complete. Launching training on all hosts... ==="

for HOST in "${HOSTS[@]}"; do
    echo "--- Launching on $HOST ---"
    ssh $SSH_OPTS "${USER}@${HOST}" \
        "bash ${REPO_DIR}/agents/rl_ppo/launch_cpu.sh ${REPO_DIR}" &
done

wait
echo ""
echo "=== All hosts launched ==="
echo ""
echo "Monitor all with:"
echo "  bash agents/rl_ppo/check_all.sh"
