#!/usr/bin/env bash
# Deploy repo to all Jetstream2 instances and start training.
# Usage: bash agents/rl_ppo/deploy_all.sh
# Edit the HOSTS array with your actual IPs before running.
set -e

HOSTS=(
    # v7 active fleet — ppo-3 through ppo-9 shelved (add back when unshelving for VARIANT-B)
    "149.165.175.228"  # ppo-1
    "149.165.175.188"  # ppo-2
    # "149.165.172.107"  # ppo-3 (shelved)
    # "149.165.159.11"   # ppo-4 (shelved)
    # "149.165.174.192"  # ppo-5 (shelved)
    # "149.165.169.105"  # ppo-6 (shelved)
    # "149.165.174.44"   # ppo-7 (shelved)
    # "149.165.150.186"  # ppo-8 (shelved)
    # "149.165.169.255"  # ppo-9 (shelved)
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
