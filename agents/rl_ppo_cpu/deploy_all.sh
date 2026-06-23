#!/usr/bin/env bash
# Deploy repo to all Jetstream2 instances and start training.
# Usage: bash agents/rl_ppo_cpu/deploy_all.sh
# Edit the HOSTS array with your actual IPs before running.
set -e

HOSTS=(
    "149.165.169.195"  # ppo-1
    "149.165.173.145"  # ppo-2
    "149.165.174.28"   # ppo-3
    "149.165.175.244"  # ppo-4
    "149.165.174.123"  # ppo-5
    "149.165.154.56"   # ppo-6
    "149.165.155.119"  # ppo-7
    "149.165.150.220"  # ppo-8
    "149.165.171.224"  # ppo-9
)
USER="exouser"
REPO_DIR="/home/${USER}/orbit_wars"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Wait for all instances to be SSH-ready before deploying
echo "=== Waiting for SSH on ${#HOSTS[@]} hosts ==="
for HOST in "${HOSTS[@]}"; do
    echo -n "  $HOST "
    until ssh $SSH_OPTS "${USER}@${HOST}" "echo ok" &>/dev/null; do
        echo -n "."
        sleep 5
    done
    echo " ready"
done

echo "=== All hosts ready. Deploying ==="

for HOST in "${HOSTS[@]}"; do
    echo ""
    echo "--- $HOST ---"

    # Sync repo (exclude venvs and cache)
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='.venv' --exclude='.venv_gpu' \
        --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='archive/' --exclude='replays/' \
        --exclude='leaderboards/' --exclude='website/' \
        --exclude='strategy/' --exclude='training/' \
        --exclude='pipeline/' --exclude='dashboard/' \
        --exclude='experiments/' --exclude='gym/' --exclude='rl/' \
        --exclude='runs/' --exclude='episodes/' \
        --exclude='agents/opponents/' \
        --exclude='agents/comet_reaper_*/' \
        --exclude='agents/schmeekler*/' \
        --exclude='agents/rl_ppo_cpu/checkpoints/' \
        -e "ssh $SSH_OPTS" \
        "$LOCAL_DIR/" "${USER}@${HOST}:${REPO_DIR}/" &
done

wait
echo ""
echo "=== Sync complete. Launching training on all hosts... ==="

for HOST in "${HOSTS[@]}"; do
    echo "--- Launching on $HOST ---"
    ssh $SSH_OPTS "${USER}@${HOST}" \
        "bash ${REPO_DIR}/agents/rl_ppo_cpu/launch_cpu.sh ${REPO_DIR}" &
done

wait
echo ""
echo "=== All hosts launched ==="
echo ""
echo "Monitor all with:"
echo "  bash agents/rl_ppo_cpu/check_all.sh"
