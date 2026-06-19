#!/usr/bin/env bash
# Quick health check across all training instances.
HOSTS=(
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
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"

for HOST in "${HOSTS[@]}"; do
    echo "=== $HOST ==="
    ssh $SSH_OPTS "${USER}@${HOST}" \
        "grep -h 'Eval\|New best\|^U' ${REPO_DIR}/agents/rl_ppo/runs/*.log 2>/dev/null | tail -8" \
        2>/dev/null || echo "  (unreachable)"
    echo ""
done
