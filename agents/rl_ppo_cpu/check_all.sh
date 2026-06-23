#!/usr/bin/env bash
# Quick health check across all training instances.
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
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"

for HOST in "${HOSTS[@]}"; do
    echo "=== $HOST ==="
    ssh $SSH_OPTS "${USER}@${HOST}" \
        "grep -rh 'Eval\|New best\|comet_reaper_WR\|^U' ${REPO_DIR}/agents/rl_ppo_cpu/runs/*/*.log 2>/dev/null | tail -12" \
        2>/dev/null || echo "  (unreachable)"
    echo ""
done
