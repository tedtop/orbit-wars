#!/usr/bin/env bash
# Launch CONTROL arm (reward_scale=0.01) — 2 jobs per host.
# Usage: bash experiments/v7_critic_ab/launch_control.sh <IP>
# Run on ppo-1 and ppo-2 alongside launch_variant_a.sh (same box = matched hardware).
set -e
IP="${1:?Usage: $0 <IP>}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"
REPO="$HOME/orbit_wars"

ssh $SSH_OPTS "exouser@${IP}" bash << 'REMOTE'
set -e
cd ~/orbit_wars
source .venv/bin/activate
pkill -f "train.py.*ctrl" 2>/dev/null || true
sleep 2

for JOB in 1 2; do
    RUN="ctrl_job${JOB}"
    mkdir -p agents/rl_ppo/runs/${RUN}
    nohup python agents/rl_ppo/train.py \
        --run_name "${RUN}" \
        --num_envs 64 \
        --reward_scale 0.01 \
        > agents/rl_ppo/runs/${RUN}/${RUN}.log 2>&1 &
    echo "Launched ${RUN} (PID $!)"
    sleep 2
done
echo "Control arm: 2 jobs running"
REMOTE
