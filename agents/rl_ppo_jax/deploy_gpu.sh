#!/usr/bin/env bash
# Deploy train_jax.py to a new GPU instance and launch training.
# Usage: ./deploy_gpu.sh <IP> <SEED> [RUN_NAME]
# Example: ./deploy_gpu.sh 149.165.172.99 3 v10_jax_gpu3

set -euo pipefail

IP="${1:?Usage: deploy_gpu.sh <IP> <SEED> [RUN_NAME]}"
SEED="${2:?}"
RUN_NAME="${3:-v10_jax_gpu_seed${SEED}}"
SSH_KEY="$HOME/.ssh/id_rsa"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC="rsync -az --timeout=30 -e \"$SSH\""
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Deploying to $IP (seed=$SEED, run=$RUN_NAME) ==="

# 1. Sync code directly into exouser's home
rsync -az --timeout=30 -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$REPO/agents/rl_ppo_jax/" \
  "exouser@$IP:~/orbit_wars/agents/rl_ppo_jax/"

rsync -az --timeout=30 -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$REPO/agents/comet_reaper/" \
  "exouser@$IP:~/orbit_wars/agents/comet_reaper/"

rsync -az --timeout=30 -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$REPO/agents/orbit_lite/" \
  "exouser@$IP:~/orbit_wars/agents/orbit_lite/" 2>/dev/null || true

# 2. Install deps + wipe old runs
ssh -i $SSH_KEY "exouser@$IP" "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env
  pip install tensorboardX -q
  pkill -f train_jax.py 2>/dev/null || true
  rm -rf ~/orbit_wars/agents/rl_ppo_jax/runs/
  mkdir -p ~/orbit_wars/agents/rl_ppo_jax/runs/
  echo deployed
"

# 3. Launch
ssh -i $SSH_KEY "exouser@$IP" "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env
  cd ~/orbit_wars
  nohup python agents/rl_ppo_jax/train_jax.py \
    --num_envs 1024 --total_updates 3000 \
    --seed $SEED --run_name $RUN_NAME \
    > /tmp/${RUN_NAME}.log 2>&1 &
  echo \"launched PID:\$!\"
"

echo "=== Done. Add $IP to sync_runs.sh GPU_IPS array ==="
echo "Reminder: tensorboard --logdir agents/rl_ppo_jax/runs/"
