#!/usr/bin/env bash
# Run on a Jetstream2 CPU instance (m3.2xl, 64 cores) after rsync.
# Launches 4 parallel training jobs with distinct run names.
set -e

REPO_DIR="${1:-$HOME/orbit_wars}"
cd "$REPO_DIR"

echo "[setup] Installing Python deps (CPU torch)..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install kaggle-environments numpy -q

echo "[setup] Verifying install..."
python -c "import torch, kaggle_environments, numpy; print(f'torch={torch.__version__} numpy={numpy.__version__}')"

mkdir -p agents/rl_ppo/runs

HOSTNAME=$(hostname -s)
echo "[launch] Starting 4 parallel training jobs on $HOSTNAME..."

for i in 1 2 3 4; do
    RUN="${HOSTNAME}_job${i}"
    LOG="agents/rl_ppo/runs/${RUN}.log"
    nohup python -u agents/rl_ppo/train.py \
        --run_name "$RUN" \
        --num_envs 64 \
        --total_updates 30000 \
        --reward_scale 0.01 \
        > "$LOG" 2>&1 &
    echo "[launch] Job $i → PID $! | log: $LOG"
done

echo ""
echo "All 4 jobs launched. Monitor with:"
echo "  tail -f agents/rl_ppo/runs/${HOSTNAME}_job1.log"
echo "  grep 'Eval\\|New best' agents/rl_ppo/runs/${HOSTNAME}_*.log"
