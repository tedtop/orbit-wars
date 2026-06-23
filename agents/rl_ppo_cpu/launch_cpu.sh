#!/usr/bin/env bash
# Run on a Jetstream2 CPU instance (m3.xl, 32 cores) after rsync — v9 ET.
# Launches 4 parallel ET training jobs with distinct run names.
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

mkdir -p agents/rl_ppo_cpu/runs agents/rl_ppo_cpu/checkpoints

HOSTNAME=$(hostname -s)
echo "[launch] Starting 4 parallel ET training jobs on $HOSTNAME..."

for i in 1 2 3 4; do
    RUN="${HOSTNAME}_v9_job${i}"
    LOG="agents/rl_ppo_cpu/runs/${RUN}/${RUN}.log"
    mkdir -p "agents/rl_ppo_cpu/runs/${RUN}"
    nohup python -u agents/rl_ppo_cpu/train.py \
        --run_name "$RUN" \
        --model-type et \
        --num_envs 64 \
        --total_updates 30000 \
        > "$LOG" 2>&1 &
    echo "[launch] Job $i → PID $! | log: $LOG"
done

echo ""
echo "All 4 jobs launched. Monitor with:"
echo "  tail -f agents/rl_ppo_cpu/runs/${HOSTNAME}_v9_job1/${HOSTNAME}_v9_job1.log"
echo "  grep 'Eval\\|New best\\|comet' agents/rl_ppo_cpu/runs/${HOSTNAME}_v9_*/*.log"
