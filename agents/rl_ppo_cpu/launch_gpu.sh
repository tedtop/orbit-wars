#!/usr/bin/env bash
# GPU server launch script for Jetstream2 (A100) or Lambda Labs GH200
# Run from repo root: bash agents/rl_ppo_cpu/launch_gpu.sh
set -e

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
LOG_DIR=agents/rl_ppo_cpu/runs
RUN_NAME="gpu_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

# ── Environment setup ─────────────────────────────────────────────────────────
echo "[setup] Creating venv..."
$PYTHON -m venv .venv_gpu
source .venv_gpu/bin/activate

echo "[setup] Installing deps..."
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
pip install kaggle-environments numpy -q

echo "[setup] Verifying CUDA..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ── Patch GPU config inline ───────────────────────────────────────────────────
# Override num_envs for GPU: 128 envs vs 64 locally.
# kaggle_environments is sequential so more envs = more diversity, not speed.
export OW_NUM_ENVS=128
export OW_TOTAL_UPDATES=30000

mkdir -p "$LOG_DIR"

echo "[launch] Starting training run: $RUN_NAME"
echo "[launch] Log: $LOG_FILE"
echo "[launch] num_envs=$OW_NUM_ENVS  total_updates=$OW_TOTAL_UPDATES"

nohup python agents/rl_ppo_cpu/train.py \
    --num_envs ${OW_NUM_ENVS} \
    --total_updates ${OW_TOTAL_UPDATES} \
    > "$LOG_FILE" 2>&1 &
PID=$!
echo "[launch] PID: $PID"
echo "$PID" > "$LOG_DIR/${RUN_NAME}.pid"

echo ""
echo "Monitor with:  tail -f $LOG_FILE"
echo "Stop with:     kill \$(cat $LOG_DIR/${RUN_NAME}.pid)"
echo ""
echo "Expected milestones:"
echo "  ~200 updates:  loss curve should be converging"
echo "  ~1000 updates: checkpoint pool active, eval vs greedy"
echo "  ~5000 updates: eval vs comet_reaper (target >55%)"
echo "  ~15000 updates: ready to package submission"
