#!/usr/bin/env bash
# Launch one PPO training job on a Jetstream2 A100 instance.
# Usage: bash launch_gpu.sh [run_name]
set -e

RUN_NAME=${1:-"v10_gpu_$(hostname)"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Orbit Wars v10 JAX PPO ==="
echo "Run name : $RUN_NAME"
echo "Script   : $SCRIPT_DIR"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Validate engine before training
echo "--- Validating JAX engine (N=20 games) ---"
python3 "$SCRIPT_DIR/validate_engine.py" --n 20
echo ""

echo "--- Starting training ---"
python3 "$SCRIPT_DIR/train_jax.py" \
    --num_envs 1024 \
    --total_updates 3000 \
    --run_name "$RUN_NAME" \
    "$@"
