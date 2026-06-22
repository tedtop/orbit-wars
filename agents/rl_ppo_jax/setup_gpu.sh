#!/usr/bin/env bash
# Install JAX + Flax on a fresh Jetstream2 A100 instance (CUDA 12).
set -e

pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax
pip install kaggle-environments

echo "Setup complete. Verify JAX GPU:"
python3 -c "import jax; print(jax.devices())"
