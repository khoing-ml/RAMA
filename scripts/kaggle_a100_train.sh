#!/usr/bin/env bash
set -euo pipefail

# Kaggle/Colab single-A100 launcher for the discrete-token Latent-RAMA pipeline.
#
# This keeps the full training flow in kaggle_t4x2_train.sh but overrides the
# distributed launch settings so Accelerate runs one process on one GPU.
#
# Common overrides:
#   IMAGES_DIR=/kaggle/input/celeba-256
#   LATENTS_DIR=/kaggle/working/rama_latents
#   OUTPUT_ROOT=/kaggle/working/rama_outputs
#   MACRO_BATCH=256
#   MICRO_BATCH=128
#   CACHE_BATCH=32
#   TRAIN_WORKERS=2
#   DISABLE_WANDB=1
#   DRY_RUN=1
#   SKIP_INSTALL=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-fp16}"
export ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-fp16}"

# A100 has much more memory than a T4, but this keeps the first run conservative.
export MACRO_BATCH="${MACRO_BATCH:-256}"
export MICRO_BATCH="${MICRO_BATCH:-128}"
export CACHE_BATCH="${CACHE_BATCH:-32}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-2}"

cd "${PROJECT_ROOT}"
exec bash scripts/kaggle_t4x2_train.sh "$@"
