#!/bin/bash

# This script trains the macro flow model with the shortcut configuration
# and then evaluates the FID of the latest checkpoint.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
CONFIG_FILE="configs/celeba256_sdvae_low_shortcut.yaml"
OUTPUT_DIR="outputs/low_shortcut"
LATENTS_DIR="data/latents"
FID_STATS_FILE="cache/fid_stats_celeba256_5k.pt"
SAMPLER="shortcut"
NUM_SAMPLES=1024

# --- 1. Train the model ---
echo "Starting training for the shortcut model..."
python scripts/train/macro_flow.py \
  --config "${CONFIG_FILE}" \
  --out "${OUTPUT_DIR}"

echo "Training finished."

# --- 2. Find the latest checkpoint ---
echo "Finding the latest checkpoint..."
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LATEST_CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/*.pt | head -n 1)

if [ -z "${LATEST_CHECKPOINT}" ]; then
  echo "Error: No checkpoint found in ${CHECKPOINT_DIR}"
  exit 1
fi

echo "Found checkpoint: ${LATEST_CHECKPOINT}"

# --- 3. Evaluate FID ---
echo "Starting FID evaluation..."
python scripts/eval/macro_fid.py \
  --checkpoint "${LATEST_CHECKPOINT}" \
  --latents "${LATENTS_DIR}" \
  --real-stats "${FID_STATS_FILE}" \
  --sampler "${SAMPLER}" \
  --num-samples "${NUM_SAMPLES}"

echo "FID evaluation finished."
