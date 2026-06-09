#!/usr/bin/env bash
set -euo pipefail

# Kaggle 2x T4 fp16 + W&B launcher for the discrete-token Latent-RAMA pipeline.
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   IMAGES_DIR=/kaggle/input/celeba-256
#   LATENTS_DIR=/kaggle/working/rama_latents
#   OUTPUT_ROOT=/kaggle/working/rama_outputs
#   WANDB_ENTITY=your_entity
#   MACRO_STEPS=200000
#   MICRO_STEPS=200000
#   MACRO_BATCH=32
#   MICRO_BATCH=16
#   CACHE_BATCH=32
#   TRAIN_WORKERS=2
#   SAMPLE_COUNT=16
#   SKIP_INSTALL=1

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGES_DIR="${IMAGES_DIR:-data/celeba256}"
LATENTS_DIR="${LATENTS_DIR:-/kaggle/working/rama_latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/rama_outputs}"
MACRO_OUT="${MACRO_OUT:-${OUTPUT_ROOT}/macro_flow}"
MICRO_OUT="${MICRO_OUT:-${OUTPUT_ROOT}/micro_rama}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/kaggle/working/hf_cache}"
BASES_PATH="${BASES_PATH:-cache/rama_bases_p256_d16.pt}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-/kaggle/working/rama_tokenizer_config.pt}"

MACRO_STEPS="${MACRO_STEPS:-200000}"  
MICRO_STEPS="${MICRO_STEPS:-200000}"
MACRO_BATCH="${MACRO_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-64}"
CACHE_BATCH="${CACHE_BATCH:-32}"
TRAIN_WORKERS="${TRAIN_WORKERS:-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "${PYTHON_BIN}" -m wandb login "${WANDB_API_KEY}"
fi

if [[ ! -d "${IMAGES_DIR}" ]]; then
  cat >&2 <<EOF
Input image directory not found: ${IMAGES_DIR}

On Kaggle, add your CelebA/CelebA-HQ dataset to the notebook and set:
  export IMAGES_DIR=/kaggle/input/<dataset-slug>/<optional-subdir>

Examples:
  export IMAGES_DIR=/kaggle/input/celeba-256
  export IMAGES_DIR=/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba

Mounted Kaggle inputs:
EOF
  find /kaggle/input -maxdepth 3 -type d 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-rama}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/train_end_to_end_discrete_rama.py \
  --images "${IMAGES_DIR}" \
  --latents "${LATENTS_DIR}" \
  --vae-checkpoint stabilityai/sd-vae-ft-mse \
  --hf-cache-dir "${HF_CACHE_DIR}" \
  --dtype fp16 \
  --device cuda \
  --macro-config configs/celeba256_sdvae_macro.yaml \
  --micro-config configs/celeba256_sdvae_micro.yaml \
  --macro-out "${MACRO_OUT}" \
  --micro-out "${MICRO_OUT}" \
  --bases "${BASES_PATH}" \
  --tokenizer-config "${TOKENIZER_CONFIG}" \
  --cache-batch-size "${CACHE_BATCH}" \
  --cache-num-workers "${TRAIN_WORKERS}" \
  --macro-batch-size "${MACRO_BATCH}" \
  --micro-batch-size "${MICRO_BATCH}" \
  --train-num-workers "${TRAIN_WORKERS}" \
  --macro-max-steps "${MACRO_STEPS}" \
  --micro-max-steps "${MICRO_STEPS}" \
  --sample-num-samples "${SAMPLE_COUNT}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --sampler heun \
  --use-accelerate \
  --accelerate-num-processes 2 \
  --accelerate-mixed-precision fp16 \
  --enable-wandb
