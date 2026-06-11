#!/usr/bin/env bash
set -euo pipefail

# Single-A100 training-only launcher for joint macro + micro RAMA training.
#
# Expected existing inputs:
#   LATENTS_DIR=/kaggle/working/rama_latents
#   TOKENIZER_CONFIG=/kaggle/working/rama_tokenizer_config.pt
#   BASES_PATH=cache/rama_bases_p256_d16.pt
#
# Common overrides:
#   OUTPUT_ROOT=/kaggle/working/rama_outputs
#   MACRO_BATCH=256
#   MICRO_BATCH=128
#   MACRO_STEPS=200000
#   MICRO_STEPS=200000
#   TRAIN_WORKERS=2
#   DISABLE_WANDB=1
#   DRY_RUN=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LATENTS_DIR="${LATENTS_DIR:-/kaggle/working/rama_latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/rama_outputs}"
MACRO_OUT="${MACRO_OUT:-${OUTPUT_ROOT}/macro_flow}"
MICRO_OUT="${MICRO_OUT:-${OUTPUT_ROOT}/micro_rama}"
MACRO_CONFIG="${MACRO_CONFIG:-configs/celeba256_sdvae_macro.yaml}"
MICRO_CONFIG="${MICRO_CONFIG:-configs/celeba256_sdvae_micro.yaml}"
BASES_PATH="${BASES_PATH:-cache/rama_bases_p256_d16.pt}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-/kaggle/working/rama_tokenizer_config.pt}"

MACRO_BATCH="${MACRO_BATCH:-256}"
MICRO_BATCH="${MICRO_BATCH:-128}"
MACRO_STEPS="${MACRO_STEPS:-200000}"
MICRO_STEPS="${MICRO_STEPS:-200000}"
TRAIN_WORKERS="${TRAIN_WORKERS:-2}"

FID_EVERY="${FID_EVERY:-0}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-1024}"
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLER="${SAMPLER:-heun}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"

ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-fp16}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"

MACRO_RESUME="${MACRO_RESUME:-}"
MICRO_RESUME="${MICRO_RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -d "${LATENTS_DIR}" ]]; then
    echo "Missing LATENTS_DIR: ${LATENTS_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${TOKENIZER_CONFIG}" ]]; then
    echo "Missing TOKENIZER_CONFIG: ${TOKENIZER_CONFIG}" >&2
    echo "Set TOKENIZER_CONFIG to an existing tokenizer config, or run quant bound estimation once before training." >&2
    exit 1
  fi
fi

command=(
  accelerate launch
  --num_processes "${ACCELERATE_NUM_PROCESSES}"
  --mixed_precision "${ACCELERATE_MIXED_PRECISION}"
  scripts/train_joint_macro_micro.py
  --macro-config "${MACRO_CONFIG}"
  --micro-config "${MICRO_CONFIG}"
  --latents "${LATENTS_DIR}"
  --macro-out "${MACRO_OUT}"
  --micro-out "${MICRO_OUT}"
  --micro-type categorical
  --tokenizer-config "${TOKENIZER_CONFIG}"
  --bases "${BASES_PATH}"
  --num-workers "${TRAIN_WORKERS}"
  --macro-batch-size "${MACRO_BATCH}"
  --micro-batch-size "${MICRO_BATCH}"
  --macro-max-steps "${MACRO_STEPS}"
  --micro-max-steps "${MICRO_STEPS}"
  --fid-every "${FID_EVERY}"
  --fid-num-samples "${FID_NUM_SAMPLES}"
  --sample-every "${SAMPLE_EVERY}"
  --num-samples "${SAMPLE_COUNT}"
  --sampler "${SAMPLER}"
  --sample-steps "${SAMPLE_STEPS}"
  --temperature "${SAMPLE_TEMPERATURE}"
)

if [[ -n "${MACRO_RESUME}" ]]; then
  command+=(--macro-resume "${MACRO_RESUME}")
fi
if [[ -n "${MICRO_RESUME}" ]]; then
  command+=(--micro-resume "${MICRO_RESUME}")
fi
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  command+=(--disable-wandb)
fi

printf 'Training command:\n'
printf '  %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${command[@]}"
