#!/usr/bin/env bash
set -euo pipefail

# Kaggle 2x T4 fp16 + W&B launcher for the discrete-token Latent-RAMA pipeline.
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   LOG_DIR=logs
#   LOG_FILE=logs/kaggle_t4x2_train_<timestamp>.log
#   IMAGES_DIR=/kaggle/input/celeba-256
#   LATENTS_DIR=/kaggle/working/rama_latents
#   OUTPUT_ROOT=/kaggle/working/rama_outputs
#   WANDB_ENTITY=your_entity
#   MACRO_STEPS=200000
#   MICRO_STEPS=200000
#   MACRO_BATCH=32
#   MICRO_BATCH=16
#   CACHE_BATCH=32
#   FID_EVERY=10000
#   FID_NUM_SAMPLES=1024
#   SAMPLE_EVERY=2000
#   QUANT_BATCH=16
#   QUANT_MAX_BATCHES=200
#   QUANT_PERCENTILE=99.5
#   NUM_BINS=256
#   TRAIN_WORKERS=2
#   SAMPLE_COUNT=16
#   SAMPLE_STEPS=50
#   SAMPLER=heun
#   SAMPLE_TEMPERATURE=1.0
#   SAMPLE_ARGMAX=1
#   FORCE_CACHE=1
#   SKIP_CACHE=1
#   SKIP_MACRO=1
#   SKIP_QUANT_RECONSTRUCTION=1
#   SKIP_MICRO=1
#   SKIP_SAMPLING=1
#   DISABLE_WANDB=1
#   DRY_RUN=1
#   SKIP_INSTALL=1

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-logs}"
IMAGES_DIR="${IMAGES_DIR:-data/celeba256}"
LATENTS_DIR="${LATENTS_DIR:-/kaggle/working/rama_latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/rama_outputs}"
MACRO_OUT="${MACRO_OUT:-${OUTPUT_ROOT}/macro_flow}"
MICRO_OUT="${MICRO_OUT:-${OUTPUT_ROOT}/micro_rama}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/kaggle/working/hf_cache}"
BASES_PATH="${BASES_PATH:-cache/rama_bases_p256_d16.pt}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-/kaggle/working/rama_tokenizer_config.pt}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-stabilityai/sd-vae-ft-mse}"
MACRO_CONFIG="${MACRO_CONFIG:-configs/celeba256_sdvae_macro.yaml}"
MICRO_CONFIG="${MICRO_CONFIG:-configs/celeba256_sdvae_micro.yaml}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"

MACRO_STEPS="${MACRO_STEPS:-200000}"
MICRO_STEPS="${MICRO_STEPS:-200000}"
MACRO_BATCH="${MACRO_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-64}"
CACHE_BATCH="${CACHE_BATCH:-32}"
FID_EVERY="${FID_EVERY:-10000}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-1024}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2000}"
QUANT_BATCH="${QUANT_BATCH:-16}"
QUANT_MAX_BATCHES="${QUANT_MAX_BATCHES:-200}"
QUANT_PERCENTILE="${QUANT_PERCENTILE:-99.5}"
NUM_BINS="${NUM_BINS:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLER="${SAMPLER:-heun}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-2}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-fp16}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"

extra_args=()

add_bool_flag() {
  local env_name="$1"
  local flag="$2"
  if [[ "${!env_name:-0}" == "1" ]]; then
    extra_args+=("${flag}")
  fi
}

add_bool_flag FORCE_CACHE --force-cache
add_bool_flag SKIP_CACHE --skip-cache
add_bool_flag SKIP_MACRO --skip-macro
add_bool_flag SKIP_QUANT_RECONSTRUCTION --skip-quant-reconstruction
add_bool_flag SKIP_MICRO --skip-micro
add_bool_flag SKIP_SAMPLING --skip-sampling
add_bool_flag SAMPLE_ARGMAX --sample-argmax
add_bool_flag DRY_RUN --dry-run

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/kaggle_t4x2_train_${RUN_ID}.log}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging run to ${LOG_FILE}"
echo "Run id: ${RUN_ID}"
echo "Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

print_file_if_exists() {
  local label="$1"
  local path="$2"

  echo
  echo "===== ${label}: ${path} ====="
  if [[ -f "${path}" ]]; then
    sed 's/^/  /' "${path}"
  else
    echo "  Missing: ${path}"
  fi
}

cat <<EOF

===== Resolved launcher config =====
PYTHON_BIN=${PYTHON_BIN}
IMAGES_DIR=${IMAGES_DIR}
LATENTS_DIR=${LATENTS_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT}
MACRO_OUT=${MACRO_OUT}
MICRO_OUT=${MICRO_OUT}
HF_CACHE_DIR=${HF_CACHE_DIR}
BASES_PATH=${BASES_PATH}
TOKENIZER_CONFIG=${TOKENIZER_CONFIG}
VAE_CHECKPOINT=${VAE_CHECKPOINT}
MACRO_CONFIG=${MACRO_CONFIG}
MICRO_CONFIG=${MICRO_CONFIG}
IMAGE_SIZE=${IMAGE_SIZE}
DTYPE=${DTYPE}
DEVICE=${DEVICE}
MACRO_STEPS=${MACRO_STEPS}
MICRO_STEPS=${MICRO_STEPS}
MACRO_BATCH=${MACRO_BATCH}
MICRO_BATCH=${MICRO_BATCH}
CACHE_BATCH=${CACHE_BATCH}
FID_EVERY=${FID_EVERY}
FID_NUM_SAMPLES=${FID_NUM_SAMPLES}
SAMPLE_EVERY=${SAMPLE_EVERY}
SAMPLE_COUNT=${SAMPLE_COUNT}
QUANT_BATCH=${QUANT_BATCH}
QUANT_MAX_BATCHES=${QUANT_MAX_BATCHES}
QUANT_PERCENTILE=${QUANT_PERCENTILE}
NUM_BINS=${NUM_BINS}
TRAIN_WORKERS=${TRAIN_WORKERS}
SAMPLE_COUNT=${SAMPLE_COUNT}
SAMPLE_STEPS=${SAMPLE_STEPS}
SAMPLER=${SAMPLER}
SAMPLE_TEMPERATURE=${SAMPLE_TEMPERATURE}
ACCELERATE_NUM_PROCESSES=${ACCELERATE_NUM_PROCESSES}
ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION}
WANDB_PROJECT=${WANDB_PROJECT:-rama}
WANDB_ENTITY=${WANDB_ENTITY:-}
SKIP_INSTALL=${SKIP_INSTALL:-0}
FORCE_CACHE=${FORCE_CACHE:-0}
SKIP_CACHE=${SKIP_CACHE:-0}
SKIP_MACRO=${SKIP_MACRO:-0}
SKIP_QUANT_RECONSTRUCTION=${SKIP_QUANT_RECONSTRUCTION:-0}
SKIP_MICRO=${SKIP_MICRO:-0}
SKIP_SAMPLING=${SKIP_SAMPLING:-0}
SAMPLE_ARGMAX=${SAMPLE_ARGMAX:-0}
DISABLE_WANDB=${DISABLE_WANDB}
DRY_RUN=${DRY_RUN:-0}
EXTRA_ARGS=${extra_args[*]:-}
EOF

print_file_if_exists "Macro YAML config" "${MACRO_CONFIG}"
print_file_if_exists "Micro YAML config" "${MICRO_CONFIG}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "${PYTHON_BIN}" -m wandb login "${WANDB_API_KEY}"
fi

if [[ "${SKIP_CACHE:-0}" != "1" && "${DRY_RUN:-0}" != "1" && ! -d "${IMAGES_DIR}" ]]; then
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

if [[ "${SKIP_CACHE:-0}" != "1" && "${DRY_RUN:-0}" != "1" && "${LATENTS_DIR}" == /kaggle/input/* ]]; then
  cat >&2 <<EOF
Latent output directory is under Kaggle's read-only input mount:
  LATENTS_DIR=${LATENTS_DIR}

Write generated latents to /kaggle/working instead, for example:
  export LATENTS_DIR=/kaggle/working/rama_latents

Use SKIP_CACHE=1 only if the latents already exist in a mounted input dataset.
EOF
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-rama}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_command() {
  echo
  echo "\$ $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  "$@"
}

train_with_accelerate() {
  local script="$1"
  shift
  local command=(accelerate launch)
  if [[ -n "${ACCELERATE_NUM_PROCESSES:-}" ]]; then
    if (( ACCELERATE_NUM_PROCESSES > 1 )); then
      command+=(--multi_gpu)
    fi
    command+=(--num_processes "${ACCELERATE_NUM_PROCESSES}")
  fi
  if [[ -n "${ACCELERATE_MIXED_PRECISION:-}" ]]; then
    command+=(--mixed_precision "${ACCELERATE_MIXED_PRECISION}")
  fi
  command+=("${script}" "$@")
  run_command "${command[@]}"
}

add_optional_cli_flag() {
  local -n args_ref="$1"
  local flag="$2"
  local enabled="${3:-0}"
  if [[ "${enabled}" == "1" ]]; then
    args_ref+=("${flag}")
  fi
}

resolve_checkpoint_path() {
  local out_dir="$1"
  local step="$2"
  local expected="${out_dir}/checkpoints/step_$(printf '%08d' "${step}").pt"
  if [[ -f "${expected}" ]]; then
    printf '%s\n' "${expected}"
    return
  fi
  find "${out_dir}/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -n 1
}

if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
  run_command "${PYTHON_BIN}" scripts/cache_sdvae_latents.py \
    --images "${IMAGES_DIR}" \
    --out "${LATENTS_DIR}" \
    --checkpoint "${VAE_CHECKPOINT}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --batch-size "${CACHE_BATCH}" \
    --num-workers "${TRAIN_WORKERS}" \
    --image-size "${IMAGE_SIZE}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --store-components
else
  echo "Skipping latent cache; using cached latents under ${LATENTS_DIR}"
fi

run_command "${PYTHON_BIN}" scripts/estimate_quant_bound.py \
  --latent-cache "${LATENTS_DIR}" \
  --bases "${BASES_PATH}" \
  --output "${TOKENIZER_CONFIG}" \
  --num-bins "${NUM_BINS}" \
  --percentile "${QUANT_PERCENTILE}" \
  --max-batches "${QUANT_MAX_BATCHES}" \
  --batch-size "${QUANT_BATCH}" \
  --device "${DEVICE}"

if [[ "${SKIP_QUANT_RECONSTRUCTION:-0}" != "1" ]]; then
  run_command "${PYTHON_BIN}" scripts/test_quant_reconstruction.py \
    --latent-cache "${LATENTS_DIR}" \
    --bases "${BASES_PATH}" \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --out outputs/quantization_tests/vae_vs_macro_vs_quant.png \
    --checkpoint "${VAE_CHECKPOINT}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}"
else
  echo "Skipping quantization reconstruction"
fi

if [[ "${SKIP_MACRO:-0}" != "1" && "${SKIP_MICRO:-0}" != "1" ]]; then
  joint_args=(
    --macro-config "${MACRO_CONFIG}" \
    --micro-config "${MICRO_CONFIG}" \
    --latents "${LATENTS_DIR}" \
    --macro-out "${MACRO_OUT}" \
    --micro-out "${MICRO_OUT}" \
    --micro-type categorical \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --bases "${BASES_PATH}" \
    --num-workers "${TRAIN_WORKERS}" \
    --macro-batch-size "${MACRO_BATCH}" \
    --micro-batch-size "${MICRO_BATCH}" \
    --macro-max-steps "${MACRO_STEPS}" \
    --micro-max-steps "${MICRO_STEPS}" \
    --fid-every "${FID_EVERY}" \
    --fid-num-samples "${FID_NUM_SAMPLES}" \
    --sample-every "${SAMPLE_EVERY}" \
    --num-samples "${SAMPLE_COUNT}" \
    --sampler "${SAMPLER}" \
    --sample-steps "${SAMPLE_STEPS}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
  )
  add_optional_cli_flag joint_args --disable-wandb "${DISABLE_WANDB:-0}"
  add_optional_cli_flag joint_args --sample-argmax "${SAMPLE_ARGMAX:-0}"
  train_with_accelerate scripts/train_joint_macro_micro.py "${joint_args[@]}" "${extra_args[@]:+"${extra_args[@]}"}"
elif [[ "${SKIP_MACRO:-0}" != "1" ]]; then
  macro_args=(
    --config "${MACRO_CONFIG}" \
    --latents "${LATENTS_DIR}" \
    --out "${MACRO_OUT}" \
    --num-workers "${TRAIN_WORKERS}" \
    --batch-size "${MACRO_BATCH}" \
    --max-steps "${MACRO_STEPS}" \
    --fid-every "${FID_EVERY}" \
    --fid-num-samples "${FID_NUM_SAMPLES}"
  )
  add_optional_cli_flag macro_args --disable-wandb "${DISABLE_WANDB:-0}"
  train_with_accelerate scripts/train_macro_flow.py "${macro_args[@]}" "${extra_args[@]:+"${extra_args[@]}"}"
  echo "Skipping micro training"
elif [[ "${SKIP_MICRO:-0}" != "1" ]]; then
  micro_args=(
    --config "${MICRO_CONFIG}" \
    --latents "${LATENTS_DIR}" \
    --out "${MICRO_OUT}" \
    --micro-type categorical \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --bases "${BASES_PATH}" \
    --num-workers "${TRAIN_WORKERS}" \
    --batch-size "${MICRO_BATCH}" \
    --max-steps "${MICRO_STEPS}" \
    --fid-every "${FID_EVERY}" \
    --fid-num-samples "${FID_NUM_SAMPLES}"
  )
  add_optional_cli_flag micro_args --disable-wandb "${DISABLE_WANDB:-0}"
  train_with_accelerate scripts/train_micro_rama.py "${micro_args[@]}" "${extra_args[@]:+"${extra_args[@]}"}"
  echo "Skipping macro training"
else
  echo "Skipping macro training"
  echo "Skipping micro training"
fi

if [[ "${SKIP_SAMPLING:-0}" == "1" ]]; then
  echo "Skipping sampling"
  exit 0
fi

macro_checkpoint="$(resolve_checkpoint_path "${MACRO_OUT}" "${MACRO_STEPS}")"
micro_checkpoint="$(resolve_checkpoint_path "${MICRO_OUT}" "${MICRO_STEPS}")"

if [[ -z "${macro_checkpoint}" || -z "${micro_checkpoint}" ]]; then
  echo "Could not resolve macro or micro checkpoint for sampling" >&2
  exit 1
fi

run_command "${PYTHON_BIN}" scripts/sample_macro_flow.py \
  --checkpoint "${macro_checkpoint}" \
  --out outputs/macro_samples/macro_only_samples.png \
  --num-samples "${SAMPLE_COUNT}" \
  --sampler "${SAMPLER}" \
  --steps "${SAMPLE_STEPS}" \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --cache-dir "${HF_CACHE_DIR}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}"

run_command "${PYTHON_BIN}" scripts/reconstruct_micro_real_zl.py \
  --micro-checkpoint "${micro_checkpoint}" \
  --latent-cache "${LATENTS_DIR}" \
  --bases "${BASES_PATH}" \
  --tokenizer-config "${TOKENIZER_CONFIG}" \
  --out outputs/micro_reconstructions/real_zL_macro_vs_micro_argmax.png \
  --vae-checkpoint "${VAE_CHECKPOINT}" \
  --cache-dir "${HF_CACHE_DIR}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}"

if [[ "${SAMPLE_ARGMAX:-0}" == "1" ]]; then
  run_command "${PYTHON_BIN}" scripts/sample_full_model.py \
    --macro-checkpoint "${macro_checkpoint}" \
    --micro-checkpoint "${micro_checkpoint}" \
    --bases "${BASES_PATH}" \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --out outputs/full_samples/generated_zL_macro_plus_micro.png \
    --macro-out outputs/full_samples/generated_zL_macro_only.png \
    --micro-out outputs/full_samples/generated_zL_micro_only.png \
    --comparison-out outputs/full_samples/generated_zL_macro_micro_full.png \
    --num-samples "${SAMPLE_COUNT}" \
    --sampler "${SAMPLER}" \
    --steps "${SAMPLE_STEPS}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --vae-checkpoint "${VAE_CHECKPOINT}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --use-argmax
else
  run_command "${PYTHON_BIN}" scripts/sample_full_model.py \
    --macro-checkpoint "${macro_checkpoint}" \
    --micro-checkpoint "${micro_checkpoint}" \
    --bases "${BASES_PATH}" \
    --tokenizer-config "${TOKENIZER_CONFIG}" \
    --out outputs/full_samples/generated_zL_macro_plus_micro.png \
    --macro-out outputs/full_samples/generated_zL_macro_only.png \
    --micro-out outputs/full_samples/generated_zL_micro_only.png \
    --comparison-out outputs/full_samples/generated_zL_macro_micro_full.png \
    --num-samples "${SAMPLE_COUNT}" \
    --sampler "${SAMPLER}" \
    --steps "${SAMPLE_STEPS}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --vae-checkpoint "${VAE_CHECKPOINT}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}"
fi
