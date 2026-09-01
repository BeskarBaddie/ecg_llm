#!/bin/bash
#SBATCH --job-name=adapter_8b_up40_test
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/data/engs-chi-biosignal/reub0692"
PYTHON="${PROJECT_ROOT}/envs/ecgllm/bin/python"
CODE_DIR="${PROJECT_ROOT}/ProjectCode"
MODEL_DIR="${PROJECT_ROOT}/hf_models/Llama-3.1-8B-Instruct"

TEST_PATH="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_test.jsonl"
EMBEDDING_BANK="${PROJECT_ROOT}/embeddings/ptbxl_csfm_embeddings.jsonl"
THRESHOLD_CONFIG="${PROJECT_ROOT}/outputs/threshold_tuning_llama8b_partial_upsample_40/threshold_tuning_results.json"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/20260725_172120_20_epochs_llama_3_1_8b_partial_upsample_yes_fraction_0_40_float16_batch1_accum2_adapter_full_all_codes"

RESULTS_PATH="${PROJECT_ROOT}/outputs/adapter_8b_partial_upsample_test_threshold_evaluation_results.json"
PREDICTIONS_PATH="${PROJECT_ROOT}/outputs/adapter_8b_partial_upsample_test_threshold_evaluation_predictions.jsonl"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/outputs"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 2
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

if [[ ! -f "${THRESHOLD_CONFIG}" ]]; then
  echo "Missing validation threshold config: ${THRESHOLD_CONFIG}" >&2
  echo "Copy the validation-tuned threshold_tuning_results.json to ARC before submitting." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${CODE_DIR}"

echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

"${PYTHON}" evaluate_adapter_on_test_with_thresholds.py \
  --test-path "${TEST_PATH}" \
  --embedding-bank "${EMBEDDING_BANK}" \
  --llm-model "${MODEL_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --threshold-config "${THRESHOLD_CONFIG}" \
  --results-path "${RESULTS_PATH}" \
  --predictions-path "${PREDICTIONS_PATH}" \
  --max-length 96 \
  --torch-dtype float16 \
  --use-safetensors \
  --device cuda \
  --progress-every 500

echo "Job finished at $(date)"
