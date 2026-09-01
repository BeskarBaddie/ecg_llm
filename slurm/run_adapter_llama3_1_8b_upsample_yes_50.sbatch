#!/bin/bash
#SBATCH --job-name=adapter_8b_up50
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/data/engs-chi-biosignal/reub0692"
PYTHON="${PROJECT_ROOT}/envs/ecgllm/bin/python"
CODE_DIR="${PROJECT_ROOT}/ProjectCode"
MODEL_DIR="${PROJECT_ROOT}/hf_models/Llama-3.1-8B-Instruct"

TRAIN_PATH="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_train.jsonl"
VAL_PATH="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl"
EMBEDDING_BANK="${PROJECT_ROOT}/embeddings/ptbxl_csfm_embeddings.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/adapter_full_all_codes"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}" "${PROJECT_ROOT}/checkpoints"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  echo "Copy or download Llama-3.1-8B-Instruct to this path before submitting." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

"${PYTHON}" "${CODE_DIR}/train_ecg_soft_prompt_adapter.py" \
  --train-path "${TRAIN_PATH}" \
  --val-path "${VAL_PATH}" \
  --embedding-bank "${EMBEDDING_BANK}" \
  --llm-model "${MODEL_DIR}" \
  --use-safetensors \
  --torch-dtype float16 \
  --adapter-dtype float32 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --epochs 20 \
  --learning-rate 1e-5 \
  --weight-decay 0.0 \
  --train-sampling-mode upsample_yes_to_no \
  --train-sampling-seed 42 \
  --grad-clip 0.5 \
  --gradient-checkpointing \
  --device cuda \
  --eval-mode forced_choice \
  --run-note "20 epochs llama 3.1 8b upsample yes to 50 percent float16 batch1 accum2 gradcheckpoint real train real val" \
  --results-path "${OUTPUT_DIR}/adapter_full_all_codes_results.json" \
  --predictions-path "${OUTPUT_DIR}/adapter_full_all_codes_predictions.jsonl" \
  --checkpoint-dir "${CHECKPOINT_DIR}"

echo "Job finished at $(date)"
