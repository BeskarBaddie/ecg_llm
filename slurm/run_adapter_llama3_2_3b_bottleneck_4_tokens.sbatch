#!/bin/bash
#SBATCH --job-name=adapter_3b_4tok
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/data/engs-chi-biosignal/reub0692"
PYTHON="${PROJECT_ROOT}/envs/ecgllm/bin/python"
CODE_DIR="${PROJECT_ROOT}/ProjectCode"
MODEL_DIR="${PROJECT_ROOT}/hf_models/Llama-3.2-3B-Instruct"

TRAIN_PATH="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_train.jsonl"
VAL_PATH="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl"
EMBEDDING_BANK="${PROJECT_ROOT}/embeddings/ptbxl_csfm_embeddings.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/adapter_full_all_codes"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}" "${PROJECT_ROOT}/checkpoints"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  echo "Copy or download Llama-3.2-3B-Instruct to this path before submitting." >&2
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
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --learning-rate 5e-6 \
  --weight-decay 0.0 \
  --num-soft-tokens 4 \
  --grad-clip 0.25 \
  --gradient-checkpointing \
  --device cuda \
  --eval-mode forced_choice \
  --run-note "20 epochs llama 3.2 3b bottleneck 4 soft tokens float16 llm float32 adapter lr5e-6 batch1 accum4 gradclip0.25 real train real val" \
  --results-path "${OUTPUT_DIR}/adapter_full_all_codes_results.json" \
  --predictions-path "${OUTPUT_DIR}/adapter_full_all_codes_predictions.jsonl" \
  --checkpoint-dir "${CHECKPOINT_DIR}"

echo "Job finished at $(date)"
