#!/bin/bash
#SBATCH --job-name=csfm_topworst
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/data/engs-chi-biosignal/reub0692
CODE_DIR="${PROJECT_ROOT}/ProjectCode"
PYTHON="${PROJECT_ROOT}/envs/ecgllm/bin/python"
DATA_DIR="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes"
SCP_METADATA="${PROJECT_ROOT}/outputs/scp_code_comparison_smollm_vs_llama_with_descriptions.csv"

cd "${CODE_DIR}"
mkdir -p "${PROJECT_ROOT}/slurm_logs"

METADATA_ARGS=()
if [[ -f "${SCP_METADATA}" ]]; then
  METADATA_ARGS=(--scp-metadata-csv "${SCP_METADATA}")
fi

"${PYTHON}" evaluate_csfm_only_top_worst_codes.py \
  --train-path "${DATA_DIR}/ecgqa_scp_binary_train.jsonl" \
  --val-path "${DATA_DIR}/ecgqa_scp_binary_val.jsonl" \
  --embedding-bank "${PROJECT_ROOT}/embeddings/ptbxl_csfm_embeddings.jsonl" \
  --adapter-predictions-path "${PROJECT_ROOT}/outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl" \
  "${METADATA_ARGS[@]}" \
  --output-dir "${PROJECT_ROOT}/outputs/csfm_only_top_worst_analysis" \
  --min-support 30 \
  --n-each 8 \
  --max-iter 2000 \
  --c-value 1.0

