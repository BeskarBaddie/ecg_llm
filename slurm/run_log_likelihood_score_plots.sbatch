#!/bin/bash
#SBATCH --job-name=score_plots
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/data/engs-chi-biosignal/reub0692
CODE_DIR="${PROJECT_ROOT}/ProjectCode"
PYTHON="${PROJECT_ROOT}/envs/ecgllm/bin/python"
SCP_METADATA="${PROJECT_ROOT}/outputs/scp_code_comparison_smollm_vs_llama_with_descriptions.csv"

cd "${CODE_DIR}"
mkdir -p "${PROJECT_ROOT}/slurm_logs"

METADATA_ARGS=()
if [[ -f "${SCP_METADATA}" ]]; then
  METADATA_ARGS=(--scp-metadata-csv "${SCP_METADATA}")
fi

"${PYTHON}" plot_log_likelihood_scores.py \
  --predictions-path "${PROJECT_ROOT}/outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl" \
  "${METADATA_ARGS[@]}" \
  --output-dir "${PROJECT_ROOT}/outputs/log_likelihood_score_analysis" \
  --figure-dir "${PROJECT_ROOT}/figures/log_likelihood_score_analysis"

