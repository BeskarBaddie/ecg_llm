#!/bin/bash
#SBATCH --job-name=qual_examples
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
PTBXL_ROOT="${PROJECT_ROOT}/data/ptbxl/1.0.3"
PTBXL_METADATA="${PTBXL_ROOT}/ptbxl_database.csv"

cd "${CODE_DIR}"
mkdir -p "${PROJECT_ROOT}/slurm_logs"

PLOT_ARGS=()
if [[ -d "${PTBXL_ROOT}" && -f "${PTBXL_METADATA}" ]]; then
  PLOT_ARGS=(--ptbxl-root "${PTBXL_ROOT}" --ptbxl-metadata "${PTBXL_METADATA}")
else
  echo "PTB-XL waveform path not found; exporting table-only qualitative examples."
fi

"${PYTHON}" export_qualitative_examples.py \
  --predictions-path "${PROJECT_ROOT}/outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl" \
  "${PLOT_ARGS[@]}" \
  --output-dir "${PROJECT_ROOT}/outputs/qualitative_examples" \
  --figure-dir "${PROJECT_ROOT}/figures/qualitative_examples" \
  --per-bucket 8 \
  --lead II \
  --sampling lr

