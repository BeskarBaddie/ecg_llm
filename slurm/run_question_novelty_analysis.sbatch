#!/bin/bash
#SBATCH --job-name=q_novelty
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
DATA_DIR="${PROJECT_ROOT}/outputs/ecgqa_scp_binary_all_codes"

cd "${CODE_DIR}"
mkdir -p "${PROJECT_ROOT}/slurm_logs"

"${PYTHON}" analyze_question_novelty.py \
  --train-path "${DATA_DIR}/ecgqa_scp_binary_train.jsonl" \
  --val-path "${DATA_DIR}/ecgqa_scp_binary_val.jsonl" \
  --test-path "${DATA_DIR}/ecgqa_scp_binary_test.jsonl" \
  --val-predictions-path "${PROJECT_ROOT}/outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl" \
  --test-predictions-path "${PROJECT_ROOT}/outputs/adapter_3b_test_threshold_evaluation_predictions.jsonl" \
  --output-dir "${PROJECT_ROOT}/outputs/question_novelty_analysis"

