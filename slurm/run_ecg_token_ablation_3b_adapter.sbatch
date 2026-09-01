#!/bin/bash
#SBATCH --job-name=ecg_tok_ablate
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/ecg_token_ablation_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/ecg_token_ablation_%j.err

set -euo pipefail

PROJECT_ROOT=/data/engs-chi-biosignal/reub0692/ProjectCode
PYTHON=/data/engs-chi-biosignal/reub0692/envs/ecgllm/bin/python

cd "${PROJECT_ROOT}"

"${PYTHON}" evaluate_ecg_token_ablation.py \
  --val-path /data/engs-chi-biosignal/reub0692/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl \
  --embedding-bank /data/engs-chi-biosignal/reub0692/embeddings/ptbxl_csfm_embeddings.jsonl \
  --llm-model /data/engs-chi-biosignal/reub0692/hf_models/Llama-3.2-3B-Instruct \
  --checkpoint-dir /data/engs-chi-biosignal/reub0692/checkpoints/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes \
  --results-path /data/engs-chi-biosignal/reub0692/outputs/ecg_token_ablation_3b_adapter_results.json \
  --predictions-path /data/engs-chi-biosignal/reub0692/outputs/ecg_token_ablation_3b_adapter_predictions.jsonl \
  --modes real,zero,mean,shuffled,random \
  --batch-size 1 \
  --max-length 96 \
  --torch-dtype float16 \
  --use-safetensors \
  --device cuda \
  --seed 42
