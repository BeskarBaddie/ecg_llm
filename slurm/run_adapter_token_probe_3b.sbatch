#!/bin/bash
#SBATCH --job-name=adapter_probe
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/engs-chi-biosignal/reub0692/slurm_logs/adapter_token_probe_%j.out
#SBATCH --error=/data/engs-chi-biosignal/reub0692/slurm_logs/adapter_token_probe_%j.err

set -euo pipefail

PROJECT_ROOT=/data/engs-chi-biosignal/reub0692/ProjectCode
PYTHON=/data/engs-chi-biosignal/reub0692/envs/ecgllm/bin/python

cd "${PROJECT_ROOT}"

"${PYTHON}" probe_adapter_token_information.py \
  --train-path /data/engs-chi-biosignal/reub0692/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_train.jsonl \
  --val-path /data/engs-chi-biosignal/reub0692/outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl \
  --embedding-bank /data/engs-chi-biosignal/reub0692/embeddings/ptbxl_csfm_embeddings.jsonl \
  --llm-model /data/engs-chi-biosignal/reub0692/hf_models/Llama-3.2-3B-Instruct \
  --checkpoint-dir /data/engs-chi-biosignal/reub0692/checkpoints/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes \
  --results-path /data/engs-chi-biosignal/reub0692/outputs/adapter_token_probe_3b_results.json \
  --feature-sets csfm,adapter_mean,adapter_flat \
  --batch-size 32 \
  --min-train-pos 20 \
  --min-val-pos 5 \
  --max-iter 1000 \
  --c-value 1.0 \
  --torch-dtype float16 \
  --use-safetensors \
  --device cuda
