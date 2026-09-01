#!/bin/bash
#SBATCH --job-name=extract_csfm_base
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
PTBXL_ROOT="${PROJECT_ROOT}/data/ptbxl/1.0.3"
CSFM_REPO="${PROJECT_ROOT}/external/Cardiac-Sensing-FM"
CSFM_CHECKPOINT="${CSFM_REPO}/pretrained/CSFM_Base.pth"
OUTPUT_JSONL="${PROJECT_ROOT}/embeddings/ptbxl_csfm_base_embeddings.jsonl"
MANIFEST_JSON="${PROJECT_ROOT}/embeddings/ptbxl_csfm_base_embeddings_manifest.json"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${PROJECT_ROOT}/embeddings"

if [[ ! -f "${CSFM_CHECKPOINT}" ]]; then
  echo "Missing CSFM Base checkpoint: ${CSFM_CHECKPOINT}" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Checkpoint: ${CSFM_CHECKPOINT}"
nvidia-smi || true

"${PYTHON}" "${CODE_DIR}/extract_ptbxl_csfm_embeddings.py" \
  --ptbxl-root "${PTBXL_ROOT}" \
  --csfm-repo-root "${CSFM_REPO}" \
  --csfm-checkpoint-path "${CSFM_CHECKPOINT}" \
  --csfm-variant Base \
  --output-jsonl "${OUTPUT_JSONL}" \
  --manifest-path "${MANIFEST_JSON}" \
  --prefer hr \
  --device cuda \
  --resume \
  --progress-every 1000

echo "Job finished at $(date)"
