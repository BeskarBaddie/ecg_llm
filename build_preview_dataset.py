from __future__ import annotations
import random

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.config import ECG_QA_TRAIN, PTBXL_METADATA, OUTPUT_DIR
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal

# ---------------------------------------------------------------------
# PATH TO CSFM REPO
# ---------------------------------------------------------------------
CSFM_REPO_ROOT = Path.home() / "Desktop" / "Oxford" / "Dissertation" / "Cardiac-Sensing-FM"
if not CSFM_REPO_ROOT.exists():
    raise FileNotFoundError(f"Could not find CSFM repo at: {CSFM_REPO_ROOT}")

sys.path.insert(0, str(CSFM_REPO_ROOT))

from network.model import CSFM_model  # noqa: E402
from utils.preprocess import preprocess_ecg  # noqa: E402


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
TARGET_N = 10000
PREFER = "hr"  # use records500
FS_ORIGINAL = 500  # PTB-XL records500 is 500 Hz

OUTPUT_FILE = OUTPUT_DIR / "ecgqa_csfm_preview_10000_sv.jsonl"


def load_csfm():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CSFM_model("Tiny").to(device)
    model.eval()
    model.mlp_head = nn.Identity()

    return model, device


def find_valid_samples(samples: List[Dict[str, Any]], metadata, target_n: int) -> List[Dict[str, Any]]:
    """
    Return the first `target_n` ECG-QA samples whose PTB-XL ECG files are available.
    """
    valid = []

    for sample in samples:
        ecg_ids = sample.get("ecg_id", [])
        if not ecg_ids:
            continue

        ecg_id = int(ecg_ids[0])

        try:
            signal = load_ecg_signal(ecg_id, metadata=metadata, prefer=PREFER)
            valid.append(
                {
                    "sample": sample,
                    "ecg_id": ecg_id,
                    "signal": signal,
                }
            )
        except Exception:
            continue

        if len(valid) >= target_n:
            break

    return valid


def main():
    print("Loading ECG-QA...")
    samples = load_ecgqa_json(ECG_QA_TRAIN)
    samples = [s for s in samples if s.get("question_type") == "single-verify"] # only use single-verify for now
    print(f"After filtering: {len(samples)} samples")
    print("Shuffling dataset...")
    random.seed(42)  # reproducibility
    random.shuffle(samples)
    print(f"Loaded {len(samples)} ECG-QA samples")

    print("Loading PTB-XL metadata...")
    metadata = load_ptbxl_metadata(PTBXL_METADATA)
    print(f"Loaded {len(metadata)} PTB-XL metadata rows")

    print(f"Finding first {TARGET_N} usable samples...")
    usable = find_valid_samples(samples, metadata, TARGET_N)
    print(f"Found {len(usable)} usable samples")

    print("Loading CSFM...")
    model, device = load_csfm()

    rows_out = []

    for i, item in enumerate(usable, start=1):
        sample = item["sample"]
        ecg_id = item["ecg_id"]
        signal = item["signal"]

        # Preprocess for CSFM
        signal = preprocess_ecg(signal, fs=FS_ORIGINAL).astype(np.float32)

        # CSFM expects batch dimension
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

        # 12-lead ECG -> channels 0..11
        channels = np.arange(signal.shape[0])

        with torch.no_grad():
            embedding = model(x, channels)

        # Move to CPU and convert to plain Python lists for JSONL
        embedding_list = embedding.squeeze(0).detach().cpu().numpy().tolist()

        row = {
            "ecg_id": ecg_id,
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "question_type": sample.get("question_type"),
            "attribute_type": sample.get("attribute_type"),
            "attribute": sample.get("attribute"),
            "embedding": embedding_list,
            "embedding_dim": len(embedding_list),
            "signal_shape": list(signal.shape),
        }

        rows_out.append(row)
        print(f"[{i}/{len(usable)}] processed ecg_id={ecg_id} embedding_dim={len(embedding_list)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved {len(rows_out)} rows to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()