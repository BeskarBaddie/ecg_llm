from pathlib import Path
import sys

import numpy as np
from scipy import signal
import torch
import torch.nn as nn

from src.config import ECG_QA_TRAIN, PTBXL_METADATA
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal

# EDIT THIS to the path of your CSFM repo
CSFM_REPO_ROOT = Path.home() / "Desktop" / "Oxford" / "Dissertation" / "Cardiac-Sensing-FM"

if not CSFM_REPO_ROOT.exists():
    raise FileNotFoundError(f"CSFM repo not found at {CSFM_REPO_ROOT}")

sys.path.insert(0, str(CSFM_REPO_ROOT))

print("CSFM path added:", CSFM_REPO_ROOT)

from network.model import CSFM_model  # noqa: E402
from utils.preprocess import preprocess_ecg


def find_first_available_sample(samples, metadata, max_scan=2000):
    for sample in samples[:max_scan]:
        ecg_ids = sample.get("ecg_id", [])
        if not ecg_ids:
            continue

        ecg_id = int(ecg_ids[0])
        try:
            signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
            return sample, ecg_id, signal
        except Exception:
            continue

    raise RuntimeError("Could not find a downloadable ECG sample in the scanned ECG-QA rows.")


def load_csfm_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CSFM_model("Tiny").to(device)
    model.eval()
    model.mlp_head = nn.Identity()
    return model, device


def main():
    print("Loading ECG-QA...")
    samples = load_ecgqa_json(ECG_QA_TRAIN)
    print(f"Loaded {len(samples)} ECG-QA samples")

    print("Loading PTB-XL metadata...")
    metadata = load_ptbxl_metadata(PTBXL_METADATA)
    print(f"Loaded {len(metadata)} PTB-XL rows")

    print("Finding first sample whose PTB-XL waveform is already downloaded...")
    sample, ecg_id, signal = find_first_available_sample(samples, metadata)

    print("\nSelected sample:")
    print("question:", sample.get("question"))
    print("answer:", sample.get("answer"))
    print("ecg_id:", ecg_id)
    print("signal shape:", signal.shape)

  

    signal = preprocess_ecg(signal, fs=100)
    print("Preprocessed signal shape:", signal.shape)

    print("\nLoading CSFM...")
    model, device = load_csfm_model()

    # PTB-XL 12-lead ECG: lead indices 0..11
    channels = np.arange(signal.shape[0])

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

    print("Running CSFM forward pass...")
    with torch.no_grad():
        features = model(x, channels)

    print("Embedding shape:", features.shape)
    print("Done.")


if __name__ == "__main__":
    main()