from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

import build_ecgqa_scp_subset as subset
from src.config import ECG_QA_VAL, OUTPUT_DIR, PTBXL_DIR, PTBXL_METADATA
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal


CSFM_REPO_ROOT = Path.home() / "Desktop" / "Oxford" / "Dissertation" / "Cardiac-Sensing-FM"
if not CSFM_REPO_ROOT.exists():
    raise FileNotFoundError(f"Could not find CSFM repo at: {CSFM_REPO_ROOT}")

sys.path.insert(0, str(CSFM_REPO_ROOT))

from network.model import CSFM_model  # noqa: E402
from utils.preprocess import preprocess_ecg  # noqa: E402


OUTPUT_FILE = OUTPUT_DIR / "ecgqa_csfm_valid_scp_sv.jsonl"
SCP_STATEMENTS_PATH = PTBXL_DIR / "scp_statements.csv"
PREFER = "hr"
FS_ORIGINAL = 500


# Function: Load the frozen CSFM encoder used for ECG embeddings.
# Inputs: None.
# Outputs: Initialized CSFM model and selected torch device.
def load_csfm() -> tuple[nn.Module, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CSFM_model("Tiny").to(device)
    model.eval()
    model.mlp_head = nn.Identity()

    return model, device


# Function: Identify official validation samples needed for the SCP yes/no subset.
# Inputs: Raw ECG-QA samples and SCP statement table.
# Outputs: Filtered sample dictionaries with normalized target SCP metadata.
def filter_validation_samples(
    samples: List[Dict[str, Any]],
    scp_statements,
) -> List[Dict[str, Any]]:
    attribute_to_code = subset.build_attribute_to_scp_code_map(scp_statements)
    target_scp_codes = set(subset.DEFAULT_TARGET_SCP_CODES)
    filtered: List[Dict[str, Any]] = []

    for sample in samples:
        if sample.get("question_type") != "single-verify":
            continue

        if sample.get("attribute_type") != "scp_code":
            continue

        answer = subset.normalize_text(sample.get("answer"))
        if answer not in subset.ANSWER_TO_LABEL:
            continue

        attribute = subset.normalize_text(sample.get("attribute"))
        target_scp_code = attribute_to_code.get(attribute)
        if target_scp_code not in target_scp_codes:
            continue

        ecg_ids = sample.get("ecg_id", [])
        if not ecg_ids:
            continue

        filtered.append(
            {
                "sample": sample,
                "ecg_id": int(ecg_ids[0]),
                "answer": answer,
                "attribute": attribute,
                "target_scp_code": target_scp_code,
            }
        )

    return filtered


# Function: Compute or retrieve CSFM embeddings for validation ECGs.
# Inputs: Filtered validation samples, PTB-XL metadata, CSFM model, and torch device.
# Outputs: List of JSON-serializable rows with official split and CSFM embedding.
def build_embedding_rows(
    filtered_samples: List[Dict[str, Any]],
    metadata,
    model: nn.Module,
    device: str,
) -> List[Dict[str, Any]]:
    embedding_cache: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    skipped_missing_signal = 0

    for idx, item in enumerate(filtered_samples, start=1):
        sample = item["sample"]
        ecg_id = item["ecg_id"]

        if ecg_id not in embedding_cache:
            try:
                signal = load_ecg_signal(ecg_id, metadata=metadata, prefer=PREFER)
            except FileNotFoundError as exc:
                skipped_missing_signal += 1
                print(
                    f"[{idx}/{len(filtered_samples)}] skipped ecg_id={ecg_id}: {exc}",
                    flush=True,
                )
                continue

            signal = preprocess_ecg(signal, fs=FS_ORIGINAL).astype(np.float32)
            x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
            channels = np.arange(signal.shape[0])

            with torch.no_grad():
                embedding = model(x, channels)

            embedding_cache[ecg_id] = {
                "embedding": embedding.squeeze(0).detach().cpu().numpy().tolist(),
                "signal_shape": list(signal.shape),
            }

        cached = embedding_cache[ecg_id]
        rows.append(
            {
                "ecg_id": ecg_id,
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "question_type": sample.get("question_type"),
                "attribute_type": sample.get("attribute_type"),
                "attribute": sample.get("attribute"),
                "embedding": cached["embedding"],
                "embedding_dim": len(cached["embedding"]),
                "signal_shape": cached["signal_shape"],
                "official_split": "val",
            }
        )

        print(
            f"[{idx}/{len(filtered_samples)}] "
            f"processed ecg_id={ecg_id} code={item['target_scp_code']}",
            flush=True,
        )

    if skipped_missing_signal:
        print("Skipped missing validation signal rows:", skipped_missing_signal)

    return rows


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: output path and rows to serialize.
# Outputs: None; writes file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Extract official ECG-QA validation embeddings for selected SCP yes/no questions.
# Inputs: Command-line-free local configuration constants.
# Outputs: Validation embedding JSONL written to outputs.
def main() -> None:
    print("Loading official ECG-QA validation samples...")
    samples = load_ecgqa_json(ECG_QA_VAL)
    print("Validation samples:", len(samples))

    print("Loading SCP statements...")
    scp_statements = subset.load_scp_statement_table(SCP_STATEMENTS_PATH)

    filtered_samples = filter_validation_samples(samples, scp_statements)
    if not filtered_samples:
        raise RuntimeError("No official validation samples matched the SCP yes/no filters.")

    print("Filtered validation samples:", len(filtered_samples))
    print("Unique validation ECGs:", len({item["ecg_id"] for item in filtered_samples}))

    print("Loading PTB-XL metadata...")
    metadata = load_ptbxl_metadata(PTBXL_METADATA)

    print("Loading CSFM...")
    model, device = load_csfm()

    rows = build_embedding_rows(filtered_samples, metadata, model, device)
    write_jsonl(OUTPUT_FILE, rows)

    print(f"Saved {len(rows)} rows to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
