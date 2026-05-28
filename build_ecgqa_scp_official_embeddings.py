from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

import build_ecgqa_scp_subset as subset
from src.config import ECG_QA_TRAIN, ECG_QA_VAL, OUTPUT_DIR, PTBXL_DIR, PTBXL_METADATA
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal


CSFM_REPO_ROOT = Path.home() / "Desktop" / "Oxford" / "Dissertation" / "Cardiac-Sensing-FM"
CSFM_CHECKPOINT_PATH = CSFM_REPO_ROOT / "pretrained" / "CSFM_tiny.pth"
SCP_STATEMENTS_PATH = PTBXL_DIR / "scp_statements.csv"

OUTPUT_ALL_PATH = OUTPUT_DIR / "ecgqa_scp_binary_subset.jsonl"
OUTPUT_TRAIN_PATH = OUTPUT_DIR / "ecgqa_scp_binary_train.jsonl"
OUTPUT_VAL_PATH = OUTPUT_DIR / "ecgqa_scp_binary_val.jsonl"
OUTPUT_STATS_PATH = OUTPUT_DIR / "ecgqa_scp_binary_subset_stats.json"

PREFER = "hr"
FS_ORIGINAL = 500

if not CSFM_REPO_ROOT.exists():
    raise FileNotFoundError(f"Could not find CSFM repo at: {CSFM_REPO_ROOT}")

sys.path.insert(0, str(CSFM_REPO_ROOT))

from network.model import CSFM_model  # noqa: E402
from utils.preprocess import preprocess_ecg  # noqa: E402


# Function: Load the pretrained frozen CSFM encoder.
# Inputs: CSFM checkpoint path.
# Outputs: Model, torch device, and checkpoint loading metadata.
def load_pretrained_csfm(checkpoint_path: Path) -> tuple[nn.Module, str, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CSFM checkpoint not found: {checkpoint_path}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CSFM_model("Tiny").to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_state_dict = {
        key.replace("encoder.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("encoder.") and "mlp_head" not in key
    }
    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
    model.mlp_head = nn.Identity()
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "loaded_keys": len(encoder_state_dict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, device, metadata


# Function: Filter ECG-QA samples to the selected SCP yes/no subset.
# Inputs: Raw ECG-QA samples, split name, SCP statements, and selected target codes.
# Outputs: Filtered item dictionaries carrying normalized metadata.
def filter_samples(
    samples: List[Dict[str, Any]],
    split: str,
    scp_statements,
    target_scp_codes: set[str],
) -> List[Dict[str, Any]]:
    attribute_to_code = subset.build_attribute_to_scp_code_map(scp_statements)
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
                "split": split,
                "ecg_id": int(ecg_ids[0]),
                "answer": answer,
                "attribute": attribute,
                "target_scp_code": target_scp_code,
            }
        )

    return filtered


# Function: Extract or reuse one CSFM embedding for a PTB-XL ECG.
# Inputs: ECG ID, PTB-XL metadata, CSFM model, device, and cache dictionary.
# Outputs: Cached embedding payload, or None when local signal files are missing.
def get_embedding_item(
    ecg_id: int,
    metadata,
    model: nn.Module,
    device: str,
    embedding_cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any] | None:
    if ecg_id in embedding_cache:
        return embedding_cache[ecg_id]

    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer=PREFER)
    except FileNotFoundError as exc:
        print(f"Skipped ecg_id={ecg_id}: {exc}", flush=True)
        return None

    signal = preprocess_ecg(signal, fs=FS_ORIGINAL).astype(np.float32)
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    channels = np.arange(signal.shape[0])

    with torch.no_grad():
        embedding = model(x, channels)

    embedding_cache[ecg_id] = {
        "embedding": embedding.squeeze(0).detach().cpu().numpy().tolist(),
        "embedding_dim": int(embedding.squeeze(0).numel()),
        "signal_shape": list(signal.shape),
    }
    return embedding_cache[ecg_id]


# Function: Convert filtered ECG-QA items into final dataset rows with CSFM embeddings.
# Inputs: Filtered items, PTB-XL metadata, SCP statements, CSFM model, and device.
# Outputs: Dataset rows and skipped-row counts.
def build_rows(
    items: List[Dict[str, Any]],
    metadata,
    metadata_by_ecg_id,
    scp_statements,
    model: nn.Module,
    device: str,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    embedding_cache: Dict[int, Dict[str, Any]] = {}

    for idx, item in enumerate(items, start=1):
        sample = item["sample"]
        ecg_id = item["ecg_id"]
        embedding_item = get_embedding_item(ecg_id, metadata, model, device, embedding_cache)
        if embedding_item is None:
            skipped["missing_signal"] += 1
            continue

        metadata_row = metadata_by_ecg_id.loc[ecg_id]
        target_scp_code = item["target_scp_code"]
        rows.append(
            {
                "ecg_id": ecg_id,
                "question": sample.get("question"),
                "answer": item["answer"],
                "label": subset.ANSWER_TO_LABEL[item["answer"]],
                "question_type": sample.get("question_type"),
                "attribute_type": sample.get("attribute_type"),
                "attribute": item["attribute"],
                "target_scp_code": target_scp_code,
                "target_scp_statement": subset.get_scp_statement(scp_statements, target_scp_code),
                "ptbxl_scp_codes": metadata_row["scp_codes_parsed"],
                "embedding": embedding_item["embedding"],
                "embedding_dim": embedding_item["embedding_dim"],
                "signal_shape": embedding_item["signal_shape"],
                "source_dataset": str(ECG_QA_TRAIN if item["split"] == "train" else ECG_QA_VAL),
                "official_split": item["split"],
                "split": item["split"],
            }
        )

        print(
            f"[{idx}/{len(items)}] split={item['split']} ecg_id={ecg_id} "
            f"code={target_scp_code}",
            flush=True,
        )

    skipped["unique_embeddings_extracted"] = len(embedding_cache)
    return rows, dict(skipped)


# Function: Summarize final rows with split, answer, and SCP-code counts.
# Inputs: Final dataset rows, skipped counts, and CSFM loading metadata.
# Outputs: JSON-serializable statistics dictionary.
def summarize_rows(
    rows: List[Dict[str, Any]],
    skipped: Dict[str, int],
    csfm_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    stats = subset.summarize_rows(rows, skipped)
    stats["target_scp_codes"] = sorted(subset.DEFAULT_TARGET_SCP_CODES)
    stats["split_source"] = "official_ecgqa_template_split"
    stats["embedding_source"] = "single_run_pretrained_csfm"
    stats["csfm"] = csfm_metadata
    stats["ecgqa_train_path"] = str(ECG_QA_TRAIN)
    stats["ecgqa_val_path"] = str(ECG_QA_VAL)
    stats["scp_statements_path"] = str(SCP_STATEMENTS_PATH)
    stats["ptbxl_metadata_path"] = str(PTBXL_METADATA)
    return stats


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: Output path and rows to serialize.
# Outputs: None; writes file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build the official ECG-QA SCP subset from one pretrained CSFM run.
# Inputs: Local path constants for ECG-QA, PTB-XL, CSFM, and outputs.
# Outputs: Train/validation JSONL files and summary statistics.
def main() -> None:
    print("Loading SCP statements...")
    scp_statements = subset.load_scp_statement_table(SCP_STATEMENTS_PATH)

    print("Loading ECG-QA official splits...")
    train_samples = load_ecgqa_json(ECG_QA_TRAIN)
    val_samples = load_ecgqa_json(ECG_QA_VAL)

    target_scp_codes = set(subset.DEFAULT_TARGET_SCP_CODES)
    train_items = filter_samples(train_samples, "train", scp_statements, target_scp_codes)
    val_items = filter_samples(val_samples, "val", scp_statements, target_scp_codes)
    items = train_items + val_items

    print("Filtered train rows:", len(train_items))
    print("Filtered val rows:", len(val_items))
    print("Filtered unique ECGs:", len({item["ecg_id"] for item in items}))

    print("Loading PTB-XL metadata...")
    metadata = load_ptbxl_metadata(PTBXL_METADATA)
    metadata_by_ecg_id = subset.load_ptbxl_metadata_by_ecg_id(PTBXL_METADATA)

    print("Loading pretrained CSFM...")
    model, device, csfm_metadata = load_pretrained_csfm(CSFM_CHECKPOINT_PATH)
    print("CSFM metadata:", csfm_metadata)

    rows, skipped = build_rows(
        items,
        metadata=metadata,
        metadata_by_ecg_id=metadata_by_ecg_id,
        scp_statements=scp_statements,
        model=model,
        device=device,
    )
    if not rows:
        raise RuntimeError("No rows were built.")

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    stats = summarize_rows(rows, skipped, csfm_metadata)

    write_jsonl(OUTPUT_ALL_PATH, rows)
    write_jsonl(OUTPUT_TRAIN_PATH, train_rows)
    write_jsonl(OUTPUT_VAL_PATH, val_rows)

    OUTPUT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved all rows:", OUTPUT_ALL_PATH)
    print("Saved train rows:", OUTPUT_TRAIN_PATH)
    print("Saved val rows:", OUTPUT_VAL_PATH)
    print("Saved stats:", OUTPUT_STATS_PATH)
    print("Total questions:", stats["total_questions"])
    print("Unique ECGs:", stats["unique_ecgs"])
    print("Splits:", stats["splits"])
    print("Answers:", stats["answers"])
    print("Questions per code:", stats["questions_per_code"])
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
