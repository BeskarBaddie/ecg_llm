from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_scp_code_performance import summarize_by_scp_code
from train_ecg_soft_prompt_adapter import load_embedding_bank, load_jsonl, write_json


DEFAULT_OUTPUT_DIR = Path("outputs/csfm_only_top_worst_analysis")


# Function: Write dictionaries to a CSV file.
# Inputs: output path and row dictionaries.
# Outputs: None; writes a CSV file.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Load optional SCP metadata CSV for descriptions.
# Inputs: optional metadata CSV path.
# Outputs: dictionary keyed by SCP code.
def load_optional_metadata(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    from analyze_scp_code_performance import load_scp_metadata

    return load_scp_metadata(path)


# Function: Select top and worst SCP codes from adapter prediction performance.
# Inputs: prediction rows, metadata, support threshold, and number of codes per side.
# Outputs: list of selected SCP codes with adapter metrics and selection group.
def select_codes_from_adapter_predictions(
    prediction_rows: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
    min_support: int,
    n_each: int,
) -> List[Dict[str, Any]]:
    per_code = summarize_by_scp_code(prediction_rows, metadata=metadata)
    eligible = [
        row for row in per_code
        if row["n"] >= min_support and row["balanced_accuracy"] is not None
    ]
    top = sorted(eligible, key=lambda row: row["balanced_accuracy"], reverse=True)[:n_each]
    worst = sorted(eligible, key=lambda row: row["balanced_accuracy"])[:n_each]

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group, rows in [("top_adapter_code", top), ("worst_adapter_code", worst)]:
        for row in rows:
            code = str(row["code"])
            if code in seen:
                continue
            seen.add(code)
            selected.append(
                {
                    "target_scp_code": code,
                    "selection_group": group,
                    "adapter_balanced_accuracy": row["balanced_accuracy"],
                    "adapter_accuracy": row["accuracy"],
                    "adapter_yes_recall": row["yes_recall"],
                    "adapter_no_recall": row["no_recall"],
                    "adapter_n": row["n"],
                    "description": row.get("description", code),
                    "clinical_group": row.get("clinical_group", ""),
                }
            )
    return selected


# Function: Build one label per ECG for each SCP code from ECG-QA rows.
# Inputs: ECG-QA rows containing ecg_id, target_scp_code, and label fields.
# Outputs: nested dictionary mapping code to ECG ID labels.
def labels_by_code_and_ecg(rows: List[Dict[str, Any]]) -> Dict[str, Dict[int, int]]:
    output: Dict[str, Dict[int, int]] = defaultdict(dict)
    conflicts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        code = str(row["target_scp_code"])
        ecg_id = int(row["ecg_id"])
        label = int(row.get("label", row.get("true_label")))
        existing = output[code].get(ecg_id)
        if existing is not None and existing != label:
            conflicts[(code, ecg_id)] += 1
            continue
        output[code][ecg_id] = label
    if conflicts:
        print(f"Warning: ignored {sum(conflicts.values())} conflicting labels.")
    return dict(output)


# Function: Build feature and label arrays for one SCP code.
# Inputs: label mapping and CSFM embedding bank.
# Outputs: ECG IDs, feature matrix, and label array.
def build_xy(
    labels: Dict[int, int],
    embedding_bank: Dict[int, np.ndarray],
) -> tuple[List[int], np.ndarray, np.ndarray]:
    ecg_ids = sorted(ecg_id for ecg_id in labels if ecg_id in embedding_bank)
    if not ecg_ids:
        return [], np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    x = np.stack([embedding_bank[ecg_id] for ecg_id in ecg_ids]).astype(np.float32)
    y = np.array([labels[ecg_id] for ecg_id in ecg_ids], dtype=np.int64)
    return ecg_ids, x, y


# Function: Train and evaluate a balanced logistic-regression classifier for one code.
# Inputs: train/validation labels and embeddings.
# Outputs: classifier metrics and support counts.
def train_eval_code(
    code: str,
    train_labels: Dict[int, int],
    val_labels: Dict[int, int],
    embedding_bank: Dict[int, np.ndarray],
    max_iter: int,
    c_value: float,
) -> Dict[str, Any]:
    train_ids, x_train, y_train = build_xy(train_labels, embedding_bank)
    val_ids, x_val, y_val = build_xy(val_labels, embedding_bank)
    result: Dict[str, Any] = {
        "target_scp_code": code,
        "train_n": int(len(y_train)),
        "val_n": int(len(y_val)),
        "train_pos": int(np.sum(y_train == 1)),
        "train_neg": int(np.sum(y_train == 0)),
        "val_pos": int(np.sum(y_val == 1)),
        "val_neg": int(np.sum(y_val == 0)),
    }
    if len(y_train) == 0 or len(y_val) == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        result["status"] = "skipped_insufficient_classes"
        return result

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            C=c_value,
            random_state=42,
        ),
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_val)
    y_prob = classifier.predict_proba(x_val)[:, 1]

    result.update(
        {
            "status": "ok",
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_val, y_prob)),
            "average_precision": float(average_precision_score(y_val, y_prob)),
            "pred_pos": int(np.sum(y_pred == 1)),
            "pred_neg": int(np.sum(y_pred == 0)),
        }
    )
    return result


# Function: Run CSFM-only classifier comparison for top and worst adapter SCP codes.
# Inputs: command-line arguments.
# Outputs: CSV and JSON summaries.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--adapter-predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--n-each", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    prediction_rows = load_jsonl(args.adapter_predictions_path)
    embedding_bank = load_embedding_bank(args.embedding_bank)
    metadata = load_optional_metadata(args.scp_metadata_csv)

    selected = select_codes_from_adapter_predictions(
        prediction_rows=prediction_rows,
        metadata=metadata,
        min_support=args.min_support,
        n_each=args.n_each,
    )
    train_labels = labels_by_code_and_ecg(train_rows)
    val_labels = labels_by_code_and_ecg(val_rows)

    rows: List[Dict[str, Any]] = []
    for item in selected:
        code = item["target_scp_code"]
        metrics = train_eval_code(
            code=code,
            train_labels=train_labels.get(code, {}),
            val_labels=val_labels.get(code, {}),
            embedding_bank=embedding_bank,
            max_iter=args.max_iter,
            c_value=args.c_value,
        )
        rows.append({**item, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "csfm_only_top_worst_codes.csv", rows)
    write_json(
        args.output_dir / "csfm_only_top_worst_codes.json",
        {
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "embedding_bank": str(args.embedding_bank),
            "adapter_predictions_path": str(args.adapter_predictions_path),
            "min_support": args.min_support,
            "n_each": args.n_each,
            "rows": rows,
        },
    )

    print("CSFM-only classifier comparison")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['target_scp_code']:8s} {row['selection_group']:18s} skipped")
            continue
        print(
            f"{row['target_scp_code']:8s} {row['selection_group']:18s} "
            f"adapter_bal={row['adapter_balanced_accuracy']:.3f} "
            f"csfm_bal={row['balanced_accuracy']:.3f} "
            f"auc={row['roc_auc']:.3f}"
        )
    print(f"\nSaved: {args.output_dir / 'csfm_only_top_worst_codes.csv'}")


if __name__ == "__main__":
    main()

