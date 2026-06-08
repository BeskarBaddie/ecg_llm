from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_PATH = Path("outputs/ecgqa_csfm_preview_10000_sv.jsonl")
REPORT_PATH = Path("outputs/ecgqa_csfm_10000_sv_validation_report.json")

EXPECTED_EMBED_DIM = 768
EXPECTED_SIGNAL_SHAPE = [12, 2500]

REQUIRED_KEYS = {
    "ecg_id",
    "question",
    "answer",
    "question_type",
    "attribute_type",
    "attribute",
    "embedding",
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_answer(answer: Any) -> str:
    """
    ECG-QA answers are often stored as ["yes"] or ["no"].
    Convert to a simple string.
    """
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        return str(answer[0]).strip().lower()
    return str(answer).strip().lower()


def safe_len(x: Any) -> int:
    try:
        return len(x)
    except Exception:
        return -1


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

    return rows


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------
def validate_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    total_rows = len(rows)
    report["total_rows"] = total_rows

    missing_key_rows = 0
    bad_embedding_dim_rows = 0
    bad_signal_shape_rows = 0
    nan_rows = 0
    inf_rows = 0
    empty_question_rows = 0
    empty_answer_rows = 0
    duplicate_ecg_ids = 0

    answer_counter = Counter()
    question_type_counter = Counter()
    attribute_type_counter = Counter()
    attribute_counter = Counter()
    ecg_id_counter = Counter()

    embedding_dims = Counter()
    signal_shapes = Counter()

    embedding_means: List[float] = []
    embedding_stds: List[float] = []

    invalid_samples: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        row_issues: List[str] = []

        # Schema / required keys
        missing_keys = REQUIRED_KEYS - set(row.keys())
        if missing_keys:
            missing_key_rows += 1
            row_issues.append(f"missing_keys={sorted(missing_keys)}")

        # Basic fields
        question = str(row.get("question", "")).strip()
        if not question:
            empty_question_rows += 1
            row_issues.append("empty_question")

        answer = normalize_answer(row.get("answer", ""))
        if not answer:
            empty_answer_rows += 1
            row_issues.append("empty_answer")
        else:
            answer_counter[answer] += 1

        question_type = str(row.get("question_type", "")).strip()
        if question_type:
            question_type_counter[question_type] += 1

        attribute_type = str(row.get("attribute_type", "")).strip()
        if attribute_type:
            attribute_type_counter[attribute_type] += 1

        attr = row.get("attribute", None)
        if isinstance(attr, list) and attr:
            attribute_counter[str(attr[0]).strip().lower()] += 1
        elif isinstance(attr, str) and attr.strip():
            attribute_counter[attr.strip().lower()] += 1

        # ECG ID
        ecg_id = row.get("ecg_id", None)
        if isinstance(ecg_id, list) and ecg_id:
            ecg_id = ecg_id[0]
        if ecg_id is not None:
            ecg_id_counter[str(ecg_id)] += 1

        # Embedding validation
        emb = row.get("embedding", None)
        emb_arr = np.array(emb, dtype=np.float32) if emb is not None else None

        if emb_arr is None:
            row_issues.append("missing_embedding")
        else:
            emb_dim = int(emb_arr.shape[0]) if emb_arr.ndim == 1 else -1
            embedding_dims[emb_dim] += 1
            if emb_dim != EXPECTED_EMBED_DIM:
                bad_embedding_dim_rows += 1
                row_issues.append(f"bad_embedding_dim={emb_dim}")

            if np.isnan(emb_arr).any():
                nan_rows += 1
                row_issues.append("nan_in_embedding")

            if np.isinf(emb_arr).any():
                inf_rows += 1
                row_issues.append("inf_in_embedding")

            if emb_arr.ndim == 1 and emb_dim > 0:
                embedding_means.append(float(np.mean(emb_arr)))
                embedding_stds.append(float(np.std(emb_arr)))

        # Signal shape validation
        signal_shape = row.get("signal_shape", None)
        if signal_shape is not None:
            signal_shapes[str(signal_shape)] += 1
            if list(signal_shape) != EXPECTED_SIGNAL_SHAPE:
                bad_signal_shape_rows += 1
                row_issues.append(f"bad_signal_shape={signal_shape}")

        if row_issues:
            invalid_samples.append(
                {
                    "row_index": idx,
                    "ecg_id": ecg_id,
                    "issues": row_issues,
                }
            )

    # Duplicate ECG IDs
    duplicate_ecg_ids = sum(1 for _, c in ecg_id_counter.items() if c > 1)

    # Coverage metrics
    unique_ecg_ids = len(ecg_id_counter)
    answer_total = sum(answer_counter.values())
    yes_count = answer_counter.get("yes", 0)
    no_count = answer_counter.get("no", 0)
    yes_ratio = (yes_count / answer_total) if answer_total else 0.0
    no_ratio = (no_count / answer_total) if answer_total else 0.0

    # Simple imbalance warning
    imbalance_warning = None
    if answer_total > 0:
        majority_share = max(yes_ratio, no_ratio)
        if majority_share > 0.75:
            imbalance_warning = "Strong answer imbalance detected (majority class > 75%)."

    report.update(
        {
            "total_rows": total_rows,
            "unique_ecg_ids": unique_ecg_ids,
            "duplicate_ecg_id_count": duplicate_ecg_ids,
            "missing_key_rows": missing_key_rows,
            "bad_embedding_dim_rows": bad_embedding_dim_rows,
            "bad_signal_shape_rows": bad_signal_shape_rows,
            "nan_rows": nan_rows,
            "inf_rows": inf_rows,
            "empty_question_rows": empty_question_rows,
            "empty_answer_rows": empty_answer_rows,
            "answer_distribution": dict(answer_counter),
            "question_type_distribution": dict(question_type_counter),
            "attribute_type_distribution": dict(attribute_type_counter),
            "top_attributes": attribute_counter.most_common(20),
            "embedding_dim_distribution": dict(embedding_dims),
            "signal_shape_distribution": dict(signal_shapes),
            "embedding_mean_overall": float(np.mean(embedding_means)) if embedding_means else None,
            "embedding_std_overall": float(np.mean(embedding_stds)) if embedding_stds else None,
            "yes_ratio": yes_ratio,
            "no_ratio": no_ratio,
            "imbalance_warning": imbalance_warning,
            "invalid_samples_preview": invalid_samples[:25],
        }
    )

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\n=== DATASET VALIDATION REPORT ===")
    print(f"Total rows: {report['total_rows']}")
    print(f"Unique ECG IDs: {report['unique_ecg_ids']}")
    print(f"Duplicate ECG ID count: {report['duplicate_ecg_id_count']}")
    print(f"Rows with missing required keys: {report['missing_key_rows']}")
    print(f"Rows with bad embedding dim: {report['bad_embedding_dim_rows']}")
    print(f"Rows with bad signal shape: {report['bad_signal_shape_rows']}")
    print(f"Rows with NaNs in embedding: {report['nan_rows']}")
    print(f"Rows with Infs in embedding: {report['inf_rows']}")
    print(f"Rows with empty question: {report['empty_question_rows']}")
    print(f"Rows with empty answer: {report['empty_answer_rows']}")

    print("\n--- Answer distribution ---")
    for k, v in sorted(report["answer_distribution"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")

    print("\n--- Question types ---")
    for k, v in sorted(report["question_type_distribution"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")

    print("\n--- Attribute types ---")
    for k, v in sorted(report["attribute_type_distribution"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")

    print("\n--- Embedding dimensions ---")
    print(report["embedding_dim_distribution"])

    print("\n--- Signal shapes ---")
    print(report["signal_shape_distribution"])

    if report["imbalance_warning"]:
        print("\nWARNING:", report["imbalance_warning"])

    print("\n--- Top attributes ---")
    for attr, count in report["top_attributes"][:20]:
        print(f"{attr}: {count}")

    print("\n--- Preview of problematic rows ---")
    for item in report["invalid_samples_preview"][:10]:
        print(item)


def main() -> None:
    rows = load_jsonl(DATA_PATH)
    report = validate_dataset(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()