from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("outputs/unique_ecg_scp_dataset.jsonl")
OUTPUT_JSON_PATH = Path("outputs/scp_code_panel_results.json")
OUTPUT_CSV_PATH = Path("outputs/scp_code_panel_results.csv")


def load_dataset(path: Path) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"SCP dataset not found: {path}. Run build_scp_ecg_dataset.py first."
        )

    rows: List[Dict[str, Any]] = []
    embeddings = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            embedding = row.get("embedding")
            scp_codes = row.get("scp_codes")

            if embedding is None or not isinstance(scp_codes, dict):
                continue

            rows.append(row)
            embeddings.append(embedding)

    X = np.array(embeddings, dtype=np.float32)
    return X, rows


def get_code_descriptions(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}

    for row in rows:
        for statement in row.get("scp_statements", []):
            code = statement.get("code")
            description = statement.get("description")
            if code and description and code not in descriptions:
                descriptions[str(code)] = str(description)

    return descriptions


def get_candidate_codes(rows: List[Dict[str, Any]], min_positive: int) -> List[str]:
    counts = Counter()

    for row in rows:
        counts.update(row.get("scp_codes", {}).keys())

    return sorted(code for code, count in counts.items() if count >= min_positive)


def make_labels(rows: List[Dict[str, Any]], code: str) -> np.ndarray:
    return np.array(
        [int(code in row.get("scp_codes", {})) for row in rows],
        dtype=np.int64,
    )


def evaluate_code(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
) -> Dict[str, float]:
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)

    return {
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "average_precision": float(average_precision_score(y_val, y_prob)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "positive_recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "positive_f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }


def summarize_repeats(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    keys = metrics[0].keys()

    for key in keys:
        values = np.array([m[key] for m in metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--min-positive", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="Optional explicit SCP code list. Defaults to all codes above --min-positive.",
    )
    args = parser.parse_args()

    X, rows = load_dataset(args.data_path)
    descriptions = get_code_descriptions(rows)

    if args.codes:
        candidate_codes = [code.strip().upper() for code in args.codes]
    else:
        candidate_codes = get_candidate_codes(rows, min_positive=args.min_positive)

    results = []

    for code in candidate_codes:
        y = make_labels(rows, code)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < args.min_positive or n_neg < args.min_positive:
            continue

        repeated_metrics = []
        for i in range(args.repeats):
            repeated_metrics.append(
                evaluate_code(
                    X,
                    y,
                    random_state=args.seed + i,
                    test_size=args.test_size,
                )
            )

        result = {
            "code": code,
            "description": descriptions.get(code, ""),
            "positives": n_pos,
            "negatives": n_neg,
            "prevalence": float(n_pos / len(y)),
        }
        result.update(summarize_repeats(repeated_metrics))
        results.append(result)

    results.sort(key=lambda item: item["roc_auc_mean"], reverse=True)

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"Evaluated {len(results)} SCP codes")
    print(f"Saved JSON: {OUTPUT_JSON_PATH}")
    print(f"Saved CSV: {OUTPUT_CSV_PATH}")
    print("\nTop codes by mean ROC AUC:")
    for result in results[:20]:
        print(
            f"{result['code']:8s} "
            f"AUC={result['roc_auc_mean']:.3f} "
            f"AP={result['average_precision_mean']:.3f} "
            f"pos={result['positives']:4d} "
            f"{result['description']}"
        )


if __name__ == "__main__":
    main()
