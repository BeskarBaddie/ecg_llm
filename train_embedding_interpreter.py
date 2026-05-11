from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("outputs/unique_ecg_dataset.jsonl")
OUTPUT_MODEL_PATH = Path("outputs/non_diagnostic_t_abnormalities_interpreter.joblib")

TARGET_ATTRIBUTE = "non-diagnostic t abnormalities"


def normalize_text(x: Any) -> str:
    if isinstance(x, list):
        if not x:
            return ""
        x = x[0]
    return str(x).strip().lower()


def load_unique_ecg_dataset(path: Path, target_attribute: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ecg_to_embedding: Dict[int, List[float]] = {}
    ecg_to_label: Dict[int, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            ecg_id = row.get("ecg_id")
            if isinstance(ecg_id, list):
                if not ecg_id:
                    continue
                ecg_id = ecg_id[0]

            if ecg_id is None:
                continue

            ecg_id = int(ecg_id)
            embedding = row.get("embedding")
            if embedding is None:
                continue

            attributes = row.get("attributes", [])
            attributes = [normalize_text(a) for a in attributes]

            if ecg_id not in ecg_to_embedding:
                ecg_to_embedding[ecg_id] = embedding
                ecg_to_label[ecg_id] = 0

            if target_attribute in attributes:
                ecg_to_label[ecg_id] = 1

    ecg_ids = sorted(ecg_to_embedding.keys())
    X = np.array([ecg_to_embedding[eid] for eid in ecg_ids], dtype=np.float32)
    y = np.array([ecg_to_label[eid] for eid in ecg_ids], dtype=np.int64)

    return X, y


def main() -> None:
    print(f"Target attribute: {TARGET_ATTRIBUTE}")

    X, y = load_unique_ecg_dataset(DATA_PATH, TARGET_ATTRIBUTE)
    print("Unique ECGs:", X.shape[0])
    print("Embedding dim:", X.shape[1])

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"Positives: {n_pos}")
    print(f"Negatives: {n_neg}")

    if n_pos == 0:
        raise RuntimeError(f"No positive examples found for attribute: {TARGET_ATTRIBUTE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(
    kernel="linear",
    class_weight="balanced",
    probability=True,
    random_state=42,
)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n=== TEST RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Average Precision:", average_precision_score(y_test, y_prob))
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # Save model bundle for later prompt generation
    bundle = {
        "target_attribute": TARGET_ATTRIBUTE,
        "scaler": scaler,
        "model": clf,
    }
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"\nSaved model to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()