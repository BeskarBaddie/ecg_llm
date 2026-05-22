from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    XGBClassifier = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None


# -----------------------------
# CONFIG
# -----------------------------
ATTRIBUTE_DATA_PATH = Path("outputs/unique_ecg_dataset.jsonl")
SCP_DATA_PATH = Path("outputs/unique_ecg_scp_dataset.jsonl")
DEFAULT_LABEL_SOURCE = "scp"
DEFAULT_TARGET = "AFIB"


def make_output_model_path(target: str, classifier_name: str, label_source: str) -> Path:
    slug = (
        target.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    return Path(f"outputs/{slug}_{label_source}_{classifier_name}_interpreter.joblib")


# -----------------------------
# TEXT NORMALISATION
# -----------------------------
def normalize_text(x: Any) -> str:
    if isinstance(x, list):
        if not x:
            return ""
        x = x[0]
    return str(x).strip().lower()


# -----------------------------
# DATA LOADING
# -----------------------------
def load_unique_ecg_dataset(path: Path, target_attribute: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one sample per unique ECG.
    Label = 1 if the ECG has the target semantic attribute, else 0.
    """
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


def load_scp_ecg_dataset(
    path: Path,
    target_scp_code: str,
    min_scp_likelihood: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one sample per unique ECG using PTB-XL SCP codes as labels.

    By default, label = 1 when the target SCP code is present at all. This is
    important for rhythm codes such as AFIB, which PTB-XL often stores with a
    likelihood value of 0.0 despite the code being present.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"SCP dataset file not found: {path}. "
            "Run build_scp_ecg_dataset.py first."
        )

    target_scp_code = target_scp_code.strip().upper()

    embeddings = []
    labels = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            embedding = row.get("embedding")
            if embedding is None:
                continue

            scp_codes = row.get("scp_codes", {})
            if not isinstance(scp_codes, dict):
                continue

            has_code = target_scp_code in scp_codes

            if min_scp_likelihood is None:
                label = int(has_code)
            else:
                code_value = float(scp_codes.get(target_scp_code, 0.0))
                label = int(has_code and code_value >= min_scp_likelihood)

            embeddings.append(embedding)
            labels.append(label)

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y


# -----------------------------
# CLASSIFIER FACTORY
# -----------------------------
def build_classifier(name: str, y_train: np.ndarray | None = None):
    name = name.lower().strip()

    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

    if name == "svm":
        return SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from _XGB_IMPORT_ERROR

        if y_train is None:
            raise ValueError("y_train is required for xgb so we can compute scale_pos_weight")

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = n_neg / max(n_pos, 1)

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

    raise ValueError("Unknown classifier. Use one of: logreg, svm, xgb.")


def print_metrics(title: str, y_true, y_pred, y_prob=None) -> None:
    print(f"\n=== {title} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))

    if y_prob is not None and len(np.unique(y_true)) > 1:
        print("ROC AUC:", roc_auc_score(y_true, y_prob))
        print("Average Precision:", average_precision_score(y_true, y_prob))

    print(classification_report(y_true, y_pred, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        default="svm",
        choices=["logreg", "svm", "xgb"],
        help="Classifier to use on top of the CSFM embeddings.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default=DEFAULT_LABEL_SOURCE,
        choices=["scp", "attribute"],
        help="Use PTB-XL SCP codes or the older ECG-QA attribute labels.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target SCP code, e.g. AFIB or NDT, or ECG-QA attribute text.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional path to the prepared JSONL dataset.",
    )
    parser.add_argument(
        "--min-scp-likelihood",
        type=float,
        default=None,
        help=(
            "Optional minimum SCP likelihood for positive labels. "
            "Leave unset to label by code presence."
        ),
    )
    args = parser.parse_args()

    data_path = args.data_path
    if data_path is None:
        data_path = SCP_DATA_PATH if args.label_source == "scp" else ATTRIBUTE_DATA_PATH

    output_model_path = make_output_model_path(
        args.target,
        args.classifier,
        args.label_source,
    )

    print(f"Label source: {args.label_source}")
    print(f"Target: {args.target}")
    print(f"Data path: {data_path}")
    print(f"Using classifier: {args.classifier}")

    if args.label_source == "scp":
        X, y = load_scp_ecg_dataset(
            data_path,
            target_scp_code=args.target,
            min_scp_likelihood=args.min_scp_likelihood,
        )
    else:
        X, y = load_unique_ecg_dataset(data_path, args.target)

    print("Unique ECGs:", X.shape[0])
    print("Embedding dim:", X.shape[1])

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"Positives: {n_pos}")
    print(f"Negatives: {n_neg}")

    if n_pos == 0:
        raise RuntimeError(f"No positive examples found for target: {args.target}")

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

    clf = build_classifier(args.classifier, y_train=y_train)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        except Exception:
            y_prob = None

    print_metrics("TEST RESULTS", y_test, y_pred, y_prob=y_prob)

    bundle = {
        "label_source": args.label_source,
        "target": args.target,
        "min_scp_likelihood": args.min_scp_likelihood,
        "classifier_name": args.classifier,
        "scaler": scaler,
        "model": clf,
    }
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_model_path)
    print(f"\nSaved model to: {output_model_path}")


if __name__ == "__main__":
    main()
