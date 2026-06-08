from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ecg_feature_extractor import (
    LEADS_12,
    extract_signal_mc_med_features,
    extract_single_lead_domain_features,
)
from src.ptbxl_loader import load_ecg_signal, load_ptbxl_metadata


TRAIN_PATH = Path("outputs/ecgqa_scp_binary_train.jsonl")
VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/domain_feature_classifier_results.json")
PREDICTIONS_PATH = Path("outputs/domain_feature_classifier_predictions.jsonl")
MODEL_BUNDLE_PATH = Path("outputs/domain_feature_classifier_models.joblib")
FEATURE_CACHE_PATH = Path("outputs/domain_feature_cache.jsonl")


# Function: Load a JSONL dataset into memory.
# Inputs: Path to a newline-delimited JSON file.
# Outputs: List of parsed row dictionaries.
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
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return rows


# Function: Write JSON-serializable rows to a JSONL file.
# Inputs: Output path and row dictionaries.
# Outputs: None; writes the file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build a stable feature cache key.
# Inputs: ECG ID, lead name, feature set, and feature count.
# Outputs: String key suitable for JSON and Python dictionaries.
def make_feature_cache_key(
    ecg_id: int,
    lead_name: str,
    feature_set: str,
    max_features: int | None,
) -> str:
    max_feature_text = "all" if max_features is None else str(max_features)
    return f"{ecg_id}|{lead_name}|{feature_set}|{max_feature_text}"


# Function: Load cached domain features from disk.
# Inputs: Cache path.
# Outputs: Dictionary keyed by feature cache key.
def load_feature_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    cache: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            cache[str(row["cache_key"])] = row
    return cache


# Function: Save domain feature cache to disk.
# Inputs: Cache path and feature cache dictionary.
# Outputs: None; writes cache rows to disk.
def save_feature_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps(cache[key]) + "\n")


# Function: Extract or retrieve single-lead domain features for one ECG.
# Inputs: ECG ID, metadata, lead, feature set, feature count, and mutable cache.
# Outputs: Feature payload containing names, values, and provenance.
def get_domain_feature_item(
    ecg_id: int,
    metadata: Any,
    lead_name: str,
    feature_set: str,
    max_features: int | None,
    feature_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = make_feature_cache_key(ecg_id, lead_name, feature_set, max_features)
    if cache_key in feature_cache:
        return feature_cache[cache_key]

    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="hr")
        fs = 500
        source = "hr"
    except Exception:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
        fs = 100
        source = "lr"

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] != len(LEADS_12):
        raise ValueError(f"Expected 12-lead ECG, got shape {signal.shape}")

    lead_idx = LEADS_12.index(lead_name)
    if feature_set == "signalmc":
        features = extract_signal_mc_med_features(
            signal[lead_idx],
            fs=fs,
            max_features=max_features,
        )
    elif feature_set == "compact":
        features = extract_single_lead_domain_features(
            signal[lead_idx],
            fs=fs,
            max_features=max_features,
        )
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    item = {
        "cache_key": cache_key,
        "ecg_id": int(ecg_id),
        "lead_name": lead_name,
        "feature_set": feature_set,
        "max_features": max_features,
        "fs": fs,
        "source": source,
        "signal_shape": list(signal.shape),
        "features": {key: float(np.nan_to_num(value)) for key, value in features.items()},
    }
    feature_cache[cache_key] = item
    return item


# Function: Convert ECG-QA rows into a domain feature matrix.
# Inputs: Dataset rows, metadata, feature extraction settings, and cache.
# Outputs: Feature matrix, label vector, feature names, kept rows, and skipped rows.
def rows_to_domain_feature_arrays(
    rows: List[Dict[str, Any]],
    metadata: Any,
    lead_name: str,
    feature_set: str,
    max_features: int | None,
    feature_cache: Dict[str, Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    feature_names: List[str] | None = None
    X: List[List[float]] = []
    y: List[int] = []
    kept_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for row in rows:
        ecg_id = int(row["ecg_id"])
        try:
            item = get_domain_feature_item(
                ecg_id=ecg_id,
                metadata=metadata,
                lead_name=lead_name,
                feature_set=feature_set,
                max_features=max_features,
                feature_cache=feature_cache,
            )
            features = item["features"]
            if feature_names is None:
                feature_names = list(features.keys())
            values = [float(features.get(name, 0.0)) for name in feature_names]
        except Exception as exc:
            skipped = dict(row)
            skipped["feature_error"] = str(exc)
            skipped_rows.append(skipped)
            continue

        X.append(values)
        y.append(int(row["label"]))
        kept_rows.append(row)

    if feature_names is None:
        raise RuntimeError("No rows were converted to domain features.")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        feature_names,
        kept_rows,
        skipped_rows,
    )


# Function: Build a classifier for domain feature validation.
# Inputs: Classifier name, random seed, and MLP hidden layer size.
# Outputs: sklearn pipeline.
def build_classifier(classifier_name: str, random_state: int, mlp_hidden: int) -> Any:
    classifier_name = classifier_name.lower().strip()

    if classifier_name == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif classifier_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(mlp_hidden,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        )
    else:
        raise ValueError("Unknown classifier. Use: logreg or mlp.")

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )


# Function: Return positive-class probabilities when a fitted model supports them.
# Inputs: Fitted model and feature matrix.
# Outputs: Positive-class probability vector.
def positive_prob(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Classifier does not expose predict_proba.")
    return model.predict_proba(X)[:, 1]


# Function: Compute ROC AUC only when both classes are present.
# Inputs: True labels and positive-class probabilities.
# Outputs: ROC AUC float, or None when undefined.
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


# Function: Compute average precision only when both classes are present.
# Inputs: True labels and positive-class probabilities.
# Outputs: Average precision float, or None when undefined.
def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


# Function: Compute binary classification metrics.
# Inputs: True labels, predicted labels, and positive-class probabilities.
# Outputs: JSON-serializable metric dictionary.
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "label_counts": dict(Counter(int(x) for x in y_true)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "labels": {
            "no": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "yes": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        ),
    }


# Function: Group row indices by target SCP code.
# Inputs: Dataset rows.
# Outputs: Mapping from SCP code to row indices.
def indices_by_code(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row["target_scp_code"])].append(idx)
    return dict(grouped)


# Function: Train one domain-feature classifier per SCP code and evaluate on validation rows.
# Inputs: Feature matrices, labels, rows, classifier settings, and random seed.
# Outputs: Models, metrics by code, and row-level predictions.
def train_and_evaluate_per_code(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_rows: List[Dict[str, Any]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_rows: List[Dict[str, Any]],
    classifier_name: str,
    random_state: int,
    mlp_hidden: int,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    train_indices_by_code = indices_by_code(train_rows)
    val_indices_by_code = indices_by_code(val_rows)

    models: Dict[str, Any] = {}
    metrics_by_code: Dict[str, Dict[str, Any]] = {}
    prediction_rows: List[Dict[str, Any]] = []

    for code in sorted(train_indices_by_code):
        train_indices = train_indices_by_code[code]
        val_indices = val_indices_by_code.get(code, [])
        if not val_indices:
            continue

        code_y_train = y_train[train_indices]
        code_y_val = y_val[val_indices]

        if len(np.unique(code_y_train)) < 2:
            metrics_by_code[code] = {
                "skipped": True,
                "reason": "only_one_training_class",
                "train_label_counts": dict(Counter(int(x) for x in code_y_train)),
                "val_label_counts": dict(Counter(int(x) for x in code_y_val)),
            }
            continue

        model = build_classifier(
            classifier_name=classifier_name,
            random_state=random_state,
            mlp_hidden=mlp_hidden,
        )
        model.fit(X_train[train_indices], code_y_train)
        y_prob = positive_prob(model, X_val[val_indices])
        y_pred = (y_prob >= 0.5).astype(np.int64)

        models[code] = model
        metrics = compute_metrics(code_y_val, y_pred, y_prob)
        metrics["skipped"] = False
        metrics["train_label_counts"] = dict(Counter(int(x) for x in code_y_train))
        metrics["val_label_counts"] = dict(Counter(int(x) for x in code_y_val))
        metrics_by_code[code] = metrics

        for local_idx, row_idx in enumerate(val_indices):
            row = val_rows[row_idx]
            prediction_rows.append(
                {
                    "method": f"domain_features_{classifier_name}_per_code",
                    "ecg_id": row.get("ecg_id"),
                    "split": row.get("split"),
                    "target_scp_code": code,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "true_label": int(code_y_val[local_idx]),
                    "pred_label": int(y_pred[local_idx]),
                    "pred_answer": "yes" if int(y_pred[local_idx]) == 1 else "no",
                    "pred_prob_yes": float(y_prob[local_idx]),
                    "correct": bool(int(y_pred[local_idx]) == int(code_y_val[local_idx])),
                }
            )

        print(
            f"{code:6s} "
            f"n_train={len(train_indices):4d} n_val={len(val_indices):3d} "
            f"bal_acc={metrics['balanced_accuracy'] if metrics['balanced_accuracy'] is not None else 'NA'} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}",
            flush=True,
        )

    return models, metrics_by_code, prediction_rows


# Function: Aggregate per-code metrics with simple macro averaging.
# Inputs: Per-code metric dictionary.
# Outputs: Aggregate summary dictionary.
def summarize_per_code_metrics(metrics_by_code: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    valid_metrics = [metrics for metrics in metrics_by_code.values() if not metrics.get("skipped")]
    summary: Dict[str, Any] = {"n_codes": len(valid_metrics)}

    for metric_name in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "average_precision"]:
        values = [
            float(metrics[metric_name])
            for metrics in valid_metrics
            if metrics.get(metric_name) is not None
        ]
        summary[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
        summary[f"median_{metric_name}"] = float(np.median(values)) if values else None

    return summary


# Function: Run domain-feature classifier validation from command-line arguments.
# Inputs: CLI arguments controlling data paths, feature extraction, and classifier.
# Outputs: Result, prediction, cache, and model files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--feature-set", type=str, default="signalmc", choices=["signalmc", "compact"])
    parser.add_argument("--lead", type=str, default="II", choices=LEADS_12)
    parser.add_argument("--max-features", type=int, default=54)
    parser.add_argument("--classifier", type=str, default="logreg", choices=["logreg", "mlp"])
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--feature-cache-path", type=Path, default=FEATURE_CACHE_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-bundle-path", type=Path, default=MODEL_BUNDLE_PATH)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)

    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_rows = val_rows[: args.max_val_rows]

    metadata = load_ptbxl_metadata()
    feature_cache = load_feature_cache(args.feature_cache_path)

    max_features = None if args.max_features <= 0 else args.max_features

    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("Feature set:", args.feature_set)
    print("Lead:", args.lead)
    print("Max features:", max_features if max_features is not None else "all")
    print("Classifier:", args.classifier)
    print("Loaded feature cache rows:", len(feature_cache))

    X_train, y_train, feature_names, kept_train_rows, skipped_train_rows = rows_to_domain_feature_arrays(
        train_rows,
        metadata=metadata,
        lead_name=args.lead,
        feature_set=args.feature_set,
        max_features=max_features,
        feature_cache=feature_cache,
    )
    X_val, y_val, val_feature_names, kept_val_rows, skipped_val_rows = rows_to_domain_feature_arrays(
        val_rows,
        metadata=metadata,
        lead_name=args.lead,
        feature_set=args.feature_set,
        max_features=max_features,
        feature_cache=feature_cache,
    )

    if feature_names != val_feature_names:
        raise RuntimeError("Train and validation feature names differ.")

    save_feature_cache(args.feature_cache_path, feature_cache)

    print("Kept train rows:", len(kept_train_rows))
    print("Kept val rows:", len(kept_val_rows))
    print("Skipped train rows:", len(skipped_train_rows))
    print("Skipped val rows:", len(skipped_val_rows))
    print("Feature dim:", X_train.shape[1])
    print("Train label counts:", dict(Counter(int(x) for x in y_train)))
    print("Val label counts:", dict(Counter(int(x) for x in y_val)))
    print("Saved feature cache rows:", len(feature_cache))

    models, metrics_by_code, prediction_rows = train_and_evaluate_per_code(
        X_train=X_train,
        y_train=y_train,
        train_rows=kept_train_rows,
        X_val=X_val,
        y_val=y_val,
        val_rows=kept_val_rows,
        classifier_name=args.classifier,
        random_state=args.seed,
        mlp_hidden=args.mlp_hidden,
    )

    results = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "feature_set": args.feature_set,
        "lead": args.lead,
        "max_features": max_features,
        "classifier": args.classifier,
        "mlp_hidden": args.mlp_hidden if args.classifier == "mlp" else None,
        "seed": args.seed,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "kept_train_rows": len(kept_train_rows),
        "kept_val_rows": len(kept_val_rows),
        "skipped_train_rows": len(skipped_train_rows),
        "skipped_val_rows": len(skipped_val_rows),
        "feature_dim": int(X_train.shape[1]),
        "feature_names": feature_names,
        "feature_cache_path": str(args.feature_cache_path),
        "feature_cache_size": len(feature_cache),
        "train_label_counts": dict(Counter(int(x) for x in y_train)),
        "val_label_counts": dict(Counter(int(x) for x in y_val)),
        "per_code": metrics_by_code,
        "summary": summarize_per_code_metrics(metrics_by_code),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_jsonl(args.predictions_path, prediction_rows)

    args.model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_names": feature_names,
            "results": results,
        },
        args.model_bundle_path,
    )

    print("\n=== Domain Feature Classifier Summary ===")
    for key, value in results["summary"].items():
        print(f"{key}: {value}")
    print(f"\nSaved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")
    print(f"Saved model bundle: {args.model_bundle_path}")


if __name__ == "__main__":
    main()
