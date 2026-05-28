from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.ecg_feature_extractor import extract_12lead_basic_features
from src.ecg_prompt_builder import build_ecg_prompt
from src.ptbxl_loader import load_ecg_signal, load_ptbxl_metadata


VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/feature_llm_baseline_results.json")
PREDICTIONS_PATH = Path("outputs/feature_llm_baseline_predictions.jsonl")
EMBEDDING_BASELINE_RESULTS_PATH = Path("outputs/ecgqa_embedding_baseline_results.json")


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


def build_binary_ecg_prompt(feature_dict: Dict[str, Dict[str, float]], question: str) -> str:
    prompt = build_ecg_prompt(
        feature_dict=feature_dict,
        question=question,
        allowed_answers=("yes", "no"),
    )
    return (
        f"{prompt}\n"
        "Return exactly one word: yes or no.\n"
        "Do not include any explanation."
    )


def parse_yes_no_response(text: str) -> tuple[int, str]:
    cleaned = text.strip().lower()

    if cleaned.startswith("yes"):
        return 1, "yes"

    if cleaned.startswith("no"):
        return 0, "no"

    tokens = cleaned.replace(".", " ").replace(",", " ").split()
    if tokens:
        if tokens[0] == "yes":
            return 1, "yes"
        if tokens[0] == "no":
            return 0, "no"

    return -1, "unknown"


def load_or_extract_features(
    ecg_id: int,
    metadata: Any,
    feature_cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    if ecg_id in feature_cache:
        return feature_cache[ecg_id]

    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="hr")
        fs = 500
        source = "hr"
    except Exception:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
        fs = 100
        source = "lr"

    signal = np.asarray(signal, dtype=np.float32)
    feature_dict = extract_12lead_basic_features(signal, fs=fs)

    feature_cache[ecg_id] = {
        "features": feature_dict,
        "fs": fs,
        "source": source,
        "signal_shape": list(signal.shape),
    }
    return feature_cache[ecg_id]


def call_ollama(model: str, prompt: str, temperature: float, num_predict: int) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "The Python package 'ollama' is not installed in this environment. "
            "Install it with: python -m pip install ollama"
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return response["message"]["content"]


def per_code_metrics(
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    by_code: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_code[str(row.get("target_scp_code", "UNKNOWN"))].append(idx)

    metrics: Dict[str, Dict[str, Any]] = {}
    for code, indices in sorted(by_code.items()):
        code_true = y_true[indices]
        code_pred = y_pred[indices]
        metrics[code] = {
            "n": int(len(indices)),
            "accuracy": float(accuracy_score(code_true, code_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(code_true, code_pred))
            if len(np.unique(code_true)) > 1
            else None,
            "macro_f1": float(f1_score(code_true, code_pred, average="macro", zero_division=0)),
            "label_counts": dict(Counter(int(x) for x in code_true)),
            "prediction_counts": dict(Counter(int(x) for x in code_pred)),
        }

    return metrics


def evaluate_predictions(
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "invalid_predictions": int(np.sum(y_pred == -1)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "confusion_matrix_labels": ["no", "yes", "unknown"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, -1]).tolist(),
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
        "classification_report": report,
        "per_code": per_code_metrics(rows, y_true, y_pred),
    }


def load_embedding_baseline_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        method: {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "roc_auc": metrics.get("roc_auc"),
        }
        for method, metrics in data.get("methods", {}).items()
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument(
        "--embedding-baseline-results-path",
        type=Path,
        default=EMBEDDING_BASELINE_RESULTS_PATH,
    )
    args = parser.parse_args()

    rows = load_jsonl(args.val_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        raise RuntimeError("No validation rows to evaluate.")

    metadata = load_ptbxl_metadata()
    feature_cache: Dict[int, Dict[str, Any]] = {}

    predictions: List[Dict[str, Any]] = []
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = args.predictions_path.open("w", encoding="utf-8")

    try:
        for idx, row in enumerate(rows, start=1):
            ecg_id = int(row["ecg_id"])
            true_label = int(row["label"])

            try:
                feature_item = load_or_extract_features(ecg_id, metadata, feature_cache)
                prompt = build_binary_ecg_prompt(feature_item["features"], row["question"])
                raw_response = call_ollama(
                    args.model,
                    prompt,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                )
                pred_label, pred_answer = parse_yes_no_response(raw_response)
                error = None
            except Exception as exc:
                feature_item = {}
                prompt = None
                raw_response = ""
                pred_label = -1
                pred_answer = "unknown"
                error = str(exc)

            prediction = {
                "method": "feature_llm",
                "model": args.model,
                "ecg_id": ecg_id,
                "split": row.get("split"),
                "target_scp_code": row.get("target_scp_code"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "true_label": true_label,
                "pred_label": int(pred_label),
                "pred_answer": pred_answer,
                "raw_response": raw_response,
                "correct": bool(pred_label == true_label),
                "feature_source": feature_item.get("source"),
                "fs": feature_item.get("fs"),
                "signal_shape": feature_item.get("signal_shape"),
                "prompt": prompt,
                "error": error,
            }
            predictions.append(prediction)
            prediction_file.write(json.dumps(prediction) + "\n")
            prediction_file.flush()

            print(
                f"[{idx}/{len(rows)}] "
                f"ecg_id={ecg_id} code={row.get('target_scp_code')} "
                f"true={row.get('answer')} pred={pred_answer}",
                flush=True,
            )
    finally:
        prediction_file.close()

    y_true = np.array([int(row["true_label"]) for row in predictions], dtype=np.int64)
    y_pred = np.array([int(row["pred_label"]) for row in predictions], dtype=np.int64)

    results = {
        "method": "feature_llm",
        "model": args.model,
        "temperature": args.temperature,
        "val_path": str(args.val_path),
        "n_rows": len(rows),
        "n_unique_ecgs": len({row["ecg_id"] for row in rows}),
        "feature_cache_size": len(feature_cache),
        "metrics": evaluate_predictions(rows, y_true, y_pred),
        "embedding_baseline_summary": load_embedding_baseline_summary(
            args.embedding_baseline_results_path
        ),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    metrics = results["metrics"]
    print("\n=== Feature LLM Baseline ===")
    print("Accuracy:", metrics["accuracy"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Invalid predictions:", metrics["invalid_predictions"])
    print(f"\nSaved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
