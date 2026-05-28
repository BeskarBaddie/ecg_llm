from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = Path("outputs/ecgqa_scp_binary_train.jsonl")
VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/classifier_llm_baseline_results.json")
PREDICTIONS_PATH = Path("outputs/classifier_llm_baseline_predictions.jsonl")
MODEL_BUNDLE_PATH = Path("outputs/classifier_llm_attribute_models.joblib")
EMBEDDING_BASELINE_RESULTS_PATH = Path("outputs/ecgqa_embedding_baseline_results.json")
FEATURE_LLM_RESULTS_PATH = Path("outputs/feature_llm_baseline_results.json")


# Function: Load a JSONL dataset into memory.
# Inputs: Path to a JSONL file.
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


# Function: Write prediction rows to a JSONL file.
# Inputs: Output path and a list of serializable prediction dictionaries.
# Outputs: None; writes the file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build human-readable names for each SCP code.
# Inputs: Dataset rows containing target SCP code metadata.
# Outputs: Dictionary mapping SCP code to display description.
def get_code_descriptions(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}

    for row in rows:
        code = str(row.get("target_scp_code"))
        statement = row.get("target_scp_statement", {})
        description = statement.get("description") or row.get("attribute") or code
        descriptions[code] = str(description)

    return descriptions


# Function: Train one binary ECG classifier per SCP code.
# Inputs: Training rows with CSFM embeddings and binary labels, plus random seed.
# Outputs: Dictionary mapping SCP code to a fitted sklearn classifier.
def train_attribute_classifiers(
    train_rows: List[Dict[str, Any]],
    random_state: int,
) -> Dict[str, Any]:
    rows_by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        rows_by_code[str(row["target_scp_code"])].append(row)

    models: Dict[str, Any] = {}

    for code, code_rows in sorted(rows_by_code.items()):
        X = np.array([row["embedding"] for row in code_rows], dtype=np.float32)
        y = np.array([int(row["label"]) for row in code_rows], dtype=np.int64)

        if len(np.unique(y)) < 2:
            raise RuntimeError(
                f"Cannot train classifier for {code}: only one class present in training rows."
            )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
        model.fit(X, y)
        models[code] = model

    return models


# Function: Predict all trained SCP-code probabilities for one ECG embedding.
# Inputs: Fitted attribute models and one CSFM ECG embedding.
# Outputs: Dictionary mapping SCP code to positive-class probability.
def predict_attribute_probs(models: Dict[str, Any], embedding: List[float]) -> Dict[str, float]:
    X = np.array([embedding], dtype=np.float32)
    probs: Dict[str, float] = {}

    for code, model in sorted(models.items()):
        probs[code] = float(model.predict_proba(X)[0, 1])

    return probs


# Function: Convert classifier probabilities into a text ECG interpretation.
# Inputs: Per-code probabilities, code descriptions, and positive threshold.
# Outputs: Text block listing positive findings and all classifier scores.
def build_classifier_text(
    probs: Dict[str, float],
    descriptions: Dict[str, str],
    threshold: float,
) -> str:
    positive_lines = []
    score_lines = []

    for code, prob in sorted(probs.items()):
        description = descriptions.get(code, code)
        score_lines.append(f"- {code} ({description}): {prob:.3f}")
        if prob >= threshold:
            positive_lines.append(f"- {code} ({description}): {prob:.3f}")

    lines = []
    lines.append("An ECG attribute classifier produced the following predictions.")
    lines.append(f"Positive findings at threshold {threshold:.2f}:")
    if positive_lines:
        lines.extend(positive_lines)
    else:
        lines.append("- none")

    lines.append("")
    lines.append("All classifier scores:")
    lines.extend(score_lines)

    return "\n".join(lines)


# Function: Build the LLM prompt for a binary ECG-QA example.
# Inputs: Text ECG interpretation and question string.
# Outputs: Prompt string constrained to yes/no answers.
def build_llm_prompt(
    classifier_text: str,
    question: str,
) -> str:
    return "\n".join(
        [
            "You are answering a binary ECG question.",
            "Use the ECG classifier findings below as the ECG interpretation.",
            "",
            classifier_text,
            "",
            f"Question: {question}",
            "Valid answers: yes, no.",
            "Return exactly one word: yes or no.",
            "Do not include any explanation.",
        ]
    )


# Function: Parse an LLM response into a binary yes/no prediction.
# Inputs: Raw model response text.
# Outputs: Tuple of numeric label and normalized answer string.
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


# Function: Call Ollama chat completion for one prompt.
# Inputs: Ollama model name, prompt, temperature, and max generated tokens.
# Outputs: Raw assistant response text.
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


# Function: Answer a question directly from the target attribute probability.
# Inputs: ECG-QA row, all attribute probabilities, and decision threshold.
# Outputs: Binary prediction label.
def direct_threshold_prediction(row: Dict[str, Any], probs: Dict[str, float], threshold: float) -> int:
    target_code = str(row["target_scp_code"])
    return int(probs[target_code] >= threshold)


# Function: Score threshold candidates for binary validation predictions.
# Inputs: True labels, predicted labels, and metric name.
# Outputs: Scalar score for the requested metric.
def threshold_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    raise ValueError(f"Unsupported threshold metric: {metric}")


# Function: Tune one decision threshold per SCP code on validation examples.
# Inputs: Validation rows, per-row attribute probabilities, true labels, metric, grid size, and fallback threshold.
# Outputs: Dictionary of tuned thresholds and per-code tuning diagnostics.
def tune_thresholds_by_code(
    rows: List[Dict[str, Any]],
    probability_rows: List[Dict[str, float]],
    y_true: np.ndarray,
    metric: str,
    grid_size: int,
    default_threshold: float,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    by_code: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_code[str(row["target_scp_code"])].append(idx)

    candidate_thresholds = np.linspace(0.0, 1.0, grid_size)
    thresholds: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}

    for code, indices in sorted(by_code.items()):
        code_true = y_true[indices]
        code_probs = np.array(
            [probability_rows[idx][code] for idx in indices],
            dtype=np.float64,
        )

        if len(np.unique(code_true)) < 2:
            thresholds[code] = float(default_threshold)
            diagnostics[code] = {
                "threshold": float(default_threshold),
                "score": None,
                "n": int(len(indices)),
                "label_counts": dict(Counter(int(x) for x in code_true)),
                "reason": "only_one_validation_class",
            }
            continue

        best_threshold = float(default_threshold)
        best_score = -np.inf
        best_prediction_counts: Dict[int, int] = {}

        for threshold in candidate_thresholds:
            code_pred = (code_probs >= threshold).astype(np.int64)
            score = threshold_score(code_true, code_pred, metric)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_prediction_counts = dict(Counter(int(x) for x in code_pred))

        thresholds[code] = best_threshold
        diagnostics[code] = {
            "threshold": best_threshold,
            "score": float(best_score),
            "n": int(len(indices)),
            "metric": metric,
            "label_counts": dict(Counter(int(x) for x in code_true)),
            "prediction_counts": best_prediction_counts,
        }

    return thresholds, diagnostics


# Function: Answer a question using a per-SCP tuned threshold.
# Inputs: ECG-QA row, all attribute probabilities, tuned thresholds, and fallback threshold.
# Outputs: Binary prediction label.
def tuned_threshold_prediction(
    row: Dict[str, Any],
    probs: Dict[str, float],
    thresholds: Dict[str, float],
    default_threshold: float,
) -> int:
    target_code = str(row["target_scp_code"])
    threshold = thresholds.get(target_code, default_threshold)
    return int(probs[target_code] >= threshold)


# Function: Compute validation metrics separately for each SCP code.
# Inputs: Validation rows, true labels, and predicted labels.
# Outputs: Nested dictionary of per-code metrics and counts.
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


# Function: Compute overall and per-code classification metrics.
# Inputs: Method name, validation rows, true labels, and predicted labels.
# Outputs: Dictionary of metrics, reports, and confusion matrix.
def evaluate_predictions(
    method: str,
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
        "method": method,
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


# Function: Load a compact summary of another baseline result file.
# Inputs: Path to a JSON result file.
# Outputs: Summary dictionary, or None if the file is missing/unrecognized.
def load_baseline_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "methods" in data:
        return {
            method: {
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "roc_auc": metrics.get("roc_auc"),
            }
            for method, metrics in data.get("methods", {}).items()
        }

    if "metrics" in data:
        metrics = data["metrics"]
        return {
            data.get("method", path.stem): {
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
            }
        }

    return None


# Function: Run the classifier-to-LLM baseline and direct upper-bound-style analyses.
# Inputs: Command-line arguments.
# Outputs: Result, prediction, and model-bundle files written to disk.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-metric",
        choices=["macro_f1", "balanced_accuracy"],
        default="macro_f1",
    )
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-bundle-path", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument(
        "--embedding-baseline-results-path",
        type=Path,
        default=EMBEDDING_BASELINE_RESULTS_PATH,
    )
    parser.add_argument(
        "--feature-llm-results-path",
        type=Path,
        default=FEATURE_LLM_RESULTS_PATH,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    if args.max_samples is not None:
        val_rows = val_rows[: args.max_samples]

    if not train_rows or not val_rows:
        raise RuntimeError("Train and validation rows must both be non-empty.")

    descriptions = get_code_descriptions(train_rows + val_rows)
    attribute_models = train_attribute_classifiers(train_rows, random_state=args.seed)

    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = args.predictions_path.open("w", encoding="utf-8")

    predictions: List[Dict[str, Any]] = []
    probability_rows: List[Dict[str, float]] = []
    direct_y_pred: List[int] = []
    llm_y_pred: List[int] = []
    y_true: List[int] = []

    try:
        for idx, row in enumerate(val_rows, start=1):
            probs = predict_attribute_probs(attribute_models, row["embedding"])
            classifier_text = build_classifier_text(
                probs,
                descriptions=descriptions,
                threshold=args.threshold,
            )
            prompt = build_llm_prompt(classifier_text, row["question"])

            true_label = int(row["label"])
            direct_pred = direct_threshold_prediction(row, probs, threshold=args.threshold)

            if args.skip_llm:
                raw_response = ""
                llm_pred = -1
                llm_answer = "skipped"
                error = None
            else:
                try:
                    raw_response = call_ollama(
                        args.model,
                        prompt,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
                    )
                    llm_pred, llm_answer = parse_yes_no_response(raw_response)
                    error = None
                except Exception as exc:
                    raw_response = ""
                    llm_pred = -1
                    llm_answer = "unknown"
                    error = str(exc)

            prediction = {
                "method": "classifier_llm",
                "model": args.model,
                "ecg_id": row.get("ecg_id"),
                "split": row.get("split"),
                "target_scp_code": row.get("target_scp_code"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "true_label": true_label,
                "direct_pred_label": int(direct_pred),
                "direct_pred_answer": "yes" if direct_pred == 1 else "no",
                "llm_pred_label": int(llm_pred),
                "llm_pred_answer": llm_answer,
                "raw_response": raw_response,
                "target_probability": float(probs[str(row["target_scp_code"])]),
                "attribute_probabilities": probs,
                "classifier_text": classifier_text,
                "prompt": prompt,
                "error": error,
                "direct_correct": bool(direct_pred == true_label),
                "llm_correct": bool(llm_pred == true_label),
            }

            predictions.append(prediction)
            probability_rows.append(probs)
            prediction_file.write(json.dumps(prediction) + "\n")
            prediction_file.flush()

            y_true.append(true_label)
            direct_y_pred.append(int(direct_pred))
            llm_y_pred.append(int(llm_pred))

            print(
                f"[{idx}/{len(val_rows)}] "
                f"ecg_id={row.get('ecg_id')} code={row.get('target_scp_code')} "
                f"true={row.get('answer')} direct={prediction['direct_pred_answer']} "
                f"llm={llm_answer}",
                flush=True,
            )
    finally:
        prediction_file.close()

    y_true_arr = np.array(y_true, dtype=np.int64)
    direct_pred_arr = np.array(direct_y_pred, dtype=np.int64)
    llm_pred_arr = np.array(llm_y_pred, dtype=np.int64)
    tuned_thresholds, threshold_tuning = tune_thresholds_by_code(
        rows=val_rows,
        probability_rows=probability_rows,
        y_true=y_true_arr,
        metric=args.threshold_metric,
        grid_size=args.threshold_grid_size,
        default_threshold=args.threshold,
    )
    tuned_direct_pred_arr = np.array(
        [
            tuned_threshold_prediction(row, probs, tuned_thresholds, args.threshold)
            for row, probs in zip(val_rows, probability_rows)
        ],
        dtype=np.int64,
    )

    for prediction, tuned_pred in zip(predictions, tuned_direct_pred_arr):
        prediction["tuned_direct_pred_label"] = int(tuned_pred)
        prediction["tuned_direct_pred_answer"] = "yes" if tuned_pred == 1 else "no"
        prediction["tuned_direct_threshold"] = float(
            tuned_thresholds.get(str(prediction["target_scp_code"]), args.threshold)
        )
        prediction["tuned_direct_correct"] = bool(tuned_pred == prediction["true_label"])

    write_jsonl(args.predictions_path, predictions)

    results = {
        "method": "classifier_llm",
        "model": args.model,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "threshold": args.threshold,
        "threshold_metric": args.threshold_metric,
        "threshold_grid_size": args.threshold_grid_size,
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "attribute_codes": sorted(attribute_models.keys()),
        "direct_attribute_threshold_metrics": evaluate_predictions(
            "direct_attribute_threshold",
            val_rows,
            y_true_arr,
            direct_pred_arr,
        ),
        "tuned_direct_attribute_threshold_metrics": evaluate_predictions(
            "tuned_direct_attribute_threshold",
            val_rows,
            y_true_arr,
            tuned_direct_pred_arr,
        ),
        "tuned_thresholds_by_code": tuned_thresholds,
        "threshold_tuning": threshold_tuning,
        "classifier_llm_metrics": None
        if args.skip_llm
        else evaluate_predictions("classifier_llm", val_rows, y_true_arr, llm_pred_arr),
        "embedding_baseline_summary": load_baseline_summary(args.embedding_baseline_results_path),
        "feature_llm_summary": load_baseline_summary(args.feature_llm_results_path),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    args.model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "attribute_models": attribute_models,
            "descriptions": descriptions,
            "threshold": args.threshold,
            "tuned_thresholds_by_code": tuned_thresholds,
            "threshold_metric": args.threshold_metric,
            "results": results,
        },
        args.model_bundle_path,
    )

    print("\n=== Direct Attribute Threshold ===")
    direct_metrics = results["direct_attribute_threshold_metrics"]
    print("Accuracy:", direct_metrics["accuracy"])
    print("Balanced accuracy:", direct_metrics["balanced_accuracy"])
    print("Macro-F1:", direct_metrics["macro_f1"])

    print("\n=== Tuned Direct Attribute Threshold ===")
    tuned_metrics = results["tuned_direct_attribute_threshold_metrics"]
    print("Accuracy:", tuned_metrics["accuracy"])
    print("Balanced accuracy:", tuned_metrics["balanced_accuracy"])
    print("Macro-F1:", tuned_metrics["macro_f1"])
    print("Threshold metric:", args.threshold_metric)

    if results["classifier_llm_metrics"] is not None:
        print("\n=== Classifier-to-LLM Baseline ===")
        llm_metrics = results["classifier_llm_metrics"]
        print("Accuracy:", llm_metrics["accuracy"])
        print("Balanced accuracy:", llm_metrics["balanced_accuracy"])
        print("Macro-F1:", llm_metrics["macro_f1"])
        print("Invalid predictions:", llm_metrics["invalid_predictions"])

    print(f"\nSaved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")
    print(f"Saved model bundle: {args.model_bundle_path}")


if __name__ == "__main__":
    main()
