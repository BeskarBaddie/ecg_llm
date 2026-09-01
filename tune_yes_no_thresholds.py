from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_PREDICTIONS_PATH = Path(
    "outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_"
    "lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/threshold_tuning")


# Function: Load newline-delimited prediction rows from disk.
# Inputs: path to a JSONL predictions file.
# Outputs: list of prediction dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


# Function: Write a dictionary to a JSON file.
# Inputs: output path and JSON-serializable payload.
# Outputs: None; writes JSON to disk.
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Write dictionaries to a CSV file.
# Inputs: output path and rows.
# Outputs: None; writes CSV to disk.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Compute the yes-vs-no score margin for one prediction row.
# Inputs: prediction row containing yes_score and no_score.
# Outputs: positive score when yes is preferred over no.
def score_margin(row: Dict[str, Any]) -> float:
    return float(row["yes_score"]) - float(row["no_score"])


# Function: Calculate binary classification metrics from true and predicted labels.
# Inputs: true labels and predicted labels.
# Outputs: dictionary of accuracy, balanced accuracy, recalls, precision, F1, and confusion counts.
def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    counts = Counter(zip(y_true, y_pred))
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    n = tn + fp + fn + tp

    no_recall = tn / (tn + fp) if (tn + fp) else None
    yes_recall = tp / (tp + fn) if (tp + fn) else None
    no_precision = tn / (tn + fn) if (tn + fn) else None
    yes_precision = tp / (tp + fp) if (tp + fp) else None
    no_f1 = (
        2 * no_precision * no_recall / (no_precision + no_recall)
        if no_precision is not None and no_recall is not None and no_precision + no_recall > 0
        else 0.0
    )
    yes_f1 = (
        2 * yes_precision * yes_recall / (yes_precision + yes_recall)
        if yes_precision is not None and yes_recall is not None and yes_precision + yes_recall > 0
        else 0.0
    )

    return {
        "n": n,
        "accuracy": (tn + tp) / n if n else None,
        "balanced_accuracy": (yes_recall + no_recall) / 2
        if yes_recall is not None and no_recall is not None
        else None,
        "macro_f1": (yes_f1 + no_f1) / 2,
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "yes_precision": yes_precision,
        "no_precision": no_precision,
        "yes_f1": yes_f1,
        "no_f1": no_f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "prediction_counts": dict(Counter(y_pred)),
        "label_counts": dict(Counter(y_true)),
    }


# Function: Convert margins into binary predictions using a threshold.
# Inputs: margin values and decision threshold.
# Outputs: predicted labels where 1 means yes.
def predict_from_threshold(margins: Sequence[float], threshold: float) -> List[int]:
    return [int(margin >= threshold) for margin in margins]


# Function: Generate candidate thresholds from score margins.
# Inputs: score margins and optional maximum number of candidates.
# Outputs: sorted candidate thresholds covering all possible decision boundaries.
def candidate_thresholds(margins: Sequence[float], max_candidates: int) -> List[float]:
    unique = sorted(set(float(x) for x in margins))
    if not unique:
        return [0.0]

    thresholds = [unique[0] - 1e-6, unique[-1] + 1e-6]
    thresholds.extend((a + b) / 2 for a, b in zip(unique, unique[1:]))
    thresholds.append(0.0)
    thresholds = sorted(set(thresholds))

    if len(thresholds) <= max_candidates:
        return thresholds

    step = (len(thresholds) - 1) / (max_candidates - 1)
    selected = [thresholds[round(i * step)] for i in range(max_candidates)]
    selected.append(0.0)
    return sorted(set(selected))


# Function: Select the threshold that maximizes a validation metric.
# Inputs: rows, metric name, maximum number of threshold candidates.
# Outputs: best threshold, best metrics, and default-threshold metrics.
def tune_global_threshold(
    rows: List[Dict[str, Any]],
    metric: str,
    max_candidates: int,
) -> Dict[str, Any]:
    y_true = [int(row["true_label"]) for row in rows]
    margins = [score_margin(row) for row in rows]
    default_metrics = binary_metrics(y_true, predict_from_threshold(margins, 0.0))

    best_threshold = 0.0
    best_metrics = default_metrics
    best_score = float(default_metrics[metric])
    for threshold in candidate_thresholds(margins, max_candidates=max_candidates):
        metrics = binary_metrics(y_true, predict_from_threshold(margins, threshold))
        score = float(metrics[metric])
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return {
        "default_threshold": 0.0,
        "default_metrics": default_metrics,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "optimized_metric": metric,
    }


# Function: Tune independent thresholds for each target SCP code.
# Inputs: rows, metric name, max candidates, and minimum positives/negatives per code.
# Outputs: per-code threshold diagnostics and aggregate metrics after applying code thresholds.
def tune_thresholds_by_code(
    rows: List[Dict[str, Any]],
    metric: str,
    max_candidates: int,
    min_pos: int,
    min_neg: int,
    fallback_threshold: float,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_scp_code"])].append(row)

    diagnostics: Dict[str, Any] = {}
    predictions: List[int] = []
    y_true_all: List[int] = []
    for code, code_rows in sorted(grouped.items()):
        y_true = [int(row["true_label"]) for row in code_rows]
        margins = [score_margin(row) for row in code_rows]
        counts = Counter(y_true)
        if counts[1] < min_pos or counts[0] < min_neg:
            threshold = fallback_threshold
            reason = "fallback_insufficient_support"
        else:
            tuned = tune_global_threshold(code_rows, metric=metric, max_candidates=max_candidates)
            threshold = float(tuned["best_threshold"])
            reason = "tuned"

        y_pred = predict_from_threshold(margins, threshold)
        diagnostics[code] = {
            "threshold": threshold,
            "reason": reason,
            **binary_metrics(y_true, y_pred),
        }
        predictions.extend(y_pred)
        y_true_all.extend(y_true)

    return {
        "aggregate_metrics": binary_metrics(y_true_all, predictions),
        "per_code": diagnostics,
    }


# Function: Build concise table rows for threshold-tuning comparison.
# Inputs: global and per-code tuning results.
# Outputs: list of summary rows.
def make_summary_rows(global_result: Dict[str, Any], per_code_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, threshold, metrics in [
        ("default_margin_threshold", 0.0, global_result["default_metrics"]),
        ("global_tuned_threshold", global_result["best_threshold"], global_result["best_metrics"]),
        ("per_code_tuned_thresholds", "varies", per_code_result["aggregate_metrics"]),
    ]:
        rows.append(
            {
                "method": name,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "yes_recall": metrics["yes_recall"],
                "no_recall": metrics["no_recall"],
                "yes_precision": metrics["yes_precision"],
                "no_precision": metrics["no_precision"],
                "pred_yes": metrics["prediction_counts"].get(1, 0),
                "pred_no": metrics["prediction_counts"].get(0, 0),
            }
        )
    return rows


# Function: Run yes/no margin threshold tuning from prediction scores.
# Inputs: command-line arguments.
# Outputs: JSON and CSV threshold-tuning artifacts.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "macro_f1", "accuracy"])
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--per-code-min-pos", type=int, default=5)
    parser.add_argument("--per-code-min-neg", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.predictions_path)
    global_result = tune_global_threshold(
        rows,
        metric=args.metric,
        max_candidates=args.max_candidates,
    )
    per_code_result = tune_thresholds_by_code(
        rows,
        metric=args.metric,
        max_candidates=args.max_candidates,
        min_pos=args.per_code_min_pos,
        min_neg=args.per_code_min_neg,
        fallback_threshold=float(global_result["best_threshold"]),
    )
    summary_rows = make_summary_rows(global_result, per_code_result)

    per_code_rows = []
    for code, metrics in sorted(per_code_result["per_code"].items()):
        per_code_rows.append(
            {
                "scp_code": code,
                "threshold": metrics["threshold"],
                "reason": metrics["reason"],
                "n": metrics["n"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "yes_recall": metrics["yes_recall"],
                "no_recall": metrics["no_recall"],
                "pred_yes": metrics["prediction_counts"].get(1, 0),
                "pred_no": metrics["prediction_counts"].get(0, 0),
                "true_yes": metrics["label_counts"].get(1, 0),
                "true_no": metrics["label_counts"].get(0, 0),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "threshold_tuning_results.json",
        {
            "predictions_path": str(args.predictions_path),
            "optimized_metric": args.metric,
            "global": global_result,
            "per_code": per_code_result,
            "summary_rows": summary_rows,
        },
    )
    write_csv(args.output_dir / "threshold_tuning_summary.csv", summary_rows)
    write_csv(args.output_dir / "threshold_tuning_by_code.csv", per_code_rows)

    print("Threshold tuning summary")
    for row in summary_rows:
        print(
            f"{row['method']:28s} threshold={row['threshold']} "
            f"bal={float(row['balanced_accuracy']):.3f} "
            f"acc={float(row['accuracy']):.3f} "
            f"yes_rec={float(row['yes_recall']):.3f} "
            f"no_rec={float(row['no_recall']):.3f} "
            f"pred_yes={row['pred_yes']}"
        )
    print(f"Saved summary: {args.output_dir / 'threshold_tuning_summary.csv'}")
    print(f"Saved per-code thresholds: {args.output_dir / 'threshold_tuning_by_code.csv'}")


if __name__ == "__main__":
    main()
