from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_RESULTS_PATH = Path("outputs/ecg_question_swap_3b_adapter_results.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/ecg_question_swap_3b_adapter_predictions.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/ecg_question_swap_analysis")


# Function: Load newline-delimited JSON prediction rows.
# Inputs: path to a JSONL file.
# Outputs: list of parsed dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Function: Write a dictionary to a JSON file.
# Inputs: output path and JSON-serializable payload.
# Outputs: None; writes JSON to disk.
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Write dictionaries to a CSV file.
# Inputs: output path and row dictionaries.
# Outputs: None; writes CSV to disk.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Compute same-question pair flip metrics grouped by SCP code.
# Inputs: prediction rows from the ECG/question swap experiment.
# Outputs: list of per-code pair-level metric rows.
def same_question_pair_rows(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_code_pair: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["test_type"] == "same_question_different_ecg":
            by_code_pair[str(row["target_scp_code"])][str(row["pair_id"])].append(row)

    rows: List[Dict[str, Any]] = []
    for code, by_pair in sorted(by_code_pair.items()):
        pairs = [pair_rows for pair_rows in by_pair.values() if len(pair_rows) == 2]
        both_correct = 0
        prediction_flips = 0
        expected_pattern = 0
        for pair_rows in pairs:
            label_to_pred = {int(row["true_label"]): int(row["pred_label"]) for row in pair_rows}
            both_correct += int(all(row["correct"] for row in pair_rows))
            prediction_flips += int(len(set(label_to_pred.values())) == 2)
            expected_pattern += int(label_to_pred.get(1) == 1 and label_to_pred.get(0) == 0)
        n = len(pairs)
        rows.append(
            {
                "scp_code": code,
                "n_pairs": n,
                "both_correct_pairs": both_correct,
                "both_correct_rate": both_correct / n if n else None,
                "prediction_flip_pairs": prediction_flips,
                "prediction_flip_rate": prediction_flips / n if n else None,
                "expected_yes_no_pattern_pairs": expected_pattern,
                "expected_yes_no_pattern_rate": expected_pattern / n if n else None,
            }
        )
    return rows


# Function: Convert per-code metrics from the result JSON into a flat table.
# Inputs: result JSON dictionary.
# Outputs: list of per-code metric rows.
def per_code_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for code, metrics in sorted(results["metrics_by_code"].items()):
        rows.append(
            {
                "scp_code": code,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "yes_recall": metrics["yes_recall"],
                "no_recall": metrics["no_recall"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
                "prediction_counts": json.dumps(metrics["prediction_counts"]),
                "true_counts": json.dumps(metrics["true_counts"]),
            }
        )
    return rows


# Function: Extract representative correct and failure examples from prediction rows.
# Inputs: prediction rows and maximum examples per category.
# Outputs: dictionary of example lists.
def collect_examples(predictions: List[Dict[str, Any]], limit: int) -> Dict[str, List[Dict[str, Any]]]:
    examples = {
        "correct_same_question": [],
        "failed_same_question": [],
        "correct_same_ecg": [],
        "failed_same_ecg": [],
    }
    for row in predictions:
        key = "correct_" if row["correct"] else "failed_"
        key += "same_question" if row["test_type"] == "same_question_different_ecg" else "same_ecg"
        if len(examples[key]) < limit:
            examples[key].append(
                {
                    "test_type": row["test_type"],
                    "pair_id": row["pair_id"],
                    "ecg_id": row["ecg_id"],
                    "target_scp_code": row["target_scp_code"],
                    "question": row["question"],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "yes_score": row["yes_score"],
                    "no_score": row["no_score"],
                }
            )
    return examples


# Function: Run ECG/question swap output analysis.
# Inputs: command-line arguments.
# Outputs: JSON and CSV analysis files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--example-limit", type=int, default=8)
    args = parser.parse_args()

    with args.results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    predictions = load_jsonl(args.predictions_path)
    pair_rows = same_question_pair_rows(predictions)
    code_rows = per_code_rows(results)
    examples = collect_examples(predictions, limit=args.example_limit)

    summary = {
        "source_results_path": str(args.results_path),
        "source_predictions_path": str(args.predictions_path),
        "n_examples": len(predictions),
        "target_codes": results["target_codes"],
        "metrics_by_type": results["metrics_by_type"],
        "same_question_pair_metrics": results["same_question_pair_metrics"],
        "same_ecg_question_metrics": results["same_ecg_question_metrics"],
        "examples": examples,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ecg_question_swap_summary.json", summary)
    write_csv(args.output_dir / "ecg_question_swap_by_code.csv", code_rows)
    write_csv(args.output_dir / "ecg_question_swap_same_question_pairs_by_code.csv", pair_rows)

    print("Metrics by test type")
    for test_type, metrics in results["metrics_by_type"].items():
        print(
            f"{test_type:30s} n={metrics['n']:>3} "
            f"bal={metrics['balanced_accuracy']:.3f} "
            f"acc={metrics['accuracy']:.3f} "
            f"yes={metrics['yes_recall']:.3f} "
            f"no={metrics['no_recall']:.3f}"
        )
    print("Same-question pair metrics:", results["same_question_pair_metrics"])
    print("Same-ECG question metrics:", results["same_ecg_question_metrics"])
    print(f"Saved summary: {args.output_dir / 'ecg_question_swap_summary.json'}")
    print(f"Saved code metrics: {args.output_dir / 'ecg_question_swap_by_code.csv'}")
    print(f"Saved pair metrics: {args.output_dir / 'ecg_question_swap_same_question_pairs_by_code.csv'}")


if __name__ == "__main__":
    main()
