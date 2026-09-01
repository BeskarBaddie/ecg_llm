from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from analyze_scp_code_performance import compute_binary_metrics
from train_ecg_soft_prompt_adapter import load_jsonl, write_json


DEFAULT_OUTPUT_DIR = Path("outputs/question_novelty_analysis")


# Function: Normalize question text for overlap checks.
# Inputs: raw question string.
# Outputs: lowercase whitespace-normalized question string.
def normalize_question(question: str) -> str:
    question = str(question).strip().lower()
    question = re.sub(r"\s+", " ", question)
    return question


# Function: Build a stable prediction lookup key.
# Inputs: prediction or dataset row containing ECG ID, target SCP code, and question.
# Outputs: tuple key used to join predictions to dataset rows.
def row_key(row: Dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["ecg_id"]),
        str(row["target_scp_code"]),
        normalize_question(str(row["question"])),
    )


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


# Function: Convert true/pred labels into the project's standard metric dictionary.
# Inputs: rows containing true_label and pred_label.
# Outputs: binary classification metrics.
def metrics_for_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), get_prediction_label(row)) for row in rows)
    return compute_binary_metrics(
        tn=counts[(0, 0)],
        fp=counts[(0, 1)],
        fn=counts[(1, 0)],
        tp=counts[(1, 1)],
    )


# Function: Read the default prediction label from either validation or test prediction format.
# Inputs: prediction row.
# Outputs: integer predicted label.
def get_prediction_label(row: Dict[str, Any]) -> int:
    if "pred_label" in row:
        return int(row["pred_label"])
    if "default_pred_label" in row:
        return int(row["default_pred_label"])
    margin = float(row["yes_score"]) - float(row["no_score"])
    return int(margin >= 0.0)


# Function: Summarize exact-question overlap between train and evaluation rows.
# Inputs: train rows, evaluation rows, prediction rows, and split name.
# Outputs: summary dictionary and per-row augmented prediction dictionaries.
def analyze_split(
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    split_name: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_questions = {normalize_question(str(row["question"])) for row in train_rows}
    train_question_code = {
        (normalize_question(str(row["question"])), str(row["target_scp_code"]))
        for row in train_rows
    }
    train_ecg_ids = {int(row["ecg_id"]) for row in train_rows}

    eval_by_key = {row_key(row): row for row in eval_rows}
    augmented: List[Dict[str, Any]] = []
    missing_dataset_rows = 0
    for pred in prediction_rows:
        key = row_key(pred)
        eval_row = eval_by_key.get(key)
        if eval_row is None:
            missing_dataset_rows += 1
        question_norm = key[2]
        question_code_key = (question_norm, str(pred["target_scp_code"]))
        augmented.append(
            {
                **pred,
                "split": split_name,
                "normalized_question": question_norm,
                "question_seen_in_train": question_norm in train_questions,
                "question_code_seen_in_train": question_code_key in train_question_code,
                "ecg_seen_in_train": int(pred["ecg_id"]) in train_ecg_ids,
            }
        )

    summary_rows: Dict[str, Dict[str, Any]] = {}
    groupings = {
        "all": lambda row: "all",
        "question_seen_in_train": lambda row: "seen" if row["question_seen_in_train"] else "novel",
        "question_code_seen_in_train": lambda row: "seen" if row["question_code_seen_in_train"] else "novel",
        "ecg_seen_in_train": lambda row: "seen" if row["ecg_seen_in_train"] else "novel",
    }
    for grouping_name, group_fn in groupings.items():
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in augmented:
            grouped[group_fn(row)].append(row)
        for group_value, rows in sorted(grouped.items()):
            metric_row = metrics_for_rows(rows)
            summary_rows[f"{grouping_name}:{group_value}"] = {
                "split": split_name,
                "grouping": grouping_name,
                "group": group_value,
                **metric_row,
            }

    summary = {
        "split": split_name,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "prediction_rows": len(prediction_rows),
        "missing_dataset_rows": missing_dataset_rows,
        "unique_train_questions": len(train_questions),
        "unique_eval_questions": len({row["normalized_question"] for row in augmented}),
        "unique_train_question_code_pairs": len(train_question_code),
        "unique_eval_question_code_pairs": len(
            {(row["normalized_question"], row["target_scp_code"]) for row in augmented}
        ),
        "metrics": summary_rows,
    }
    return summary, augmented


# Function: Print a concise question-novelty summary.
# Inputs: summary dictionaries.
# Outputs: None; prints key metrics.
def print_summary(summaries: Iterable[Dict[str, Any]]) -> None:
    for summary in summaries:
        print(f"\nSplit: {summary['split']}")
        print(f"Rows: predictions={summary['prediction_rows']} eval={summary['eval_rows']}")
        print(f"Unique train questions: {summary['unique_train_questions']}")
        print(f"Unique eval questions: {summary['unique_eval_questions']}")
        for key, metrics in summary["metrics"].items():
            if not key.startswith("question_seen_in_train:"):
                continue
            print(
                f"{key:32s} n={metrics['n']:>5} "
                f"bal={metrics['balanced_accuracy']:.3f} "
                f"acc={metrics['accuracy']:.3f} "
                f"yes={metrics['yes_recall']:.3f} "
                f"no={metrics['no_recall']:.3f}"
            )


# Function: Run question-novelty analysis for validation and/or test predictions.
# Inputs: command-line paths.
# Outputs: JSON, CSV, and JSONL files with novelty-stratified metrics.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--val-predictions-path", type=Path)
    parser.add_argument("--test-predictions-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    analyses: List[Dict[str, Any]] = []
    augmented_all: List[Dict[str, Any]] = []

    split_specs = [
        ("val", args.val_path, args.val_predictions_path),
        ("test", args.test_path, args.test_predictions_path),
    ]
    for split_name, split_path, pred_path in split_specs:
        if split_path is None or pred_path is None:
            continue
        split_rows = load_jsonl(split_path)
        predictions = load_jsonl(pred_path)
        summary, augmented = analyze_split(
            train_rows=train_rows,
            eval_rows=split_rows,
            prediction_rows=predictions,
            split_name=split_name,
        )
        analyses.append(summary)
        augmented_all.extend(augmented)

    if not analyses:
        raise ValueError("Provide at least one split path and matching predictions path.")

    flat_rows: List[Dict[str, Any]] = []
    for summary in analyses:
        for metrics in summary["metrics"].values():
            flat_rows.append(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "question_novelty_summary.json", {"splits": analyses})
    write_csv(args.output_dir / "question_novelty_metrics.csv", flat_rows)
    with (args.output_dir / "question_novelty_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in augmented_all:
            f.write(json.dumps(row) + "\n")

    print_summary(analyses)
    print(f"\nSaved summary: {args.output_dir / 'question_novelty_summary.json'}")
    print(f"Saved metrics: {args.output_dir / 'question_novelty_metrics.csv'}")


if __name__ == "__main__":
    main()
