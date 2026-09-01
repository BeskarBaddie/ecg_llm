from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_scp_code_performance import clinical_group, load_scp_metadata
from train_ecg_soft_prompt_adapter import load_jsonl, write_json


DEFAULT_OUTPUT_DIR = Path("outputs/log_likelihood_score_analysis")
DEFAULT_FIGURE_DIR = Path("figures/log_likelihood_score_analysis")


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


# Function: Add score-margin and metadata fields to prediction rows.
# Inputs: prediction rows and optional SCP metadata.
# Outputs: augmented rows with score_diff and group fields.
def augment_rows(rows: List[Dict[str, Any]], metadata: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        code = str(row["target_scp_code"])
        meta = metadata.get(code, {})
        yes_score = float(row["yes_score"])
        no_score = float(row["no_score"])
        pred_label = int(row.get("pred_label", row.get("default_pred_label", int((yes_score - no_score) >= 0.0))))
        pred_answer = str(row.get("pred_answer", "yes" if pred_label == 1 else "no"))
        correct = bool(row.get("correct", pred_label == int(row["true_label"])))
        output.append(
            {
                **row,
                "score_diff": yes_score - no_score,
                "clinical_group": clinical_group(meta) if meta else "unknown",
                "description": meta.get("description", code),
                "true_answer": "yes" if int(row["true_label"]) == 1 else "no",
                "pred_answer": pred_answer,
                "correct": correct,
            }
        )
    return output


# Function: Calculate simple distribution summaries for score fields.
# Inputs: augmented rows and grouping field.
# Outputs: list of summary dictionaries.
def summarize_scores(rows: List[Dict[str, Any]], group_field: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)

    summaries: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        for answer in ["yes", "no"]:
            answer_rows = [row for row in group_rows if row["true_answer"] == answer]
            if not answer_rows:
                continue
            diffs = sorted(float(row["score_diff"]) for row in answer_rows)
            n = len(diffs)
            summaries.append(
                {
                    "group_field": group_field,
                    "group": group,
                    "true_answer": answer,
                    "n": n,
                    "score_diff_min": diffs[0],
                    "score_diff_q25": diffs[int(0.25 * (n - 1))],
                    "score_diff_median": diffs[int(0.50 * (n - 1))],
                    "score_diff_q75": diffs[int(0.75 * (n - 1))],
                    "score_diff_max": diffs[-1],
                }
            )
    return summaries


# Function: Plot score-difference histograms by true answer.
# Inputs: augmented rows, figure path, and plot title.
# Outputs: None; saves PNG.
def plot_overall_histogram(rows: List[Dict[str, Any]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yes = [row["score_diff"] for row in rows if row["true_answer"] == "yes"]
    no = [row["score_diff"] for row in rows if row["true_answer"] == "no"]

    plt.figure(figsize=(9, 5))
    plt.hist(no, bins=80, alpha=0.65, label="True no", density=True)
    plt.hist(yes, bins=80, alpha=0.65, label="True yes", density=True)
    plt.axvline(0, color="black", linestyle="--", linewidth=1, label="Default threshold")
    plt.xlabel("yes_score - no_score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# Function: Plot score-difference boxplots grouped by clinical group.
# Inputs: augmented rows and figure path.
# Outputs: None; saves PNG.
def plot_group_boxplot(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted({row["clinical_group"] for row in rows})
    data = [[row["score_diff"] for row in rows if row["clinical_group"] == group] for group in groups]

    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=groups, showfliers=False)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("yes_score - no_score")
    plt.title("Score margins by clinical group")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# Function: Run score-distribution analysis for adapter predictions.
# Inputs: command-line paths.
# Outputs: CSV/JSON summaries and PNG figures.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    metadata = load_scp_metadata(args.scp_metadata_csv) if args.scp_metadata_csv and args.scp_metadata_csv.exists() else {}
    rows = augment_rows(load_jsonl(args.predictions_path), metadata=metadata)

    summaries = (
        summarize_scores(rows, "true_answer")
        + summarize_scores(rows, "target_scp_code")
        + summarize_scores(rows, "clinical_group")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "score_distribution_summary.csv", summaries)
    write_json(
        args.output_dir / "score_distribution_metadata.json",
        {
            "predictions_path": str(args.predictions_path),
            "scp_metadata_csv": str(args.scp_metadata_csv) if args.scp_metadata_csv else None,
            "n": len(rows),
        },
    )
    write_csv(
        args.output_dir / "score_rows.csv",
        [
            {
                "ecg_id": row["ecg_id"],
                "target_scp_code": row["target_scp_code"],
                "clinical_group": row["clinical_group"],
                "true_answer": row["true_answer"],
                "pred_answer": row["pred_answer"],
                "correct": row["correct"],
                "yes_score": row["yes_score"],
                "no_score": row["no_score"],
                "score_diff": row["score_diff"],
                "question": row["question"],
            }
            for row in rows
        ],
    )

    plot_overall_histogram(
        rows,
        args.figure_dir / "score_diff_by_true_answer.png",
        "LLM yes/no log-likelihood score margin by true answer",
    )
    plot_group_boxplot(rows, args.figure_dir / "score_diff_by_clinical_group.png")

    print(f"Rows: {len(rows)}")
    print(f"Saved summary: {args.output_dir / 'score_distribution_summary.csv'}")
    print(f"Saved figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
