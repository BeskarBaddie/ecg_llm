from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BASELINE_RESULTS = Path(
    "outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_"
    "lr5e_6_batch1_accum4_adapter_full_all_codes_results.json"
)
DEFAULT_CLASS_WEIGHTED_RESULTS = Path(
    "outputs/20260720_191645_20_epochs_llama_3_2_3b_class_weighted_balanced_loss_float16_"
    "llm_float32_adapter_adapter_full_all_codes_results.json"
)
DEFAULT_BASELINE_THRESHOLD = Path("outputs/threshold_tuning/threshold_tuning_summary.csv")
DEFAULT_CLASS_WEIGHTED_THRESHOLD = Path("outputs/threshold_tuning_class_weighted/threshold_tuning_summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/class_weighted_analysis")


# Function: Load a JSON file.
# Inputs: path to JSON.
# Outputs: parsed dictionary.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load a CSV file.
# Inputs: path to CSV.
# Outputs: list of dictionaries.
def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Function: Write dictionaries to CSV.
# Inputs: path and row dictionaries.
# Outputs: None; writes CSV.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Extract headline validation metrics from a training result file.
# Inputs: run label and result JSON.
# Outputs: flat metric row.
def result_row(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    return {
        "run": name,
        "best_epoch": result.get("best_epoch"),
        "best_balanced_accuracy": result.get("best_metric_value"),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "yes_recall": metrics["labels"]["yes"]["recall"],
        "no_recall": metrics["labels"]["no"]["recall"],
        "yes_precision": metrics["labels"]["yes"]["precision"],
        "no_precision": metrics["labels"]["no"]["precision"],
        "pred_yes": metrics["prediction_counts"].get("1", metrics["prediction_counts"].get(1, 0)),
        "pred_no": metrics["prediction_counts"].get("0", metrics["prediction_counts"].get(0, 0)),
        "class_weight_mode": result.get("class_weight_mode"),
        "class_weights": json.dumps(result.get("class_weights")),
    }


# Function: Load threshold tuning summary rows and tag with a run name.
# Inputs: run label and threshold summary path.
# Outputs: tagged threshold rows.
def threshold_rows(name: str, path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in load_csv(path):
        rows.append({"run": name, **row})
    return rows


# Function: Compare baseline and class-weighted validation behavior.
# Inputs: command-line arguments.
# Outputs: CSV tables and console summary.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--class-weighted-results", type=Path, default=DEFAULT_CLASS_WEIGHTED_RESULTS)
    parser.add_argument("--baseline-threshold-summary", type=Path, default=DEFAULT_BASELINE_THRESHOLD)
    parser.add_argument("--class-weighted-threshold-summary", type=Path, default=DEFAULT_CLASS_WEIGHTED_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    baseline = load_json(args.baseline_results)
    class_weighted = load_json(args.class_weighted_results)
    validation_rows = [
        result_row("baseline_unweighted", baseline),
        result_row("class_weighted_balanced", class_weighted),
    ]
    threshold_summary = (
        threshold_rows("baseline_unweighted", args.baseline_threshold_summary)
        + threshold_rows("class_weighted_balanced", args.class_weighted_threshold_summary)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "class_weighted_validation_comparison.csv", validation_rows)
    write_csv(args.output_dir / "class_weighted_threshold_comparison.csv", threshold_summary)

    print("Validation comparison")
    for row in validation_rows:
        print(
            f"{row['run']:24s} bal={float(row['balanced_accuracy']):.3f} "
            f"acc={float(row['accuracy']):.3f} "
            f"yes={float(row['yes_recall']):.3f} "
            f"no={float(row['no_recall']):.3f} "
            f"best_epoch={row['best_epoch']}"
        )
    print("\nThreshold-tuned validation comparison")
    for row in threshold_summary:
        if row["method"] == "global_tuned_threshold":
            print(
                f"{row['run']:24s} global_bal={float(row['balanced_accuracy']):.3f} "
                f"yes={float(row['yes_recall']):.3f} "
                f"no={float(row['no_recall']):.3f} "
                f"threshold={row['threshold']}"
            )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
