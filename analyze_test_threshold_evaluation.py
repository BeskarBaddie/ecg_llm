from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RESULTS_PATH = Path("outputs/adapter_3b_test_threshold_evaluation_results.json")
DEFAULT_OUTPUT_DIR = Path("outputs/test_threshold_analysis")


# Function: Load a JSON dictionary from disk.
# Inputs: path to a JSON file.
# Outputs: parsed JSON dictionary.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


# Function: Convert overall test metrics into a flat table.
# Inputs: result JSON dictionary.
# Outputs: list of summary rows, one per threshold method.
def make_overall_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in results["metrics"]["overall"].items():
        rows.append(
            {
                "method": method,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "yes_recall": metrics["yes_recall"],
                "no_recall": metrics["no_recall"],
                "yes_precision": metrics["yes_precision"],
                "no_precision": metrics["no_precision"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
                "pred_yes": metrics["prediction_counts"].get("1", metrics["prediction_counts"].get(1, 0)),
                "pred_no": metrics["prediction_counts"].get("0", metrics["prediction_counts"].get(0, 0)),
            }
        )
    return rows


# Function: Convert per-code test metrics into a flat comparison table.
# Inputs: result JSON dictionary.
# Outputs: list of per-code rows comparing default, global, and per-code thresholds.
def make_per_code_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for code, methods in sorted(results["metrics"]["per_code"].items()):
        default = methods["default_threshold"]
        global_tuned = methods["global_tuned_threshold"]
        per_code = methods["per_code_tuned_thresholds"]
        rows.append(
            {
                "scp_code": code,
                "n": default["n"],
                "true_yes": default["label_counts"].get("1", default["label_counts"].get(1, 0)),
                "true_no": default["label_counts"].get("0", default["label_counts"].get(0, 0)),
                "default_balanced_accuracy": default["balanced_accuracy"],
                "global_balanced_accuracy": global_tuned["balanced_accuracy"],
                "per_code_balanced_accuracy": per_code["balanced_accuracy"],
                "global_delta_vs_default": global_tuned["balanced_accuracy"] - default["balanced_accuracy"],
                "per_code_delta_vs_default": per_code["balanced_accuracy"] - default["balanced_accuracy"],
                "per_code_delta_vs_global": per_code["balanced_accuracy"] - global_tuned["balanced_accuracy"],
                "default_yes_recall": default["yes_recall"],
                "global_yes_recall": global_tuned["yes_recall"],
                "per_code_yes_recall": per_code["yes_recall"],
                "default_no_recall": default["no_recall"],
                "global_no_recall": global_tuned["no_recall"],
                "per_code_no_recall": per_code["no_recall"],
            }
        )
    return rows


# Function: Print concise held-out test threshold findings.
# Inputs: overall and per-code rows.
# Outputs: None; prints summary.
def print_summary(overall_rows: List[Dict[str, Any]], per_code_rows: List[Dict[str, Any]]) -> None:
    print("Held-out test threshold comparison")
    for row in overall_rows:
        print(
            f"{row['method']:28s} "
            f"bal={row['balanced_accuracy']:.3f} "
            f"acc={row['accuracy']:.3f} "
            f"yes={row['yes_recall']:.3f} "
            f"no={row['no_recall']:.3f} "
            f"pred_yes={row['pred_yes']}"
        )

    print("\nLargest global-threshold gains")
    for row in sorted(per_code_rows, key=lambda item: item["global_delta_vs_default"], reverse=True)[:10]:
        print(
            f"{row['scp_code']:8s} n={row['n']:>4} "
            f"default={row['default_balanced_accuracy']:.3f} "
            f"global={row['global_balanced_accuracy']:.3f} "
            f"delta={row['global_delta_vs_default']:+.3f}"
        )

    print("\nLargest global-threshold drops")
    for row in sorted(per_code_rows, key=lambda item: item["global_delta_vs_default"])[:10]:
        print(
            f"{row['scp_code']:8s} n={row['n']:>4} "
            f"default={row['default_balanced_accuracy']:.3f} "
            f"global={row['global_balanced_accuracy']:.3f} "
            f"delta={row['global_delta_vs_default']:+.3f}"
        )


# Function: Run held-out test threshold analysis from saved result JSON.
# Inputs: command-line arguments.
# Outputs: CSV tables for overall and per-code metrics.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results = load_json(args.results_path)
    overall_rows = make_overall_rows(results)
    per_code_rows = make_per_code_rows(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "test_threshold_overall.csv", overall_rows)
    write_csv(args.output_dir / "test_threshold_by_code.csv", per_code_rows)
    print_summary(overall_rows, per_code_rows)
    print(f"\nSaved overall table: {args.output_dir / 'test_threshold_overall.csv'}")
    print(f"Saved per-code table: {args.output_dir / 'test_threshold_by_code.csv'}")


if __name__ == "__main__":
    main()
