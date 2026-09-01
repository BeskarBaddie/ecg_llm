from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analyze_scp_code_performance import (
    clinical_group,
    compute_binary_metrics,
    load_jsonl,
    load_scp_metadata,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_PATH = Path("outputs/ecg_token_ablation_3b_adapter_results.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/ecg_token_ablation_3b_adapter_predictions.jsonl")
DEFAULT_SCP_METADATA = Path("outputs/scp_code_comparison_smollm_vs_llama_with_descriptions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/ecg_token_ablation_analysis")


# Function: Compute binary metrics from prediction rows.
# Inputs: prediction rows containing true_label and pred_label.
# Outputs: dictionary of accuracy, balanced accuracy, recall, precision, F1, and confusion counts.
def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    return compute_binary_metrics(
        tn=counts[(0, 0)],
        fp=counts[(0, 1)],
        fn=counts[(1, 0)],
        tp=counts[(1, 1)],
    )


# Function: Split ablation prediction rows by ablation mode.
# Inputs: prediction rows from the ablation evaluator.
# Outputs: dictionary keyed by ablation mode.
def group_by_mode(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ablation_mode"])].append(row)
    return dict(grouped)


# Function: Summarize ablation performance by SCP code or clinical group.
# Inputs: prediction rows, grouping field name, SCP metadata, and whether to use clinical grouping.
# Outputs: nested summary dictionary keyed by group and ablation mode.
def summarize_grouped(
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
    use_clinical_group: bool,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        code = str(row["target_scp_code"])
        if use_clinical_group:
            group = clinical_group(metadata.get(code, {}))
        else:
            group = code
        mode = str(row["ablation_mode"])
        grouped[group][mode].append(row)

    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group, mode_rows in grouped.items():
        summaries[group] = {}
        for mode, group_mode_rows in mode_rows.items():
            summaries[group][mode] = compute_metrics(group_mode_rows)
    return summaries


# Function: Convert nested ablation summaries into a flat comparison table.
# Inputs: grouped summary, key column name, optional SCP metadata, and baseline mode.
# Outputs: list of table rows with real, ablated, and delta metrics.
def make_delta_rows(
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    key_name: str,
    metadata: Dict[str, Dict[str, str]] | None,
    baseline_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, by_mode in sorted(summary.items()):
        if baseline_mode not in by_mode:
            continue
        baseline = by_mode[baseline_mode]
        for mode, metrics in sorted(by_mode.items()):
            if mode == baseline_mode:
                continue
            row: Dict[str, Any] = {
                key_name: group,
                "ablation_mode": mode,
                "n": metrics["n"],
                "real_balanced_accuracy": baseline["balanced_accuracy"],
                "ablation_balanced_accuracy": metrics["balanced_accuracy"],
                "delta_balanced_accuracy": metrics["balanced_accuracy"] - baseline["balanced_accuracy"],
                "real_accuracy": baseline["accuracy"],
                "ablation_accuracy": metrics["accuracy"],
                "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
                "real_yes_recall": baseline["yes_recall"],
                "ablation_yes_recall": metrics["yes_recall"],
                "delta_yes_recall": metrics["yes_recall"] - baseline["yes_recall"],
                "real_no_recall": baseline["no_recall"],
                "ablation_no_recall": metrics["no_recall"],
                "delta_no_recall": metrics["no_recall"] - baseline["no_recall"],
            }
            if metadata and key_name == "scp_code":
                meta = metadata.get(group, {})
                row["description"] = meta.get("description", group)
                row["clinical_group"] = clinical_group(meta)
            rows.append(row)
    return rows


# Function: Print high-signal ablation findings to the terminal.
# Inputs: overall metrics and per-code delta rows.
# Outputs: None; prints concise summary.
def print_summary(overall: Dict[str, Any], code_rows: List[Dict[str, Any]]) -> None:
    print("Overall ablation metrics")
    for mode, metrics in overall["metrics_by_mode"].items():
        print(
            f"{mode:8s} bal={metrics['balanced_accuracy']:.3f} "
            f"acc={metrics['accuracy']:.3f} "
            f"yes_rec={metrics['yes_recall']:.3f} "
            f"no_rec={metrics['no_recall']:.3f} "
            f"pred_yes={metrics['pred_yes_count']}"
        )

    shuffled = [
        row for row in code_rows
        if row["ablation_mode"] == "shuffled" and row["n"] >= 50
    ]
    print("\nLargest performance drops under shuffled ECG tokens")
    for row in sorted(shuffled, key=lambda item: item["delta_balanced_accuracy"])[:12]:
        print(
            f"{row['scp_code']:8s} n={row['n']:>4} "
            f"real={row['real_balanced_accuracy']:.3f} "
            f"shuffled={row['ablation_balanced_accuracy']:.3f} "
            f"delta={row['delta_balanced_accuracy']:+.3f} "
            f"{row.get('description', '')}"
        )


# Function: Run ECG-token ablation analysis and write reusable CSV/JSON outputs.
# Inputs: command-line arguments.
# Outputs: JSON and CSV analysis artifacts.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--scp-metadata-csv", type=Path, default=DEFAULT_SCP_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-mode", type=str, default="real")
    args = parser.parse_args()

    with args.results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)

    by_mode = group_by_mode(predictions)
    overall_metrics = {mode: compute_metrics(rows) for mode, rows in sorted(by_mode.items())}
    code_summary = summarize_grouped(predictions, metadata=metadata, use_clinical_group=False)
    clinical_summary = summarize_grouped(predictions, metadata=metadata, use_clinical_group=True)

    code_rows = make_delta_rows(
        code_summary,
        key_name="scp_code",
        metadata=metadata,
        baseline_mode=args.baseline_mode,
    )
    clinical_rows = make_delta_rows(
        clinical_summary,
        key_name="clinical_group",
        metadata=None,
        baseline_mode=args.baseline_mode,
    )

    summary = {
        "source_results_path": str(args.results_path),
        "source_predictions_path": str(args.predictions_path),
        "baseline_mode": args.baseline_mode,
        "n_prediction_rows": len(predictions),
        "n_validation_examples_per_mode": {
            mode: len(rows) for mode, rows in sorted(by_mode.items())
        },
        "metrics_by_mode": overall_metrics,
        "original_results_metrics_by_mode": results.get("metrics_by_mode", {}),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ecg_token_ablation_summary.json", summary)
    write_csv(args.output_dir / "ecg_token_ablation_by_scp_code.csv", code_rows)
    write_csv(args.output_dir / "ecg_token_ablation_by_clinical_group.csv", clinical_rows)
    print_summary(summary, code_rows)
    print(f"\nSaved summary: {args.output_dir / 'ecg_token_ablation_summary.json'}")
    print(f"Saved SCP-code table: {args.output_dir / 'ecg_token_ablation_by_scp_code.csv'}")
    print(f"Saved clinical-group table: {args.output_dir / 'ecg_token_ablation_by_clinical_group.csv'}")


if __name__ == "__main__":
    main()
