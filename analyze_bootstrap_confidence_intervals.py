from __future__ import annotations

import argparse
import random
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
from analyze_scp_group_performance import describe_group_codes


METRICS_FOR_INTERVALS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "yes_recall",
    "no_recall",
)


# Function: Compute a percentile from a sorted list of numeric values.
# Inputs: sorted values and percentile between 0 and 100.
# Outputs: interpolated percentile value, or None for empty inputs.
def percentile(sorted_values: List[float], percentile_value: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


# Function: Compute binary classification metrics for prediction rows.
# Inputs: prediction rows containing true_label and pred_label.
# Outputs: metric dictionary from confusion counts.
def metrics_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    return compute_binary_metrics(
        tn=counts[(0, 0)],
        fp=counts[(0, 1)],
        fn=counts[(1, 0)],
        tp=counts[(1, 1)],
    )


# Function: Group rows by ECG ID for clustered bootstrap resampling.
# Inputs: prediction rows.
# Outputs: list where each element contains all rows for one ECG.
def rows_by_ecg(predictions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("ecg_id", "UNKNOWN"))].append(row)
    return list(grouped.values())


# Function: Estimate bootstrap confidence intervals for one set of prediction rows.
# Inputs: rows, number of resamples, RNG seed, interval size, and resampling unit.
# Outputs: metric point estimates plus lower/upper confidence interval columns.
def bootstrap_metric_intervals(
    rows: List[Dict[str, Any]],
    num_bootstrap: int,
    seed: int,
    confidence: float,
    bootstrap_unit: str,
) -> Dict[str, Any]:
    point = metrics_for_rows(rows)
    rng = random.Random(seed)

    if bootstrap_unit == "ecg":
        units = rows_by_ecg(rows)
    elif bootstrap_unit == "row":
        units = [[row] for row in rows]
    else:
        raise ValueError(f"Unsupported bootstrap unit: {bootstrap_unit}")

    metric_samples: Dict[str, List[float]] = {name: [] for name in METRICS_FOR_INTERVALS}
    for _ in range(num_bootstrap):
        sampled_rows: List[Dict[str, Any]] = []
        for _unit_idx in range(len(units)):
            sampled_rows.extend(rng.choice(units))
        sample_metrics = metrics_for_rows(sampled_rows)
        for metric_name in METRICS_FOR_INTERVALS:
            value = sample_metrics.get(metric_name)
            if value is not None:
                metric_samples[metric_name].append(float(value))

    alpha = (1.0 - confidence) / 2.0
    interval_row = {
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_resamples": num_bootstrap,
        "confidence": confidence,
        "num_bootstrap_units": len(units),
    }
    interval_row.update(point)

    for metric_name, values in metric_samples.items():
        sorted_values = sorted(values)
        interval_row[f"{metric_name}_ci_low"] = percentile(sorted_values, alpha * 100.0)
        interval_row[f"{metric_name}_ci_high"] = percentile(sorted_values, (1.0 - alpha) * 100.0)

    return interval_row


# Function: Build bootstrap rows for overall, clinical-group, or SCP-code analysis.
# Inputs: predictions, metadata, grouping level, bootstrap settings, and minimum support.
# Outputs: list of rows with point estimates and confidence intervals.
def build_bootstrap_rows(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
    level: str,
    num_bootstrap: int,
    seed: int,
    confidence: float,
    bootstrap_unit: str,
    min_support: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if level == "overall":
        grouped["overall"] = predictions
    elif level == "clinical_group":
        for row in predictions:
            code = str(row.get("target_scp_code", "UNKNOWN"))
            grouped[clinical_group(metadata.get(code, {"description": code}))].append(row)
    elif level == "scp_code":
        for row in predictions:
            grouped[str(row.get("target_scp_code", "UNKNOWN"))].append(row)
    else:
        raise ValueError(f"Unsupported bootstrap level: {level}")

    output_rows: List[Dict[str, Any]] = []
    for idx, (group_name, rows) in enumerate(sorted(grouped.items())):
        if len(rows) < min_support:
            continue

        summary = bootstrap_metric_intervals(
            rows=rows,
            num_bootstrap=num_bootstrap,
            seed=seed + idx,
            confidence=confidence,
            bootstrap_unit=bootstrap_unit,
        )

        if level == "scp_code":
            meta = metadata.get(group_name, {"description": group_name})
            prefix = {
                "level": level,
                "code": group_name,
                "description": meta.get("description", group_name),
                "clinical_group": clinical_group(meta),
            }
        elif level == "clinical_group":
            prefix = {
                "level": level,
                "clinical_group": group_name,
                "num_scp_codes": len({str(row.get("target_scp_code", "UNKNOWN")) for row in rows}),
                "scp_code_descriptions": describe_group_codes(rows, metadata),
            }
        else:
            prefix = {"level": level, "group": group_name}

        prefix.update(summary)
        output_rows.append(prefix)

    return output_rows


# Function: Print a compact confidence interval summary.
# Inputs: overall, clinical-group, and code bootstrap rows.
# Outputs: None; prints human-readable CI summary.
def print_summary(
    overall_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    code_rows: List[Dict[str, Any]],
) -> None:
    print("Overall bootstrap confidence interval")
    for row in overall_rows:
        print(
            f"balanced_accuracy={row['balanced_accuracy']:.3f} "
            f"95%CI=[{row['balanced_accuracy_ci_low']:.3f}, {row['balanced_accuracy_ci_high']:.3f}] "
            f"yes_recall={row['yes_recall']:.3f} "
            f"95%CI=[{row['yes_recall_ci_low']:.3f}, {row['yes_recall_ci_high']:.3f}]"
        )

    print("\nClinical groups by balanced accuracy CI")
    for row in sorted(group_rows, key=lambda item: item["balanced_accuracy"], reverse=True):
        print(
            f"{row['clinical_group']:30s} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"95%CI=[{row['balanced_accuracy_ci_low']:.3f}, {row['balanced_accuracy_ci_high']:.3f}] "
            f"n={row['n']:>4}"
        )

    print("\nMost uncertain SCP-code estimates by balanced accuracy CI width")
    code_rows_with_width = [
        {
            **row,
            "ci_width": row["balanced_accuracy_ci_high"] - row["balanced_accuracy_ci_low"],
        }
        for row in code_rows
        if row.get("balanced_accuracy_ci_low") is not None
        and row.get("balanced_accuracy_ci_high") is not None
    ]
    for row in sorted(code_rows_with_width, key=lambda item: item["ci_width"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"95%CI=[{row['balanced_accuracy_ci_low']:.3f}, {row['balanced_accuracy_ci_high']:.3f}] "
            f"width={row['ci_width']:.3f} "
            f"n={row['n']:>3} "
            f"{row['description']}"
        )


# Function: Run bootstrap confidence interval analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON files with overall, clinical-group, and SCP-code confidence intervals.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-overall-csv", type=Path, required=True)
    parser.add_argument("--output-overall-json", type=Path, required=True)
    parser.add_argument("--output-group-csv", type=Path, required=True)
    parser.add_argument("--output-group-json", type=Path, required=True)
    parser.add_argument("--output-code-csv", type=Path, required=True)
    parser.add_argument("--output-code-json", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-unit", choices=("ecg", "row"), default="ecg")
    parser.add_argument("--min-code-support", type=int, default=50)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)

    overall_rows = build_bootstrap_rows(
        predictions=predictions,
        metadata=metadata,
        level="overall",
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        confidence=args.confidence,
        bootstrap_unit=args.bootstrap_unit,
        min_support=1,
    )
    group_rows = build_bootstrap_rows(
        predictions=predictions,
        metadata=metadata,
        level="clinical_group",
        num_bootstrap=args.num_bootstrap,
        seed=args.seed + 10_000,
        confidence=args.confidence,
        bootstrap_unit=args.bootstrap_unit,
        min_support=1,
    )
    code_rows = build_bootstrap_rows(
        predictions=predictions,
        metadata=metadata,
        level="scp_code",
        num_bootstrap=args.num_bootstrap,
        seed=args.seed + 20_000,
        confidence=args.confidence,
        bootstrap_unit=args.bootstrap_unit,
        min_support=args.min_code_support,
    )

    write_csv(args.output_overall_csv, overall_rows)
    write_json(args.output_overall_json, overall_rows)
    write_csv(args.output_group_csv, group_rows)
    write_json(args.output_group_json, group_rows)
    write_csv(args.output_code_csv, code_rows)
    write_json(args.output_code_json, code_rows)
    print_summary(overall_rows=overall_rows, group_rows=group_rows, code_rows=code_rows)
    print(f"\nSaved overall CSV: {args.output_overall_csv}")
    print(f"Saved clinical-group CSV: {args.output_group_csv}")
    print(f"Saved SCP-code CSV: {args.output_code_csv}")


if __name__ == "__main__":
    main()
