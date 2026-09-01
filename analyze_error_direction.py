from __future__ import annotations

import argparse
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


# Function: Add false-negative and false-positive direction metrics to a metric row.
# Inputs: a metric row containing yes/no recall and false positive/negative counts.
# Outputs: the same row updated with miss rates and an error direction label.
def add_error_direction(row: Dict[str, Any]) -> Dict[str, Any]:
    yes_recall = row.get("yes_recall")
    no_recall = row.get("no_recall")
    false_negative_rate = None if yes_recall is None else 1.0 - float(yes_recall)
    false_positive_rate = None if no_recall is None else 1.0 - float(no_recall)

    if false_negative_rate is None or false_positive_rate is None:
        error_bias = None
        error_direction = "undefined"
    else:
        error_bias = false_negative_rate - false_positive_rate
        if error_bias >= 0.20:
            error_direction = "mainly_false_negative"
        elif error_bias <= -0.20:
            error_direction = "mainly_false_positive"
        else:
            error_direction = "mixed_or_balanced"

    row["false_negative_rate"] = false_negative_rate
    row["false_positive_rate"] = false_positive_rate
    row["error_bias_fn_minus_fp"] = error_bias
    row["error_direction"] = error_direction
    return row


# Function: Compute binary metrics and error direction for one set of rows.
# Inputs: prediction rows belonging to one group.
# Outputs: metric dictionary with confusion counts and error-direction columns.
def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    metrics = compute_binary_metrics(
        tn=counts[(0, 0)],
        fp=counts[(0, 1)],
        fn=counts[(1, 0)],
        tp=counts[(1, 1)],
    )
    return add_error_direction(metrics)


# Function: Summarize error direction for each SCP code.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: per-code rows with clinical metadata and error-direction metrics.
def summarize_code_errors(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("target_scp_code", "UNKNOWN"))].append(row)

    summaries: List[Dict[str, Any]] = []
    for code, rows in sorted(grouped.items()):
        meta = metadata.get(code, {"description": code})
        summary = {
            "code": code,
            "description": meta.get("description", code),
            "clinical_group": clinical_group(meta),
            "diagnostic_class": meta.get("diagnostic_class", ""),
            "diagnostic_subclass": meta.get("diagnostic_subclass", ""),
            "diagnostic": meta.get("diagnostic", ""),
            "form": meta.get("form", ""),
            "rhythm": meta.get("rhythm", ""),
        }
        summary.update(summarize_rows(rows))
        summaries.append(summary)

    return summaries


# Function: Summarize error direction for each broad clinical group.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: per-clinical-group rows with code coverage and error-direction metrics.
def summarize_group_errors(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        code = str(row.get("target_scp_code", "UNKNOWN"))
        grouped[clinical_group(metadata.get(code, {"description": code}))].append(row)

    summaries: List[Dict[str, Any]] = []
    for group, rows in sorted(grouped.items()):
        codes = sorted({str(row.get("target_scp_code", "UNKNOWN")) for row in rows})
        summary = {
            "clinical_group": group,
            "num_scp_codes": len(codes),
            "scp_codes": ",".join(codes),
            "scp_code_descriptions": describe_group_codes(rows, metadata),
        }
        summary.update(summarize_rows(rows))
        summaries.append(summary)

    return sorted(summaries, key=lambda row: row["error_bias_fn_minus_fp"], reverse=True)


# Function: Print the most clinically relevant error-direction findings.
# Inputs: per-code and per-group rows plus minimum support threshold.
# Outputs: None; prints false-negative-heavy and false-positive-heavy summaries.
def print_summary(
    code_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    min_support: int,
) -> None:
    print("Clinical-group error direction")
    for row in group_rows:
        print(
            f"{row['clinical_group']:30s} "
            f"n={row['n']:>4} "
            f"FN_rate={row['false_negative_rate']:.3f} "
            f"FP_rate={row['false_positive_rate']:.3f} "
            f"bias={row['error_bias_fn_minus_fp']:.3f} "
            f"{row['error_direction']}"
        )

    eligible = [row for row in code_rows if row["n"] >= min_support]
    print(f"\nFalse-negative-heavy SCP codes with n >= {min_support}")
    for row in sorted(eligible, key=lambda item: item["error_bias_fn_minus_fp"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"FN_rate={row['false_negative_rate']:.3f} "
            f"FP_rate={row['false_positive_rate']:.3f} "
            f"{row['description']}"
        )

    print(f"\nFalse-positive-heavy SCP codes with n >= {min_support}")
    for row in sorted(eligible, key=lambda item: item["error_bias_fn_minus_fp"])[:12]:
        print(
            f"{row['code']:8s} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"FN_rate={row['false_negative_rate']:.3f} "
            f"FP_rate={row['false_positive_rate']:.3f} "
            f"{row['description']}"
        )


# Function: Run false-negative versus false-positive error-direction analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON files for per-code and per-group error direction.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-code-csv", type=Path, required=True)
    parser.add_argument("--output-code-json", type=Path, required=True)
    parser.add_argument("--output-group-csv", type=Path, required=True)
    parser.add_argument("--output-group-json", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)
    code_rows = summarize_code_errors(predictions, metadata)
    group_rows = summarize_group_errors(predictions, metadata)

    write_csv(args.output_code_csv, code_rows)
    write_json(args.output_code_json, code_rows)
    write_csv(args.output_group_csv, group_rows)
    write_json(args.output_group_json, group_rows)
    print_summary(code_rows, group_rows, min_support=args.min_support)
    print(f"\nSaved code CSV: {args.output_code_csv}")
    print(f"Saved group CSV: {args.output_group_csv}")


if __name__ == "__main__":
    main()
