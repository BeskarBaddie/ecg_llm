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


# Function: Group prediction rows by broad clinical SCP class.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: dictionary mapping clinical group names to prediction rows.
def group_predictions_by_clinical_class(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        code = str(row.get("target_scp_code", "UNKNOWN"))
        group = clinical_group(metadata.get(code, {"description": code}))
        grouped[group].append(row)
    return dict(grouped)


# Function: Collect SCP-code descriptions represented inside one clinical group.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: compact string listing code and description pairs.
def describe_group_codes(
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> str:
    codes = sorted({str(row.get("target_scp_code", "UNKNOWN")) for row in rows})
    labels = []
    for code in codes:
        description = metadata.get(code, {}).get("description", code)
        labels.append(f"{code}: {description}")
    return "; ".join(labels)


# Function: Compute group-level metrics from all rows in one clinical group.
# Inputs: clinical group name, prediction rows, and SCP metadata.
# Outputs: dictionary containing support, code coverage, confusion counts, and metrics.
def summarize_group(
    group: str,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    metrics = compute_binary_metrics(tn=tn, fp=fp, fn=fn, tp=tp)
    codes = sorted({str(row.get("target_scp_code", "UNKNOWN")) for row in rows})

    return {
        "clinical_group": group,
        "num_scp_codes": len(codes),
        "scp_codes": ",".join(codes),
        "scp_code_descriptions": describe_group_codes(rows, metadata),
        **metrics,
    }


# Function: Compute group-level metrics for all clinical SCP classes.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: list of group metric dictionaries sorted by support.
def summarize_groups(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped = group_predictions_by_clinical_class(predictions, metadata)
    rows = [
        summarize_group(group=group, rows=group_rows, metadata=metadata)
        for group, group_rows in grouped.items()
    ]
    return sorted(rows, key=lambda row: row["n"], reverse=True)


# Function: Print a compact group-level performance summary.
# Inputs: group summary rows.
# Outputs: None; prints human-readable summary.
def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("Clinical groups:", len(rows))
    print(
        "group | n | codes | balanced_accuracy | yes_recall | no_recall | false_negative | false_positive"
    )
    for row in rows:
        print(
            f"{row['clinical_group']:30s} "
            f"n={row['n']:>4} "
            f"codes={row['num_scp_codes']:>2} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"yes_rec={row['yes_recall']:.3f} "
            f"no_rec={row['no_recall']:.3f} "
            f"FN={row['false_negative']:>3} "
            f"FP={row['false_positive']:>3}"
        )


# Function: Run clinical class-level SCP performance analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON group-level metric files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)
    rows = summarize_groups(predictions=predictions, metadata=metadata)

    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows)
    print_summary(rows)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
