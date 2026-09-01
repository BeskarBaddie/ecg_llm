from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analyze_lead_specific_performance import classify_question_localization
from analyze_scp_code_performance import (
    clinical_group,
    compute_binary_metrics,
    load_jsonl,
    load_scp_metadata,
    write_csv,
    write_json,
)


# Function: Compute Pearson correlation for paired numeric values.
# Inputs: x and y numeric arrays with equal length.
# Outputs: Pearson r, or None when correlation is undefined.
def pearson_correlation(x_values: List[float], y_values: List[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denom == 0 or y_denom == 0:
        return None
    return numerator / (x_denom * y_denom)


# Function: Convert values to average ranks for Spearman correlation.
# Inputs: numeric values.
# Outputs: rank values where tied observations receive their average rank.
def average_ranks(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        start = cursor
        value = indexed[cursor][1]
        while cursor < len(indexed) and indexed[cursor][1] == value:
            cursor += 1
        end = cursor
        average_rank = (start + 1 + end) / 2.0
        for original_index, _ in indexed[start:end]:
            ranks[original_index] = average_rank
    return ranks


# Function: Compute Spearman rank correlation for paired numeric values.
# Inputs: x and y numeric arrays with equal length.
# Outputs: Spearman rho, or None when correlation is undefined.
def spearman_correlation(x_values: List[float], y_values: List[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return pearson_correlation(average_ranks(x_values), average_ranks(y_values))


# Function: Compute metrics for a possibly empty subset of prediction rows.
# Inputs: prediction rows.
# Outputs: metric dictionary, or empty metrics when no rows exist.
def metrics_for_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "balanced_accuracy": None,
            "yes_recall": None,
            "no_recall": None,
        }
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    return compute_binary_metrics(
        tn=counts[(0, 0)],
        fp=counts[(0, 1)],
        fn=counts[(1, 0)],
        tp=counts[(1, 1)],
    )


# Function: Summarize localization burden and performance for each SCP code.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: per-code rows with localization fractions and whole/localized metrics.
def summarize_code_localization_burden(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("target_scp_code", "UNKNOWN"))].append(row)

    output_rows: List[Dict[str, Any]] = []
    for code, rows in sorted(grouped.items()):
        meta = metadata.get(code, {"description": code})
        localized_rows = [
            row for row in rows
            if classify_question_localization(str(row.get("question", ""))) != "whole_ecg"
        ]
        explicit_rows = [
            row for row in rows
            if classify_question_localization(str(row.get("question", ""))) == "explicit_lead"
        ]
        regional_rows = [
            row for row in rows
            if classify_question_localization(str(row.get("question", ""))) == "regional_leads"
        ]
        whole_rows = [
            row for row in rows
            if classify_question_localization(str(row.get("question", ""))) == "whole_ecg"
        ]

        overall_metrics = metrics_for_subset(rows)
        whole_metrics = metrics_for_subset(whole_rows)
        localized_metrics = metrics_for_subset(localized_rows)

        whole_bal_acc = whole_metrics.get("balanced_accuracy")
        localized_bal_acc = localized_metrics.get("balanced_accuracy")
        within_code_localization_delta = (
            localized_bal_acc - whole_bal_acc
            if localized_bal_acc is not None and whole_bal_acc is not None
            else None
        )

        output_rows.append(
            {
                "code": code,
                "description": meta.get("description", code),
                "clinical_group": clinical_group(meta),
                "n": len(rows),
                "whole_ecg_n": len(whole_rows),
                "localized_any_n": len(localized_rows),
                "explicit_lead_n": len(explicit_rows),
                "regional_leads_n": len(regional_rows),
                "localized_fraction": len(localized_rows) / len(rows) if rows else None,
                "explicit_lead_fraction": len(explicit_rows) / len(rows) if rows else None,
                "regional_leads_fraction": len(regional_rows) / len(rows) if rows else None,
                "overall_balanced_accuracy": overall_metrics.get("balanced_accuracy"),
                "overall_yes_recall": overall_metrics.get("yes_recall"),
                "overall_no_recall": overall_metrics.get("no_recall"),
                "whole_ecg_balanced_accuracy": whole_bal_acc,
                "whole_ecg_yes_recall": whole_metrics.get("yes_recall"),
                "whole_ecg_no_recall": whole_metrics.get("no_recall"),
                "localized_balanced_accuracy": localized_bal_acc,
                "localized_yes_recall": localized_metrics.get("yes_recall"),
                "localized_no_recall": localized_metrics.get("no_recall"),
                "localized_minus_whole_balanced_accuracy": within_code_localization_delta,
            }
        )

    return output_rows


# Function: Compute correlation summary between localization burden and SCP performance.
# Inputs: per-code rows and minimum support.
# Outputs: correlation summary dictionary.
def summarize_correlations(rows: List[Dict[str, Any]], min_support: int) -> Dict[str, Any]:
    eligible = [
        row for row in rows
        if row["n"] >= min_support
        and row["localized_fraction"] is not None
        and row["overall_balanced_accuracy"] is not None
    ]
    localized_fraction = [float(row["localized_fraction"]) for row in eligible]
    explicit_fraction = [float(row["explicit_lead_fraction"]) for row in eligible]
    balanced_accuracy = [float(row["overall_balanced_accuracy"]) for row in eligible]

    within_code_rows = [
        row for row in eligible
        if row["whole_ecg_n"] > 0
        and row["localized_any_n"] > 0
        and row["localized_minus_whole_balanced_accuracy"] is not None
    ]

    return {
        "min_support": min_support,
        "num_codes": len(rows),
        "num_codes_with_support": len(eligible),
        "pearson_localized_fraction_vs_balanced_accuracy": pearson_correlation(
            localized_fraction,
            balanced_accuracy,
        ),
        "spearman_localized_fraction_vs_balanced_accuracy": spearman_correlation(
            localized_fraction,
            balanced_accuracy,
        ),
        "pearson_explicit_lead_fraction_vs_balanced_accuracy": pearson_correlation(
            explicit_fraction,
            balanced_accuracy,
        ),
        "spearman_explicit_lead_fraction_vs_balanced_accuracy": spearman_correlation(
            explicit_fraction,
            balanced_accuracy,
        ),
        "num_codes_with_both_whole_and_localized": len(within_code_rows),
        "mean_localized_minus_whole_balanced_accuracy": (
            sum(float(row["localized_minus_whole_balanced_accuracy"]) for row in within_code_rows)
            / len(within_code_rows)
            if within_code_rows else None
        ),
    }


# Function: Print concise SCP localization burden results.
# Inputs: per-code rows, correlation summary, and support threshold.
# Outputs: None; prints human-readable findings.
def print_summary(
    rows: List[Dict[str, Any]],
    correlations: Dict[str, Any],
    min_support: int,
) -> None:
    eligible = [row for row in rows if row["n"] >= min_support]
    print("Localization burden correlation summary")
    print(
        "localized_fraction vs balanced_accuracy: "
        f"Pearson={correlations['pearson_localized_fraction_vs_balanced_accuracy']:.3f} "
        f"Spearman={correlations['spearman_localized_fraction_vs_balanced_accuracy']:.3f}"
    )
    print(
        "explicit_lead_fraction vs balanced_accuracy: "
        f"Pearson={correlations['pearson_explicit_lead_fraction_vs_balanced_accuracy']:.3f} "
        f"Spearman={correlations['spearman_explicit_lead_fraction_vs_balanced_accuracy']:.3f}"
    )
    print(
        "Mean localized-minus-whole balanced accuracy among codes with both question types: "
        f"{correlations['mean_localized_minus_whole_balanced_accuracy']:.3f}"
    )

    print(f"\nMost localized SCP codes with n >= {min_support}")
    for row in sorted(eligible, key=lambda item: item["localized_fraction"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} "
            f"loc_frac={row['localized_fraction']:.3f} "
            f"bal_acc={row['overall_balanced_accuracy']:.3f} "
            f"whole_n={row['whole_ecg_n']:>3} "
            f"loc_n={row['localized_any_n']:>3} "
            f"{row['description']}"
        )

    print(f"\nWeak SCP codes with high localization burden, n >= {min_support}")
    weak_localized = [
        row for row in eligible
        if row["localized_fraction"] >= 0.5 and row["overall_balanced_accuracy"] <= 0.70
    ]
    for row in sorted(weak_localized, key=lambda item: item["overall_balanced_accuracy"])[:12]:
        print(
            f"{row['code']:8s} "
            f"bal_acc={row['overall_balanced_accuracy']:.3f} "
            f"loc_frac={row['localized_fraction']:.3f} "
            f"localized_yes_rec={row['localized_yes_recall']} "
            f"{row['description']}"
        )


# Function: Run SCP-code localization burden analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON files with per-code localization burden and correlation summary.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-code-csv", type=Path, required=True)
    parser.add_argument("--output-code-json", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)
    rows = summarize_code_localization_burden(predictions, metadata)
    correlations = summarize_correlations(rows, min_support=args.min_support)

    write_csv(args.output_code_csv, rows)
    write_json(args.output_code_json, rows)
    write_json(args.output_summary_json, correlations)
    print_summary(rows, correlations, min_support=args.min_support)
    print(f"\nSaved code CSV: {args.output_code_csv}")
    print(f"Saved summary JSON: {args.output_summary_json}")


if __name__ == "__main__":
    main()
