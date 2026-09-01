from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


# Function: Load a JSON file from disk.
# Inputs: path to a JSON file.
# Outputs: parsed JSON object.
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Write rows to a CSV file.
# Inputs: output path and list of row dictionaries.
# Outputs: None; writes a CSV file.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Write a JSON object to disk.
# Inputs: output path and JSON-serializable object.
# Outputs: None; writes a JSON file.
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Load SCP descriptions and group labels from an existing per-code CSV.
# Inputs: CSV path containing code, description, and clinical_group fields.
# Outputs: dictionary keyed by SCP code.
def load_code_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"SCP metadata source not found: {path}")

    metadata: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            code = str(row.get("code", "")).strip()
            if not code:
                continue
            metadata[code] = {
                "description": str(row.get("description") or code),
                "clinical_group": str(row.get("clinical_group") or "uncategorised"),
                "diagnostic_class": str(row.get("diagnostic_class") or ""),
                "diagnostic_subclass": str(row.get("diagnostic_subclass") or ""),
            }
    return metadata


# Function: Convert a possibly string-keyed dictionary into integer counts.
# Inputs: dictionary loaded from JSON.
# Outputs: dictionary mapping integer labels to integer counts.
def int_count_dict(counts: Dict[str, Any]) -> Dict[int, int]:
    return {int(key): int(value) for key, value in counts.items()}


# Function: Recover yes/no recall from per-code aggregate metrics.
# Inputs: per-code metric dictionary with accuracy, balanced accuracy, and label counts.
# Outputs: tuple of no recall and yes recall.
def recover_recalls(metric: Dict[str, Any]) -> tuple[float, float]:
    label_counts = int_count_dict(metric["label_counts"])
    no_count = int(label_counts.get(0, 0))
    yes_count = int(label_counts.get(1, 0))
    n = no_count + yes_count
    accuracy = float(metric["accuracy"])
    balanced_accuracy = float(metric["balanced_accuracy"])

    if no_count == 0 or yes_count == 0:
        raise ValueError("Both labels are required to recover binary recalls.")

    if no_count == yes_count:
        # With equal class counts, accuracy and balanced accuracy collapse to the same equation.
        # The current ECG-QA SCP validation subset uses unequal no/yes counts for the relevant codes.
        raise ValueError(
            "Cannot uniquely recover per-class recalls when no_count equals yes_count."
        )

    yes_recall = ((accuracy * n) - (2.0 * balanced_accuracy * no_count)) / (
        yes_count - no_count
    )
    no_recall = (2.0 * balanced_accuracy) - yes_recall
    return max(0.0, min(1.0, no_recall)), max(0.0, min(1.0, yes_recall))


# Function: Reconstruct approximate confusion counts from stored aggregate metrics.
# Inputs: per-code metric dictionary.
# Outputs: dictionary with class counts, prediction counts, and approximate confusion counts.
def reconstruct_counts(metric: Dict[str, Any]) -> Dict[str, Any]:
    label_counts = int_count_dict(metric["label_counts"])
    pred_counts = int_count_dict(metric["prediction_counts"])
    no_count = int(label_counts.get(0, 0))
    yes_count = int(label_counts.get(1, 0))
    no_recall, yes_recall = recover_recalls(metric)

    true_negative = int(round(no_recall * no_count))
    true_positive = int(round(yes_recall * yes_count))
    false_positive = no_count - true_negative
    false_negative = yes_count - true_positive

    return {
        "n": int(metric["n"]),
        "no_count": no_count,
        "yes_count": yes_count,
        "pred_no_count": int(pred_counts.get(0, 0)),
        "pred_yes_count": int(pred_counts.get(1, 0)),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "no_recall": no_recall,
        "yes_recall": yes_recall,
    }


# Function: Compute binary metrics from reconstructed confusion counts.
# Inputs: counts for true negatives, false positives, false negatives, and true positives.
# Outputs: dictionary with accuracy, balanced accuracy, macro-F1, and recalls.
def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    n = tn + fp + fn + tp
    no_recall = tn / (tn + fp) if (tn + fp) else 0.0
    yes_recall = tp / (tp + fn) if (tp + fn) else 0.0
    no_precision = tn / (tn + fn) if (tn + fn) else 0.0
    yes_precision = tp / (tp + fp) if (tp + fp) else 0.0
    no_f1 = (
        2 * no_precision * no_recall / (no_precision + no_recall)
        if no_precision + no_recall
        else 0.0
    )
    yes_f1 = (
        2 * yes_precision * yes_recall / (yes_precision + yes_recall)
        if yes_precision + yes_recall
        else 0.0
    )
    return {
        "accuracy": (tn + tp) / n if n else 0.0,
        "balanced_accuracy": (no_recall + yes_recall) / 2.0,
        "macro_f1": (no_f1 + yes_f1) / 2.0,
        "no_recall": no_recall,
        "yes_recall": yes_recall,
    }


# Function: Build per-SCP-code comparison rows between two adapter result files.
# Inputs: baseline result JSON, comparison result JSON, and SCP metadata.
# Outputs: list of per-code comparison rows.
def compare_per_code(
    baseline: Dict[str, Any],
    comparison: Dict[str, Any],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    baseline_codes = baseline["metrics"]["per_code"]
    comparison_codes = comparison["metrics"]["per_code"]
    common_codes = sorted(set(baseline_codes) & set(comparison_codes))

    rows: List[Dict[str, Any]] = []
    for code in common_codes:
        base_metric = baseline_codes[code]
        comp_metric = comparison_codes[code]
        base_counts = reconstruct_counts(base_metric)
        comp_counts = reconstruct_counts(comp_metric)
        meta = metadata.get(code, {})

        rows.append(
            {
                "code": code,
                "description": meta.get("description", code),
                "clinical_group": meta.get("clinical_group", "uncategorised"),
                "diagnostic_class": meta.get("diagnostic_class", ""),
                "diagnostic_subclass": meta.get("diagnostic_subclass", ""),
                "n": int(base_metric["n"]),
                "yes_count": base_counts["yes_count"],
                "no_count": base_counts["no_count"],
                "baseline_balanced_accuracy": float(base_metric["balanced_accuracy"]),
                "comparison_balanced_accuracy": float(comp_metric["balanced_accuracy"]),
                "delta_balanced_accuracy": float(comp_metric["balanced_accuracy"])
                - float(base_metric["balanced_accuracy"]),
                "baseline_accuracy": float(base_metric["accuracy"]),
                "comparison_accuracy": float(comp_metric["accuracy"]),
                "delta_accuracy": float(comp_metric["accuracy"]) - float(base_metric["accuracy"]),
                "baseline_macro_f1": float(base_metric["macro_f1"]),
                "comparison_macro_f1": float(comp_metric["macro_f1"]),
                "delta_macro_f1": float(comp_metric["macro_f1"]) - float(base_metric["macro_f1"]),
                "baseline_yes_recall": base_counts["yes_recall"],
                "comparison_yes_recall": comp_counts["yes_recall"],
                "delta_yes_recall": comp_counts["yes_recall"] - base_counts["yes_recall"],
                "baseline_no_recall": base_counts["no_recall"],
                "comparison_no_recall": comp_counts["no_recall"],
                "delta_no_recall": comp_counts["no_recall"] - base_counts["no_recall"],
                "baseline_pred_yes_count": base_counts["pred_yes_count"],
                "comparison_pred_yes_count": comp_counts["pred_yes_count"],
                "delta_pred_yes_count": comp_counts["pred_yes_count"]
                - base_counts["pred_yes_count"],
                "baseline_true_positive": base_counts["true_positive"],
                "comparison_true_positive": comp_counts["true_positive"],
                "baseline_false_positive": base_counts["false_positive"],
                "comparison_false_positive": comp_counts["false_positive"],
            }
        )
    return rows


# Function: Aggregate reconstructed per-code comparison rows by clinical group.
# Inputs: per-code comparison rows.
# Outputs: list of clinical-group comparison rows.
def summarize_by_clinical_group(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clinical_group"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        base_tn = sum(round(row["baseline_no_recall"] * row["no_count"]) for row in group_rows)
        base_tp = sum(round(row["baseline_yes_recall"] * row["yes_count"]) for row in group_rows)
        comp_tn = sum(round(row["comparison_no_recall"] * row["no_count"]) for row in group_rows)
        comp_tp = sum(round(row["comparison_yes_recall"] * row["yes_count"]) for row in group_rows)
        no_count = sum(int(row["no_count"]) for row in group_rows)
        yes_count = sum(int(row["yes_count"]) for row in group_rows)
        base_fp = no_count - base_tn
        base_fn = yes_count - base_tp
        comp_fp = no_count - comp_tn
        comp_fn = yes_count - comp_tp
        base_metrics = metrics_from_counts(base_tn, base_fp, base_fn, base_tp)
        comp_metrics = metrics_from_counts(comp_tn, comp_fp, comp_fn, comp_tp)

        summaries.append(
            {
                "clinical_group": group,
                "num_codes": len(group_rows),
                "n": sum(int(row["n"]) for row in group_rows),
                "yes_count": yes_count,
                "no_count": no_count,
                "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
                "comparison_balanced_accuracy": comp_metrics["balanced_accuracy"],
                "delta_balanced_accuracy": comp_metrics["balanced_accuracy"]
                - base_metrics["balanced_accuracy"],
                "baseline_accuracy": base_metrics["accuracy"],
                "comparison_accuracy": comp_metrics["accuracy"],
                "delta_accuracy": comp_metrics["accuracy"] - base_metrics["accuracy"],
                "baseline_macro_f1": base_metrics["macro_f1"],
                "comparison_macro_f1": comp_metrics["macro_f1"],
                "delta_macro_f1": comp_metrics["macro_f1"] - base_metrics["macro_f1"],
                "baseline_yes_recall": base_metrics["yes_recall"],
                "comparison_yes_recall": comp_metrics["yes_recall"],
                "delta_yes_recall": comp_metrics["yes_recall"] - base_metrics["yes_recall"],
                "baseline_no_recall": base_metrics["no_recall"],
                "comparison_no_recall": comp_metrics["no_recall"],
                "delta_no_recall": comp_metrics["no_recall"] - base_metrics["no_recall"],
            }
        )
    return summaries


# Function: Run an adapter-vs-adapter SCP-code comparison.
# Inputs: command-line arguments.
# Outputs: per-code and clinical-group comparison tables.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--comparison-results", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline_results)
    comparison = load_json(args.comparison_results)
    metadata = load_code_metadata(args.metadata_csv)

    per_code_rows = compare_per_code(baseline, comparison, metadata)
    group_rows = summarize_by_clinical_group(per_code_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_scp_code_comparison.csv", per_code_rows)
    write_json(args.output_dir / "per_scp_code_comparison.json", per_code_rows)
    write_csv(args.output_dir / "clinical_group_comparison.csv", group_rows)
    write_json(args.output_dir / "clinical_group_comparison.json", group_rows)

    print("Overall baseline:", baseline["metrics"]["balanced_accuracy"])
    print("Overall comparison:", comparison["metrics"]["balanced_accuracy"])
    print("\nClinical groups by delta balanced accuracy:")
    for row in sorted(group_rows, key=lambda item: item["delta_balanced_accuracy"], reverse=True):
        print(
            f"{row['clinical_group']:30s} "
            f"delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"base={row['baseline_balanced_accuracy']:.3f} "
            f"comp={row['comparison_balanced_accuracy']:.3f} "
            f"yes_delta={row['delta_yes_recall']:+.3f} "
            f"no_delta={row['delta_no_recall']:+.3f} "
            f"n={row['n']}"
        )

    print("\nTop per-code gains:")
    for row in sorted(per_code_rows, key=lambda item: item["delta_balanced_accuracy"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"base={row['baseline_balanced_accuracy']:.3f} "
            f"comp={row['comparison_balanced_accuracy']:.3f} "
            f"yes_delta={row['delta_yes_recall']:+.3f} "
            f"no_delta={row['delta_no_recall']:+.3f} "
            f"{row['description']}"
        )

    print("\nTop per-code drops:")
    for row in sorted(per_code_rows, key=lambda item: item["delta_balanced_accuracy"])[:12]:
        print(
            f"{row['code']:8s} delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"base={row['baseline_balanced_accuracy']:.3f} "
            f"comp={row['comparison_balanced_accuracy']:.3f} "
            f"yes_delta={row['delta_yes_recall']:+.3f} "
            f"no_delta={row['delta_no_recall']:+.3f} "
            f"{row['description']}"
        )

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
