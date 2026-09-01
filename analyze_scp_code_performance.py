from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Function: Load newline-delimited prediction rows from disk.
# Inputs: path to a JSONL predictions file.
# Outputs: list of parsed prediction dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return rows


# Function: Load SCP code descriptions and PTB-XL class metadata.
# Inputs: path to the enriched SCP comparison CSV.
# Outputs: dictionary keyed by SCP code with human-readable metadata.
def load_scp_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"SCP metadata CSV not found: {path}")

    metadata: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            code = str(row.get("code", "")).strip()
            if not code:
                continue
            metadata[code] = {
                "description": str(row.get("description") or code),
                "diagnostic_class": str(row.get("diagnostic_class") or ""),
                "diagnostic_subclass": str(row.get("diagnostic_subclass") or ""),
                "diagnostic": str(row.get("diagnostic") or ""),
                "form": str(row.get("form") or ""),
                "rhythm": str(row.get("rhythm") or ""),
            }
    return metadata


# Function: Assign one broad clinical group from PTB-XL SCP metadata.
# Inputs: metadata dictionary for one SCP code.
# Outputs: broad group label used for clinical interpretation.
def clinical_group(meta: Dict[str, str]) -> str:
    if meta.get("rhythm") == "1.0":
        return "rhythm"
    diagnostic_class = meta.get("diagnostic_class", "")
    if diagnostic_class == "CD":
        return "conduction disturbance"
    if diagnostic_class == "MI":
        return "myocardial infarction/injury"
    if diagnostic_class == "HYP":
        return "hypertrophy"
    if diagnostic_class == "STTC":
        return "ST/T change"
    if diagnostic_class == "NORM":
        return "normal"
    if meta.get("form") == "1.0":
        return "form/morphology"
    if diagnostic_class:
        return f"diagnostic:{diagnostic_class}"
    return "uncategorised"


# Function: Safely divide two numbers for metric calculation.
# Inputs: numerator and denominator.
# Outputs: quotient, or None when denominator is zero.
def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


# Function: Compute binary classification metrics from confusion counts.
# Inputs: true negative, false positive, false negative, and true positive counts.
# Outputs: dictionary containing accuracy, balanced accuracy, recalls, precision, F1, and counts.
def compute_binary_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, Any]:
    n = tn + fp + fn + tp
    no_recall = safe_divide(tn, tn + fp)
    yes_recall = safe_divide(tp, tp + fn)
    no_precision = safe_divide(tn, tn + fn)
    yes_precision = safe_divide(tp, tp + fp)
    no_f1 = (
        2 * no_precision * no_recall / (no_precision + no_recall)
        if no_precision is not None and no_recall is not None and no_precision + no_recall > 0
        else 0.0
    )
    yes_f1 = (
        2 * yes_precision * yes_recall / (yes_precision + yes_recall)
        if yes_precision is not None and yes_recall is not None and yes_precision + yes_recall > 0
        else 0.0
    )
    balanced_accuracy = (
        (no_recall + yes_recall) / 2
        if no_recall is not None and yes_recall is not None
        else None
    )

    return {
        "n": n,
        "yes_count": tp + fn,
        "no_count": tn + fp,
        "pred_yes_count": tp + fp,
        "pred_no_count": tn + fn,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": safe_divide(tn + tp, n),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": (no_f1 + yes_f1) / 2,
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "yes_precision": yes_precision,
        "no_precision": no_precision,
        "yes_f1": yes_f1,
        "no_f1": no_f1,
    }


# Function: Group prediction rows by target SCP code and compute per-code metrics.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: sorted list of per-SCP-code metric dictionaries.
def summarize_by_scp_code(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("target_scp_code", "UNKNOWN"))].append(row)

    summaries: List[Dict[str, Any]] = []
    for code, rows in sorted(grouped.items()):
        counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
        tn = counts[(0, 0)]
        fp = counts[(0, 1)]
        fn = counts[(1, 0)]
        tp = counts[(1, 1)]
        meta = metadata.get(code, {"description": code})
        metric_row = {
            "code": code,
            "description": meta.get("description", code),
            "clinical_group": clinical_group(meta),
            "diagnostic_class": meta.get("diagnostic_class", ""),
            "diagnostic_subclass": meta.get("diagnostic_subclass", ""),
            "diagnostic": meta.get("diagnostic", ""),
            "form": meta.get("form", ""),
            "rhythm": meta.get("rhythm", ""),
        }
        metric_row.update(compute_binary_metrics(tn=tn, fp=fp, fn=fn, tp=tp))
        summaries.append(metric_row)

    return summaries


# Function: Write dictionaries to a CSV file.
# Inputs: output path and rows with matching keys.
# Outputs: None; writes CSV to disk.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Write a dictionary or list to a JSON file.
# Inputs: output path and JSON-serializable payload.
# Outputs: None; writes JSON to disk.
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Print concise top/bottom SCP-code summaries to the terminal.
# Inputs: per-code summary rows and minimum support threshold.
# Outputs: None; prints human-readable summary.
def print_summary(rows: List[Dict[str, Any]], min_support: int) -> None:
    eligible = [
        row for row in rows
        if row["n"] >= min_support and row["balanced_accuracy"] is not None
    ]

    print(f"SCP codes: {len(rows)}")
    print(f"SCP codes with n >= {min_support}: {len(eligible)}")
    print("\nTop SCP codes by balanced accuracy")
    for row in sorted(eligible, key=lambda item: item["balanced_accuracy"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} {row['balanced_accuracy']:.3f} "
            f"n={row['n']:>4} yes_recall={row['yes_recall']:.3f} "
            f"no_recall={row['no_recall']:.3f} {row['description']}"
        )

    print("\nBottom SCP codes by balanced accuracy")
    for row in sorted(eligible, key=lambda item: item["balanced_accuracy"])[:12]:
        print(
            f"{row['code']:8s} {row['balanced_accuracy']:.3f} "
            f"n={row['n']:>4} yes_recall={row['yes_recall']:.3f} "
            f"no_recall={row['no_recall']:.3f} {row['description']}"
        )


# Function: Run per-SCP-code adapter performance analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON files with per-code metrics.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)
    rows = summarize_by_scp_code(predictions=predictions, metadata=metadata)

    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows)
    print_summary(rows, min_support=args.min_support)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
