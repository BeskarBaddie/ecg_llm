from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from analyze_scp_code_performance import clinical_group, write_csv, write_json


# Function: Convert a CSV value to float while preserving missing values.
# Inputs: CSV string or numeric value.
# Outputs: float value, or None when the input is empty/not parseable.
def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# Function: Convert a CSV value to int while preserving missing values.
# Inputs: CSV string or numeric value.
# Outputs: integer value, or zero when the input is empty/not parseable.
def parse_int(value: Any) -> int:
    parsed = parse_optional_float(value)
    if parsed is None:
        return 0
    return int(parsed)


# Function: Load per-SCP-code comparison rows and add clinical group labels.
# Inputs: comparison CSV with SmolLM and Llama balanced accuracy columns.
# Outputs: list of normalized comparison rows.
def load_comparison_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            smollm_bal_acc = parse_optional_float(row.get("smollm_bal_acc"))
            llama_bal_acc = parse_optional_float(row.get("llama_bal_acc"))
            delta_bal_acc = (
                llama_bal_acc - smollm_bal_acc
                if smollm_bal_acc is not None and llama_bal_acc is not None
                else parse_optional_float(row.get("delta_bal_acc"))
            )
            meta = {
                "diagnostic_class": str(row.get("diagnostic_class") or ""),
                "diagnostic_subclass": str(row.get("diagnostic_subclass") or ""),
                "diagnostic": str(row.get("diagnostic") or ""),
                "form": str(row.get("form") or ""),
                "rhythm": str(row.get("rhythm") or ""),
            }
            rows.append(
                {
                    "code": str(row.get("code") or ""),
                    "description": str(row.get("description") or row.get("code") or ""),
                    "clinical_group": clinical_group(meta),
                    "diagnostic_class": meta["diagnostic_class"],
                    "diagnostic_subclass": meta["diagnostic_subclass"],
                    "diagnostic": meta["diagnostic"],
                    "form": meta["form"],
                    "rhythm": meta["rhythm"],
                    "n": parse_int(row.get("n")),
                    "smollm_bal_acc": smollm_bal_acc,
                    "llama_bal_acc": llama_bal_acc,
                    "delta_bal_acc": delta_bal_acc,
                }
            )
    return rows


# Function: Compute a weighted mean while skipping missing values.
# Inputs: rows, value column, and weight column.
# Outputs: weighted mean, or None when no valid rows are available.
def weighted_mean(rows: List[Dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator == 0:
        return None
    return numerator / denominator


# Function: Compute an unweighted mean while skipping missing values.
# Inputs: rows and value column.
# Outputs: arithmetic mean, or None when no valid rows are available.
def mean(rows: List[Dict[str, Any]], value_key: str) -> float | None:
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


# Function: Summarize model-size performance deltas by clinical group.
# Inputs: per-code comparison rows.
# Outputs: per-clinical-group summary rows.
def summarize_by_group(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["clinical_group"], []).append(row)

    summaries: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        valid_rows = [
            row for row in group_rows
            if row.get("smollm_bal_acc") is not None and row.get("llama_bal_acc") is not None
        ]
        improved = [row for row in valid_rows if float(row["delta_bal_acc"]) > 0]
        worsened = [row for row in valid_rows if float(row["delta_bal_acc"]) < 0]
        unchanged = [row for row in valid_rows if float(row["delta_bal_acc"]) == 0]

        summaries.append(
            {
                "clinical_group": group,
                "num_codes": len(group_rows),
                "num_codes_with_valid_comparison": len(valid_rows),
                "total_rows": sum(int(row.get("n", 0)) for row in group_rows),
                "codes": ",".join(sorted(row["code"] for row in group_rows)),
                "mean_smollm_bal_acc": mean(valid_rows, "smollm_bal_acc"),
                "mean_llama_bal_acc": mean(valid_rows, "llama_bal_acc"),
                "mean_delta_bal_acc": mean(valid_rows, "delta_bal_acc"),
                "weighted_smollm_bal_acc": weighted_mean(valid_rows, "smollm_bal_acc", "n"),
                "weighted_llama_bal_acc": weighted_mean(valid_rows, "llama_bal_acc", "n"),
                "weighted_delta_bal_acc": weighted_mean(valid_rows, "delta_bal_acc", "n"),
                "improved_codes": len(improved),
                "worsened_codes": len(worsened),
                "unchanged_codes": len(unchanged),
            }
        )
    return sorted(summaries, key=lambda row: row["weighted_delta_bal_acc"] or 0.0, reverse=True)


# Function: Print concise model comparison findings.
# Inputs: per-code and per-group comparison rows.
# Outputs: None; prints improvement/regression summaries.
def print_summary(rows: List[Dict[str, Any]], group_rows: List[Dict[str, Any]], min_support: int) -> None:
    print("Clinical-group comparison: Llama minus SmolLM balanced accuracy")
    for row in group_rows:
        print(
            f"{row['clinical_group']:30s} "
            f"codes={row['num_codes_with_valid_comparison']:>2} "
            f"weighted_delta={row['weighted_delta_bal_acc']:.3f} "
            f"weighted_smollm={row['weighted_smollm_bal_acc']:.3f} "
            f"weighted_llama={row['weighted_llama_bal_acc']:.3f} "
            f"improved={row['improved_codes']:>2} "
            f"worsened={row['worsened_codes']:>2}"
        )

    eligible = [
        row for row in rows
        if row.get("delta_bal_acc") is not None and int(row.get("n", 0)) >= min_support
    ]
    print(f"\nLargest SCP-code improvements with n >= {min_support}")
    for row in sorted(eligible, key=lambda item: item["delta_bal_acc"], reverse=True)[:12]:
        print(
            f"{row['code']:8s} "
            f"delta={row['delta_bal_acc']:.3f} "
            f"SmolLM={row['smollm_bal_acc']:.3f} "
            f"Llama={row['llama_bal_acc']:.3f} "
            f"{row['description']}"
        )

    print(f"\nLargest SCP-code regressions with n >= {min_support}")
    for row in sorted(eligible, key=lambda item: item["delta_bal_acc"])[:12]:
        print(
            f"{row['code']:8s} "
            f"delta={row['delta_bal_acc']:.3f} "
            f"SmolLM={row['smollm_bal_acc']:.3f} "
            f"Llama={row['llama_bal_acc']:.3f} "
            f"{row['description']}"
        )


# Function: Run model comparison analysis from per-SCP-code comparison CSV.
# Inputs: command-line arguments.
# Outputs: enriched per-code comparison CSV/JSON and group summary CSV/JSON.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-code-csv", type=Path, required=True)
    parser.add_argument("--output-code-json", type=Path, required=True)
    parser.add_argument("--output-group-csv", type=Path, required=True)
    parser.add_argument("--output-group-json", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    args = parser.parse_args()

    rows = load_comparison_rows(args.comparison_csv)
    group_rows = summarize_by_group(rows)

    write_csv(args.output_code_csv, rows)
    write_json(args.output_code_json, rows)
    write_csv(args.output_group_csv, group_rows)
    write_json(args.output_group_json, group_rows)
    print_summary(rows=rows, group_rows=group_rows, min_support=args.min_support)
    print(f"\nSaved code comparison CSV: {args.output_code_csv}")
    print(f"Saved group comparison CSV: {args.output_group_csv}")


if __name__ == "__main__":
    main()
