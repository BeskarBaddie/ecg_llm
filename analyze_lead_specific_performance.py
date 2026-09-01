from __future__ import annotations

import argparse
import re
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


EXPLICIT_LEAD_PATTERN = re.compile(
    r"\blead\s+(?:i{1,3}|avr|avl|avf|v[1-6])\b",
    flags=re.IGNORECASE,
)

BARE_CHEST_LEAD_PATTERN = re.compile(r"\bv[1-6]\b", flags=re.IGNORECASE)

REGIONAL_LEAD_TERMS = (
    "anteroseptal leads",
    "anterolateral leads",
    "inferolateral leads",
    "inferior leads",
    "lateral leads",
    "anterior leads",
    "septal leads",
    "limb leads",
    "chest leads",
    "frontal and horizontal leads",
)


# Function: Classify whether a question asks about the whole ECG or localized leads.
# Inputs: ECG-QA question text.
# Outputs: localization label: explicit_lead, regional_leads, or whole_ecg.
def classify_question_localization(question: str) -> str:
    question_lower = question.lower()
    if EXPLICIT_LEAD_PATTERN.search(question_lower) or BARE_CHEST_LEAD_PATTERN.search(question_lower):
        return "explicit_lead"
    if any(term in question_lower for term in REGIONAL_LEAD_TERMS):
        return "regional_leads"
    return "whole_ecg"


# Function: Collect example questions for each localization class.
# Inputs: prediction rows and maximum number of examples per class.
# Outputs: dictionary of localization class to representative question strings.
def collect_examples(
    predictions: List[Dict[str, Any]],
    max_examples: int,
) -> Dict[str, List[str]]:
    examples: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        question = str(row.get("question", ""))
        localization = classify_question_localization(question)
        if question in seen[localization]:
            continue
        if len(examples[localization]) >= max_examples:
            continue
        seen[localization].add(question)
        examples[localization].append(question)
    return dict(examples)


# Function: Compute binary metrics for one named group of prediction rows.
# Inputs: group name, rows, and optional extra metadata columns.
# Outputs: one summary dictionary with confusion counts and metrics.
def summarize_prediction_group(
    group_name: str,
    rows: List[Dict[str, Any]],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    summary = {"group": group_name}
    if extra:
        summary.update(extra)
    summary.update(compute_binary_metrics(tn=tn, fp=fp, fn=fn, tp=tp))
    return summary


# Function: Summarize adapter performance by question localization.
# Inputs: prediction rows.
# Outputs: list of metric rows for whole-ECG, explicit-lead, regional, and combined localized questions.
def summarize_by_localization(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[classify_question_localization(str(row.get("question", "")))].append(row)

    localized = grouped["explicit_lead"] + grouped["regional_leads"]
    output_rows = [
        summarize_prediction_group("whole_ecg", grouped["whole_ecg"]),
        summarize_prediction_group("explicit_lead", grouped["explicit_lead"]),
        summarize_prediction_group("regional_leads", grouped["regional_leads"]),
        summarize_prediction_group("localized_any", localized),
    ]
    return sorted(output_rows, key=lambda row: row["n"], reverse=True)


# Function: Summarize performance by clinical group and question localization.
# Inputs: prediction rows and SCP metadata lookup.
# Outputs: list of metric rows for each clinical-group/localization pair.
def summarize_by_group_and_localization(
    predictions: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        code = str(row.get("target_scp_code", "UNKNOWN"))
        group = clinical_group(metadata.get(code, {"description": code}))
        localization = classify_question_localization(str(row.get("question", "")))
        grouped[(group, localization)].append(row)

    summaries: List[Dict[str, Any]] = []
    for (clinical_group_name, localization), rows in sorted(grouped.items()):
        codes = sorted({str(row.get("target_scp_code", "UNKNOWN")) for row in rows})
        summaries.append(
            summarize_prediction_group(
                group_name=f"{clinical_group_name} | {localization}",
                rows=rows,
                extra={
                    "clinical_group": clinical_group_name,
                    "localization": localization,
                    "num_scp_codes": len(codes),
                    "scp_codes": ",".join(codes),
                },
            )
        )
    return sorted(summaries, key=lambda row: (row["clinical_group"], row["localization"]))


# Function: Print concise lead-specific performance summaries.
# Inputs: localization rows and group/localization rows.
# Outputs: None; prints human-readable results.
def print_summary(
    localization_rows: List[Dict[str, Any]],
    group_localization_rows: List[Dict[str, Any]],
    min_support: int,
) -> None:
    print("Performance by question localization")
    print("group | n | balanced_accuracy | yes_recall | no_recall | FN | FP")
    for row in localization_rows:
        print(
            f"{row['group']:16s} "
            f"n={row['n']:>4} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"yes_rec={row['yes_recall']:.3f} "
            f"no_rec={row['no_recall']:.3f} "
            f"FN={row['false_negative']:>3} "
            f"FP={row['false_positive']:>3}"
        )

    print(f"\nClinical group x localization rows with n >= {min_support}")
    eligible = [row for row in group_localization_rows if row["n"] >= min_support]
    for row in sorted(eligible, key=lambda item: item["balanced_accuracy"]):
        print(
            f"{row['clinical_group']:30s} "
            f"{row['localization']:14s} "
            f"n={row['n']:>4} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"yes_rec={row['yes_recall']:.3f} "
            f"no_rec={row['no_recall']:.3f}"
        )


# Function: Run lead-specific question performance analysis.
# Inputs: command-line arguments.
# Outputs: CSV and JSON files with localization and group-localization metrics.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--scp-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-localization-csv", type=Path, required=True)
    parser.add_argument("--output-localization-json", type=Path, required=True)
    parser.add_argument("--output-group-localization-csv", type=Path, required=True)
    parser.add_argument("--output-group-localization-json", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=50)
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    metadata = load_scp_metadata(args.scp_metadata_csv)
    localization_rows = summarize_by_localization(predictions)
    group_localization_rows = summarize_by_group_and_localization(predictions, metadata)
    examples = collect_examples(predictions, max_examples=8)

    write_csv(args.output_localization_csv, localization_rows)
    write_json(args.output_localization_json, localization_rows)
    write_csv(args.output_group_localization_csv, group_localization_rows)
    write_json(args.output_group_localization_json, group_localization_rows)
    write_json(args.examples_json, examples)

    print_summary(
        localization_rows=localization_rows,
        group_localization_rows=group_localization_rows,
        min_support=args.min_support,
    )
    print(f"\nSaved localization CSV: {args.output_localization_csv}")
    print(f"Saved group/localization CSV: {args.output_group_localization_csv}")
    print(f"Saved examples JSON: {args.examples_json}")


if __name__ == "__main__":
    main()
