from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


DATA_PATH = Path("outputs/ecgqa_scp_binary_subset.jsonl")
RESULTS_PATH = Path("outputs/ecgqa_label_alignment_results.json")
MISMATCHES_PATH = Path("outputs/ecgqa_label_alignment_mismatches.csv")


# Function: Load newline-delimited JSON rows from disk.
# Inputs: path to a JSONL dataset file.
# Outputs: list of parsed row dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

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


# Function: Parse command-line SCP alias assignments.
# Inputs: Repeated alias strings formatted as ECGQA_CODE=PTBXL_CODE.
# Outputs: Dictionary mapping ECG-QA target codes to PTB-XL lookup codes.
def parse_code_aliases(alias_values: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for value in alias_values:
        if "=" not in value:
            raise ValueError(f"Invalid --code-alias value: {value}. Expected A=B.")
        source, target = value.split("=", maxsplit=1)
        aliases[source.strip().upper()] = target.strip().upper()
    return aliases


# Function: Convert PTB-XL SCP metadata into a binary code-presence label.
# Inputs: ECG-QA row, likelihood threshold, code aliases, and key-presence code set.
# Outputs: Integer binary label where 1 means the target SCP code is present.
def ptbxl_presence_label(
    row: Dict[str, Any],
    likelihood_threshold: float,
    code_aliases: Dict[str, str],
    key_presence_codes: set[str],
) -> int:
    original_code = str(row["target_scp_code"]).upper()
    target_code = code_aliases.get(original_code, original_code)
    scp_codes = row.get("ptbxl_scp_codes") or {}

    if target_code in key_presence_codes:
        return int(target_code in scp_codes)

    likelihood = float(scp_codes.get(target_code, 0.0))
    return int(likelihood >= likelihood_threshold)


# Function: Compute agreement metrics between ECG-QA labels and PTB-XL SCP presence.
# Inputs: Rows, likelihood threshold, code aliases, and key-presence code set.
# Outputs: Metrics dictionary with overall, per-split, and per-code agreement.
def compute_alignment(
    rows: List[Dict[str, Any]],
    likelihood_threshold: float,
    code_aliases: Dict[str, str],
    key_presence_codes: set[str],
) -> Dict[str, Any]:
    y_ecgqa = [int(row["label"]) for row in rows]
    y_ptbxl = [
        ptbxl_presence_label(row, likelihood_threshold, code_aliases, key_presence_codes)
        for row in rows
    ]

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[f"split:{row['split']}"].append(idx)
        groups[f"code:{row['target_scp_code']}"].append(idx)
        groups[f"split_code:{row['split']}:{row['target_scp_code']}"].append(idx)

    def metric_block(indices: List[int]) -> Dict[str, Any]:
        true = [y_ecgqa[idx] for idx in indices]
        pred = [y_ptbxl[idx] for idx in indices]
        return {
            "n": len(indices),
            "agreement_accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred))
            if len(set(true)) > 1
            else None,
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "ecgqa_label_counts": dict(Counter(true)),
            "ptbxl_presence_counts": dict(Counter(pred)),
            "confusion_matrix_labels": ["ecgqa_no", "ecgqa_yes"],
            "confusion_matrix_rows_ecgqa_cols_ptbxl": confusion_matrix(
                true,
                pred,
                labels=[0, 1],
            ).tolist(),
        }

    return {
        "data_path": str(DATA_PATH),
        "likelihood_threshold": likelihood_threshold,
        "code_aliases": code_aliases,
        "key_presence_codes": sorted(key_presence_codes),
        "overall": metric_block(list(range(len(rows)))),
        "by_group": {
            group: metric_block(indices)
            for group, indices in sorted(groups.items())
        },
    }


# Function: Extract rows where ECG-QA label disagrees with PTB-XL SCP presence.
# Inputs: Rows, likelihood threshold, code aliases, and key-presence code set.
# Outputs: List of compact mismatch dictionaries for CSV writing.
def collect_mismatches(
    rows: List[Dict[str, Any]],
    likelihood_threshold: float,
    code_aliases: Dict[str, str],
    key_presence_codes: set[str],
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for row in rows:
        ecgqa_label = int(row["label"])
        ptbxl_label = ptbxl_presence_label(
            row,
            likelihood_threshold,
            code_aliases,
            key_presence_codes,
        )
        if ecgqa_label == ptbxl_label:
            continue

        target_code = str(row["target_scp_code"])
        lookup_code = code_aliases.get(target_code.upper(), target_code.upper())
        scp_codes = row.get("ptbxl_scp_codes") or {}
        mismatches.append(
            {
                "split": row.get("split"),
                "ecg_id": row.get("ecg_id"),
                "target_scp_code": target_code,
                "ptbxl_lookup_code": lookup_code,
                "ecgqa_label": ecgqa_label,
                "ecgqa_answer": row.get("answer"),
                "ptbxl_presence_label": ptbxl_label,
                "ptbxl_likelihood": scp_codes.get(lookup_code, 0.0),
                "ptbxl_scp_codes": json.dumps(scp_codes, sort_keys=True),
                "question": row.get("question"),
            }
        )

    return mismatches


# Function: Write mismatch rows to CSV for manual inspection.
# Inputs: Output path and mismatch dictionaries.
# Outputs: None; writes CSV file to disk.
def write_mismatches_csv(path: Path, mismatches: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "ecg_id",
        "target_scp_code",
        "ptbxl_lookup_code",
        "ecgqa_label",
        "ecgqa_answer",
        "ptbxl_presence_label",
        "ptbxl_likelihood",
        "ptbxl_scp_codes",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatches)


# Function: Run the ECG-QA versus PTB-XL SCP label alignment diagnostic.
# Inputs: CLI arguments for dataset path, likelihood threshold, and output files.
# Outputs: JSON metrics and CSV mismatch file written to disk.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--mismatches-path", type=Path, default=MISMATCHES_PATH)
    parser.add_argument("--likelihood-threshold", type=float, default=1.0)
    parser.add_argument(
        "--code-alias",
        action="append",
        default=[],
        help="Map ECG-QA target code to PTB-XL lookup code, e.g. LVH=VCLVH.",
    )
    parser.add_argument(
        "--key-presence-code",
        action="append",
        default=[],
        help="Treat code as present when the key exists, regardless of likelihood.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.data_path)
    if not rows:
        raise RuntimeError("No rows available for label alignment analysis.")

    code_aliases = parse_code_aliases(args.code_alias)
    key_presence_codes = {code.strip().upper() for code in args.key_presence_code}

    results = compute_alignment(
        rows,
        args.likelihood_threshold,
        code_aliases,
        key_presence_codes,
    )
    mismatches = collect_mismatches(
        rows,
        args.likelihood_threshold,
        code_aliases,
        key_presence_codes,
    )
    results["n_mismatches"] = len(mismatches)
    results["mismatch_rate"] = len(mismatches) / len(rows)
    results["mismatches_path"] = str(args.mismatches_path)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_mismatches_csv(args.mismatches_path, mismatches)

    overall = results["overall"]
    print("Rows:", overall["n"])
    print("Agreement accuracy:", overall["agreement_accuracy"])
    print("Balanced accuracy:", overall["balanced_accuracy"])
    print("Macro-F1:", overall["macro_f1"])
    print("Mismatches:", results["n_mismatches"])
    print("Mismatch rate:", results["mismatch_rate"])
    print(f"Saved results: {args.results_path}")
    print(f"Saved mismatches: {args.mismatches_path}")


if __name__ == "__main__":
    main()
