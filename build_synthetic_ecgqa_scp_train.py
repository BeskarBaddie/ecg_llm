from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_QUESTION_TEMPLATES = [
    "Does this ECG show {description}?",
    "Is {description} present on this ECG?",
    "Are there ECG findings consistent with {description}?",
    "Does the ECG indicate {description}?",
    "Is there evidence of {description} in this ECG?",
]


# Function: Load newline-delimited JSON rows from disk.
# Inputs: path to a JSONL file.
# Outputs: list of parsed row dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

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


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: output path and rows to serialize.
# Outputs: None; writes rows to disk.
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Write a dictionary to a JSON file.
# Inputs: output path and JSON-serializable dictionary.
# Outputs: None; writes JSON to disk.
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Normalize text for duplicate detection.
# Inputs: raw text value.
# Outputs: lowercase whitespace-normalized text.
def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# Function: Convert an SCP code description into question-friendly text.
# Inputs: SCP code and optional SCP statement dictionary.
# Outputs: human-readable attribute description.
def get_code_description(code: str, statement: Dict[str, Any] | None) -> str:
    if statement:
        description = statement.get("description")
        if description:
            return str(description).strip().lower()
    return code.replace("_", " ").replace("/", " or ").lower()


# Function: Decide whether an ECG should be labelled positive for one SCP code.
# Inputs: parsed PTB-XL SCP code dictionary, target SCP code, and likelihood threshold.
# Outputs: 1 if code likelihood meets threshold, otherwise 0.
def label_from_ptbxl_scp_codes(
    scp_codes: Dict[str, Any],
    code: str,
    positive_threshold: float,
) -> int:
    try:
        value = float(scp_codes.get(code, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return int(value >= positive_threshold)


# Function: Build a unique ECG lookup from real ECG-QA train rows.
# Inputs: real train rows.
# Outputs: dictionary keyed by ecg_id with PTB-XL SCP labels and signal metadata.
def build_unique_ecg_lookup(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    ecgs: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        ecg_id = int(row["ecg_id"])
        if ecg_id in ecgs:
            continue
        scp_codes = row.get("ptbxl_scp_codes") or {}
        if not isinstance(scp_codes, dict):
            scp_codes = {}
        ecgs[ecg_id] = {
            "ecg_id": ecg_id,
            "ptbxl_scp_codes": scp_codes,
            "embedding_dim": row.get("embedding_dim"),
            "signal_shape": row.get("signal_shape"),
            "official_split": row.get("official_split", row.get("split", "train")),
            "split": row.get("split", "train"),
        }
    return ecgs


# Function: Build SCP code descriptions from existing real rows.
# Inputs: real train rows.
# Outputs: dictionary mapping target SCP code to display description and statement metadata.
def build_code_metadata(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        code = str(row.get("target_scp_code", "")).strip()
        if not code or code in metadata:
            continue
        statement = row.get("target_scp_statement")
        if not isinstance(statement, dict):
            statement = {"code": code, "description": None}
        metadata[code] = {
            "target_scp_statement": statement,
            "description": get_code_description(code, statement),
        }
    return metadata


# Function: Build duplicate keys for real rows to avoid exact synthetic repeats.
# Inputs: real rows.
# Outputs: set of normalized duplicate keys.
def build_existing_question_keys(rows: List[Dict[str, Any]]) -> set[tuple[int, str, int, str]]:
    keys: set[tuple[int, str, int, str]] = set()
    for row in rows:
        keys.add(
            (
                int(row["ecg_id"]),
                str(row.get("target_scp_code", "")),
                int(row["label"]),
                normalize_text(row.get("question", "")),
            )
        )
    return keys


# Function: Select balanced positive and negative ECGs for each SCP code.
# Inputs: ECG lookup, target code, per-label cap, likelihood threshold, and RNG.
# Outputs: sampled (ecg, label) pairs with matched positive and negative counts when possible.
def sample_balanced_ecgs_for_code(
    ecgs: Dict[int, Dict[str, Any]],
    code: str,
    examples_per_label: int,
    positive_threshold: float,
    rng: random.Random,
) -> List[tuple[Dict[str, Any], int]]:
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []

    for ecg in ecgs.values():
        label = label_from_ptbxl_scp_codes(
            ecg["ptbxl_scp_codes"],
            code,
            positive_threshold=positive_threshold,
        )
        if label == 1:
            positives.append(ecg)
        else:
            negatives.append(ecg)

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n_per_label = min(examples_per_label, len(positives), len(negatives))
    sampled = [(ecg, 1) for ecg in positives[:n_per_label]]
    sampled.extend((ecg, 0) for ecg in negatives[:n_per_label])
    rng.shuffle(sampled)
    return sampled


# Function: Create a synthetic ECG-QA row from one ECG/code/label/template combination.
# Inputs: ECG metadata, SCP code metadata, label, question text, and source metadata.
# Outputs: JSON-serializable ECG-QA row compatible with adapter training.
def make_synthetic_row(
    ecg: Dict[str, Any],
    code: str,
    code_meta: Dict[str, Any],
    label: int,
    question: str,
    template: str,
    synthetic_index: int,
) -> Dict[str, Any]:
    answer = "yes" if label == 1 else "no"
    return {
        "ecg_id": int(ecg["ecg_id"]),
        "question": question,
        "answer": answer,
        "label": label,
        "question_type": "single-verify",
        "attribute_type": "scp_code",
        "attribute": code_meta["description"],
        "target_scp_code": code,
        "target_scp_statement": code_meta["target_scp_statement"],
        "ptbxl_scp_codes": ecg["ptbxl_scp_codes"],
        "embedding_dim": ecg.get("embedding_dim"),
        "signal_shape": ecg.get("signal_shape"),
        "official_split": "train",
        "split": "train",
        "source": "synthetic_scp_balanced_template",
        "label_source": "ptbxl_scp_codes",
        "base_template": template,
        "synthetic_index": synthetic_index,
    }


# Function: Generate balanced synthetic SCP-code ECG-QA rows.
# Inputs: real rows, per-label/code sample cap, phrasing count, threshold, code list, and seed.
# Outputs: synthetic rows and generation diagnostics.
def generate_synthetic_rows(
    real_rows: List[Dict[str, Any]],
    examples_per_label_per_code: int,
    phrasings_per_example: int,
    positive_threshold: float,
    target_codes: List[str] | None,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    ecgs = build_unique_ecg_lookup(real_rows)
    code_metadata = build_code_metadata(real_rows)
    codes = target_codes or sorted(code_metadata)
    existing_keys = build_existing_question_keys(real_rows)
    synthetic_keys = set(existing_keys)

    synthetic_rows: List[Dict[str, Any]] = []
    code_stats: Dict[str, Any] = {}
    skipped_exact_duplicates = 0
    synthetic_index = 0

    for code in codes:
        if code not in code_metadata:
            code_stats[code] = {"reason": "missing_code_metadata", "synthetic_rows": 0}
            continue

        sampled = sample_balanced_ecgs_for_code(
            ecgs=ecgs,
            code=code,
            examples_per_label=examples_per_label_per_code,
            positive_threshold=positive_threshold,
            rng=rng,
        )
        description = code_metadata[code]["description"]
        code_rows_before = len(synthetic_rows)
        label_counter: Counter[int] = Counter()

        for ecg, label in sampled:
            templates = list(DEFAULT_QUESTION_TEMPLATES)
            rng.shuffle(templates)
            for template in templates[:phrasings_per_example]:
                question = template.format(description=description)
                key = (int(ecg["ecg_id"]), code, label, normalize_text(question))
                if key in synthetic_keys:
                    skipped_exact_duplicates += 1
                    continue
                synthetic_keys.add(key)
                synthetic_index += 1
                label_counter[label] += 1
                synthetic_rows.append(
                    make_synthetic_row(
                        ecg=ecg,
                        code=code,
                        code_meta=code_metadata[code],
                        label=label,
                        question=question,
                        template=template,
                        synthetic_index=synthetic_index,
                    )
                )

        code_stats[code] = {
            "sampled_ecg_label_pairs": len(sampled),
            "synthetic_rows": len(synthetic_rows) - code_rows_before,
            "labels": dict(label_counter),
        }

    rng.shuffle(synthetic_rows)
    diagnostics = {
        "unique_train_ecgs": len(ecgs),
        "target_codes": codes,
        "examples_per_label_per_code": examples_per_label_per_code,
        "phrasings_per_example": phrasings_per_example,
        "positive_threshold": positive_threshold,
        "skipped_exact_duplicates": skipped_exact_duplicates,
        "synthetic_rows": len(synthetic_rows),
        "synthetic_label_counts": dict(Counter(row["label"] for row in synthetic_rows)),
        "synthetic_code_counts": dict(Counter(row["target_scp_code"] for row in synthetic_rows)),
        "code_stats": code_stats,
    }
    return synthetic_rows, diagnostics


# Function: Summarize real, synthetic, and combined train rows.
# Inputs: real rows, synthetic rows, generation diagnostics, and command-line arguments.
# Outputs: JSON-serializable statistics dictionary.
def summarize_outputs(
    real_rows: List[Dict[str, Any]],
    synthetic_rows: List[Dict[str, Any]],
    diagnostics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    combined_rows = real_rows + synthetic_rows
    return {
        "real_train_path": str(args.train_path),
        "synthetic_output_path": str(args.synthetic_output_path),
        "combined_output_path": str(args.combined_output_path),
        "real_rows": len(real_rows),
        "synthetic_rows": len(synthetic_rows),
        "combined_rows": len(combined_rows),
        "real_label_counts": dict(Counter(row["label"] for row in real_rows)),
        "synthetic_label_counts": dict(Counter(row["label"] for row in synthetic_rows)),
        "combined_label_counts": dict(Counter(row["label"] for row in combined_rows)),
        "real_code_counts": dict(Counter(row["target_scp_code"] for row in real_rows)),
        "synthetic_code_counts": dict(Counter(row["target_scp_code"] for row in synthetic_rows)),
        "combined_code_counts": dict(Counter(row["target_scp_code"] for row in combined_rows)),
        "generation": diagnostics,
    }


# Function: Parse a comma-separated SCP code argument.
# Inputs: raw target code string or "all".
# Outputs: sorted code list, or None when all codes should be inferred from train rows.
def parse_target_codes(raw: str) -> List[str] | None:
    if raw.strip().lower() == "all":
        return None
    codes = [code.strip().upper() for code in raw.split(",") if code.strip()]
    if not codes:
        raise ValueError("--target-scp-codes must be 'all' or a comma-separated code list.")
    return sorted(set(codes))


# Function: Build balanced synthetic ECG-QA training rows from real train rows.
# Inputs: command-line arguments.
# Outputs: synthetic JSONL, combined real+synthetic JSONL, and stats JSON.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--synthetic-output-path", type=Path, required=True)
    parser.add_argument("--combined-output-path", type=Path, required=True)
    parser.add_argument("--stats-output-path", type=Path, required=True)
    parser.add_argument("--target-scp-codes", type=str, default="all")
    parser.add_argument("--examples-per-label-per-code", type=int, default=300)
    parser.add_argument("--phrasings-per-example", type=int, default=1)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=50.0,
        help="PTB-XL SCP likelihood threshold for synthetic positive labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.examples_per_label_per_code <= 0:
        raise ValueError("--examples-per-label-per-code must be positive.")
    if args.phrasings_per_example <= 0 or args.phrasings_per_example > len(DEFAULT_QUESTION_TEMPLATES):
        raise ValueError(
            f"--phrasings-per-example must be between 1 and {len(DEFAULT_QUESTION_TEMPLATES)}."
        )

    real_rows = load_jsonl(args.train_path)
    synthetic_rows, diagnostics = generate_synthetic_rows(
        real_rows=real_rows,
        examples_per_label_per_code=args.examples_per_label_per_code,
        phrasings_per_example=args.phrasings_per_example,
        positive_threshold=args.positive_threshold,
        target_codes=parse_target_codes(args.target_scp_codes),
        seed=args.seed,
    )

    combined_rows = list(real_rows)
    for row in combined_rows:
        row.setdefault("source", "real_ecgqa")
    combined_rows.extend(synthetic_rows)

    write_jsonl(args.synthetic_output_path, synthetic_rows)
    write_jsonl(args.combined_output_path, combined_rows)
    stats = summarize_outputs(real_rows, synthetic_rows, diagnostics, args)
    write_json(args.stats_output_path, stats)

    print("Real train rows:", len(real_rows))
    print("Synthetic rows:", len(synthetic_rows))
    print("Combined rows:", len(combined_rows))
    print("Synthetic label counts:", stats["synthetic_label_counts"])
    print("Combined label counts:", stats["combined_label_counts"])
    print("Saved synthetic:", args.synthetic_output_path)
    print("Saved combined:", args.combined_output_path)
    print("Saved stats:", args.stats_output_path)


if __name__ == "__main__":
    main()
