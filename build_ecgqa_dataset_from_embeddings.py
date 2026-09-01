from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.ecgqa_loader import load_ecgqa_json


DEFAULT_TARGET_SCP_CODES = {
    "AFIB",
    "LAFB",
    "LVH",
    "NORM",
    "CLBBB",
    "CRBBB",
    "ASMI",
}

ANSWER_TO_LABEL = {
    "no": 0,
    "yes": 1,
}

MANUAL_ATTRIBUTE_TO_SCP_CODE = {
    "atrial fibrillation": "AFIB",
    "left anterior fascicular block": "LAFB",
    "left ventricular hypertrophy": "LVH",
    "voltage criteria (qrs) for left ventricular hypertrophy": "LVH",
    "normal ecg": "NORM",
    "complete left bundle branch block": "CLBBB",
    "complete right bundle branch block": "CRBBB",
    "myocardial infarction in anteroseptal leads": "ASMI",
}


# Function: Normalize ECG-QA scalar/list text fields.
# Inputs: Raw field value from ECG-QA.
# Outputs: Lowercase stripped string.
def normalize_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]

    return str(value).strip().lower()


# Function: Convert pandas missing values into JSON-safe nulls.
# Inputs: Value from a pandas row.
# Outputs: Original value or None.
def clean_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


# Function: Parse PTB-XL SCP code dictionaries stored as strings.
# Inputs: Raw PTB-XL scp_codes cell.
# Outputs: Dictionary mapping SCP code to likelihood.
def parse_scp_codes(raw_codes: Any) -> Dict[str, float]:
    if not isinstance(raw_codes, str) or not raw_codes.strip():
        return {}

    try:
        parsed = ast.literal_eval(raw_codes)
    except (SyntaxError, ValueError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    codes: Dict[str, float] = {}
    for code, value in parsed.items():
        try:
            codes[str(code)] = float(value)
        except (TypeError, ValueError):
            codes[str(code)] = 0.0

    return codes


# Function: Load the PTB-XL SCP statement lookup table.
# Inputs: Path to scp_statements.csv.
# Outputs: DataFrame indexed by SCP code.
def load_scp_statement_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SCP statements file not found: {path}")

    return pd.read_csv(path, index_col=0)


# Function: Load PTB-XL metadata indexed by ECG ID.
# Inputs: Path to ptbxl_database.csv.
# Outputs: DataFrame indexed by ecg_id with parsed SCP codes.
def load_ptbxl_metadata_by_ecg_id(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PTB-XL metadata file not found: {path}")

    metadata = pd.read_csv(path)
    metadata["scp_codes_parsed"] = metadata["scp_codes"].apply(parse_scp_codes)
    return metadata.set_index("ecg_id")


# Function: Build mapping from ECG-QA attribute text to SCP code.
# Inputs: SCP statement table.
# Outputs: Dictionary mapping normalized attribute descriptions to SCP codes.
def build_attribute_to_scp_code_map(scp_statements: pd.DataFrame) -> Dict[str, str]:
    attribute_to_code = dict(MANUAL_ATTRIBUTE_TO_SCP_CODE)
    text_columns = [
        "description",
        "SCP-ECG Statement Description",
    ]

    for code, row in scp_statements.iterrows():
        for column in text_columns:
            if column not in row:
                continue

            text = normalize_text(row.get(column))
            if text:
                attribute_to_code.setdefault(text, str(code))

    return attribute_to_code


# Function: Get JSON-safe metadata for one SCP statement.
# Inputs: SCP statement table and SCP code.
# Outputs: Dictionary of code description and diagnostic metadata.
def get_scp_statement(scp_statements: pd.DataFrame, code: str) -> Dict[str, Any]:
    if code not in scp_statements.index:
        return {
            "code": code,
            "description": None,
            "diagnostic": None,
            "form": None,
            "rhythm": None,
            "diagnostic_class": None,
            "diagnostic_subclass": None,
        }

    row = scp_statements.loc[code]
    return {
        "code": code,
        "description": clean_json_value(row.get("description")),
        "diagnostic": clean_json_value(row.get("diagnostic")),
        "form": clean_json_value(row.get("form")),
        "rhythm": clean_json_value(row.get("rhythm")),
        "diagnostic_class": clean_json_value(row.get("diagnostic_class")),
        "diagnostic_subclass": clean_json_value(row.get("diagnostic_subclass")),
    }


# Function: Load a CSFM embedding JSONL bank keyed by ECG ID.
# Inputs: embedding JSONL path and whether full embedding vectors are required.
# Outputs: dictionary mapping ecg_id to embedding metadata or full embedding payload.
def load_embedding_bank(path: Path, include_embeddings: bool) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding bank not found: {path}")

    embeddings: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ecg_id = int(row["ecg_id"])
            if ecg_id in embeddings:
                raise ValueError(f"Duplicate ecg_id={ecg_id} in embedding bank line {line_no}")
            if include_embeddings:
                embeddings[ecg_id] = row
            else:
                embeddings[ecg_id] = {
                    "ecg_id": ecg_id,
                    "embedding_dim": row.get("embedding_dim"),
                    "signal_shape": row.get("signal_shape"),
                }

    return embeddings


# Function: Parse a comma-separated SCP-code argument.
# Inputs: Optional string such as "AFIB,LVH,NORM"; "all" or empty means no SCP-code filter.
# Outputs: set of uppercase SCP codes, or None when all matched SCP codes are allowed.
def parse_target_codes(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None

    if value.strip().lower() == "all":
        return None

    return {item.strip().upper() for item in value.split(",") if item.strip()}


# Function: Keep only ECG-QA samples matching the binary single-verify SCP-code task.
# Inputs: ECG-QA samples, split name, lookup tables, embedding bank, and selected target codes.
# Outputs: final dataset rows and skipped-row reason counts.
def build_split_rows(
    samples: List[Dict[str, Any]],
    split: str,
    embedding_bank: Dict[int, Dict[str, Any]],
    metadata_by_ecg_id: pd.DataFrame,
    scp_statements: pd.DataFrame,
    attribute_to_code: Dict[str, str],
    target_scp_codes: set[str] | None,
    include_embeddings: bool,
) -> tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    skipped: Counter = Counter()

    for sample in samples:
        if sample.get("question_type") != "single-verify":
            skipped["not_single_verify"] += 1
            continue

        if sample.get("attribute_type") != "scp_code":
            skipped["not_scp_code"] += 1
            continue

        answer = normalize_text(sample.get("answer"))
        if answer not in ANSWER_TO_LABEL:
            skipped["not_binary_yes_no"] += 1
            continue

        attribute = normalize_text(sample.get("attribute"))
        target_scp_code = attribute_to_code.get(attribute)
        if target_scp_code is None:
            skipped["unmatched_attribute"] += 1
            continue

        if target_scp_codes is not None and target_scp_code not in target_scp_codes:
            skipped["not_selected_code"] += 1
            continue

        ecg_ids = sample.get("ecg_id", [])
        if not isinstance(ecg_ids, list):
            ecg_ids = [ecg_ids]
        if not ecg_ids:
            skipped["missing_ecg_id"] += 1
            continue

        ecg_id = int(ecg_ids[0])
        if ecg_id not in embedding_bank:
            skipped["missing_embedding"] += 1
            continue

        if ecg_id not in metadata_by_ecg_id.index:
            skipped["missing_ptbxl_metadata"] += 1
            continue

        embedding_item = embedding_bank[ecg_id]
        metadata_row = metadata_by_ecg_id.loc[ecg_id]
        row = {
            "ecg_id": ecg_id,
            "question": sample.get("question"),
            "answer": answer,
            "label": ANSWER_TO_LABEL[answer],
            "question_type": sample.get("question_type"),
            "attribute_type": sample.get("attribute_type"),
            "attribute": attribute,
            "target_scp_code": target_scp_code,
            "target_scp_statement": get_scp_statement(scp_statements, target_scp_code),
            "ptbxl_scp_codes": metadata_row["scp_codes_parsed"],
            "embedding_dim": embedding_item.get("embedding_dim"),
            "signal_shape": embedding_item.get("signal_shape"),
            "official_split": split,
            "split": split,
        }
        if include_embeddings:
            row["embedding"] = embedding_item["embedding"]

        rows.append(row)

    return rows, skipped


# Function: Summarize subset size, label balance, split balance, and code balance.
# Inputs: Final subset rows and skipped-row counts.
# Outputs: JSON-serializable statistics dictionary.
def summarize_rows(rows: List[Dict[str, Any]], skipped: Dict[str, int]) -> Dict[str, Any]:
    split_counter = Counter(row["split"] for row in rows)
    answer_counter = Counter(row["answer"] for row in rows)
    code_counter = Counter(row["target_scp_code"] for row in rows)
    split_code_counter = Counter((row["split"], row["target_scp_code"]) for row in rows)
    split_answer_counter = Counter((row["split"], row["answer"]) for row in rows)

    code_stats: Dict[str, Dict[str, Any]] = {}
    rows_by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_code[row["target_scp_code"]].append(row)

    for code, code_rows in sorted(rows_by_code.items()):
        code_stats[code] = {
            "questions": len(code_rows),
            "unique_ecgs": len({row["ecg_id"] for row in code_rows}),
            "answers": dict(Counter(row["answer"] for row in code_rows)),
            "splits": dict(Counter(row["split"] for row in code_rows)),
        }

    return {
        "total_questions": len(rows),
        "unique_ecgs": len({row["ecg_id"] for row in rows}),
        "splits": dict(split_counter),
        "answers": dict(answer_counter),
        "questions_per_code": dict(code_counter),
        "questions_per_split_and_code": {
            f"{split}:{code}": count
            for (split, code), count in sorted(split_code_counter.items())
        },
        "questions_per_split_and_answer": {
            f"{split}:{answer}": count
            for (split, answer), count in sorted(split_answer_counter.items())
        },
        "code_stats": code_stats,
        "skipped": skipped,
    }


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: output path and rows to serialize.
# Outputs: None; writes file to disk.
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build ECG-QA binary SCP-code datasets from a precomputed embedding bank.
# Inputs: command-line arguments with ECG-QA, PTB-XL, embedding, and output paths.
# Outputs: train/val/test/all JSONL files plus a stats JSON file.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--ecgqa-root", type=Path, required=True)
    parser.add_argument("--ptbxl-root", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--scp-statements-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-scp-codes",
        type=str,
        default="all",
        help=(
            "Comma-separated SCP codes, e.g. AFIB,LVH,NORM. "
            "Use 'all' to include every ECG-QA SCP attribute that maps to an SCP code."
        ),
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Store full embedding vectors in every QA row. Useful for current training scripts.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata_path or args.ptbxl_root / "ptbxl_database.csv"
    scp_statements_path = args.scp_statements_path or args.ptbxl_root / "scp_statements.csv"
    target_scp_codes = parse_target_codes(args.target_scp_codes)

    print("Loading embedding bank...")
    embedding_bank = load_embedding_bank(args.embedding_bank, include_embeddings=args.include_embeddings)
    print("Embedding ECGs:", len(embedding_bank))

    print("Loading PTB-XL metadata/SCP statements...")
    metadata_by_ecg_id = load_ptbxl_metadata_by_ecg_id(metadata_path)
    scp_statements = load_scp_statement_table(scp_statements_path)
    attribute_to_code = build_attribute_to_scp_code_map(scp_statements)

    split_specs = {
        "train": args.ecgqa_root / "template" / "train",
        "val": args.ecgqa_root / "template" / "valid",
        "test": args.ecgqa_root / "template" / "test",
    }

    all_rows: List[Dict[str, Any]] = []
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    skipped_total: Counter = Counter()

    for split, path in split_specs.items():
        print(f"Loading ECG-QA {split}: {path}")
        samples = load_ecgqa_json(path)
        rows, skipped = build_split_rows(
            samples,
            split=split,
            embedding_bank=embedding_bank,
            metadata_by_ecg_id=metadata_by_ecg_id,
            scp_statements=scp_statements,
            attribute_to_code=attribute_to_code,
            target_scp_codes=target_scp_codes,
            include_embeddings=args.include_embeddings,
        )
        rows_by_split[split] = rows
        all_rows.extend(rows)
        skipped_total.update({f"{split}:{key}": value for key, value in skipped.items()})
        print(f"{split} rows:", len(rows))

    stats = summarize_rows(all_rows, dict(skipped_total))
    stats.update(
        {
            "embedding_bank": str(args.embedding_bank),
            "embedding_bank_ecgs": len(embedding_bank),
            "ecgqa_root": str(args.ecgqa_root),
            "ptbxl_root": str(args.ptbxl_root),
            "metadata_path": str(metadata_path),
            "scp_statements_path": str(scp_statements_path),
            "target_scp_codes": "all" if target_scp_codes is None else sorted(target_scp_codes),
            "include_embeddings": args.include_embeddings,
            "split_source": "official_ecgqa_template_train_valid_test",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_all = args.output_dir / "ecgqa_scp_binary_subset.jsonl"
    output_train = args.output_dir / "ecgqa_scp_binary_train.jsonl"
    output_val = args.output_dir / "ecgqa_scp_binary_val.jsonl"
    output_test = args.output_dir / "ecgqa_scp_binary_test.jsonl"
    output_stats = args.output_dir / "ecgqa_scp_binary_subset_stats.json"

    write_jsonl(output_all, all_rows)
    write_jsonl(output_train, rows_by_split["train"])
    write_jsonl(output_val, rows_by_split["val"])
    write_jsonl(output_test, rows_by_split["test"])

    with output_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved all rows:", output_all)
    print("Saved train rows:", output_train)
    print("Saved val rows:", output_val)
    print("Saved test rows:", output_test)
    print("Saved stats:", output_stats)
    print("Total questions:", stats["total_questions"])
    print("Unique ECGs:", stats["unique_ecgs"])
    print("Splits:", stats["splits"])
    print("Answers:", stats["answers"])
    print("Questions per code:", stats["questions_per_code"])


if __name__ == "__main__":
    main()
