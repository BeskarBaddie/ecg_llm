from __future__ import annotations

import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.config import OUTPUT_DIR, PTBXL_DIR, PTBXL_METADATA


INPUT_TRAIN_DATA_PATH = OUTPUT_DIR / "ecgqa_csfm_preview_20000_sv.jsonl"
INPUT_VAL_DATA_PATH = OUTPUT_DIR / "ecgqa_csfm_valid_scp_sv.jsonl"
SCP_STATEMENTS_PATH = PTBXL_DIR / "scp_statements.csv"

OUTPUT_ALL_PATH = OUTPUT_DIR / "ecgqa_scp_binary_subset.jsonl"
OUTPUT_TRAIN_PATH = OUTPUT_DIR / "ecgqa_scp_binary_train.jsonl"
OUTPUT_VAL_PATH = OUTPUT_DIR / "ecgqa_scp_binary_val.jsonl"
OUTPUT_STATS_PATH = OUTPUT_DIR / "ecgqa_scp_binary_subset_stats.json"

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


# Function: Load PTB-XL metadata indexed by ECG ID.
# Inputs: Path to ptbxl_database.csv.
# Outputs: DataFrame indexed by ecg_id with parsed SCP codes.
def load_ptbxl_metadata_by_ecg_id(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PTB-XL metadata file not found: {path}")

    metadata = pd.read_csv(path)
    metadata["scp_codes_parsed"] = metadata["scp_codes"].apply(parse_scp_codes)
    return metadata.set_index("ecg_id")


# Function: Load eligible ECG-QA rows from official train/validation embedding files.
# Inputs: Input path/split pairs, PTB-XL metadata, SCP statements, and selected SCP codes.
# Outputs: Matched subset rows and skipped-row reason counts.
def load_candidate_rows(
    input_specs: List[tuple[Path, str]],
    metadata_by_ecg_id: pd.DataFrame,
    scp_statements: pd.DataFrame,
    target_scp_codes: set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    attribute_to_code = build_attribute_to_scp_code_map(scp_statements)
    rows: List[Dict[str, Any]] = []
    skipped = Counter()

    for input_path, default_official_split in input_specs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input embedding dataset not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                if row.get("question_type") != "single-verify":
                    skipped["not_single_verify"] += 1
                    continue

                if row.get("attribute_type") != "scp_code":
                    skipped["not_scp_code"] += 1
                    continue

                answer = normalize_text(row.get("answer"))
                if answer not in ANSWER_TO_LABEL:
                    skipped["not_binary_yes_no"] += 1
                    continue

                attribute = normalize_text(row.get("attribute"))
                target_scp_code = attribute_to_code.get(attribute)
                if target_scp_code is None:
                    skipped["unmatched_attribute"] += 1
                    continue

                if target_scp_code not in target_scp_codes:
                    skipped["not_selected_code"] += 1
                    continue

                ecg_id = row.get("ecg_id")
                if isinstance(ecg_id, list):
                    if not ecg_id:
                        skipped["missing_ecg_id"] += 1
                        continue
                    ecg_id = ecg_id[0]

                if ecg_id is None:
                    skipped["missing_ecg_id"] += 1
                    continue

                ecg_id = int(ecg_id)
                if ecg_id not in metadata_by_ecg_id.index:
                    skipped["missing_ptbxl_metadata"] += 1
                    continue

                embedding = row.get("embedding")
                if embedding is None:
                    skipped["missing_embedding"] += 1
                    continue

                metadata_row = metadata_by_ecg_id.loc[ecg_id]
                official_split = row.get("official_split") or default_official_split

                rows.append(
                    {
                        "ecg_id": ecg_id,
                        "question": row.get("question"),
                        "answer": answer,
                        "label": ANSWER_TO_LABEL[answer],
                        "question_type": row.get("question_type"),
                        "attribute_type": row.get("attribute_type"),
                        "attribute": attribute,
                        "target_scp_code": target_scp_code,
                        "target_scp_statement": get_scp_statement(scp_statements, target_scp_code),
                        "ptbxl_scp_codes": metadata_row["scp_codes_parsed"],
                        "embedding": embedding,
                        "embedding_dim": row.get("embedding_dim"),
                        "signal_shape": row.get("signal_shape"),
                        "source_dataset": str(input_path),
                        "official_split": official_split,
                        "split": official_split,
                    }
                )

    return rows, dict(skipped)


# Function: Randomly split ECG IDs into train and validation IDs.
# Inputs: ECG IDs, validation fraction, and random seed.
# Outputs: Train ECG ID set and validation ECG ID set.
def split_ecg_ids(
    ecg_ids: Iterable[int],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    unique_ecg_ids = sorted(set(ecg_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ecg_ids)

    n_val = max(1, round(len(unique_ecg_ids) * val_fraction))
    val_ids = set(unique_ecg_ids[:n_val])
    train_ids = set(unique_ecg_ids[n_val:])

    return train_ids, val_ids


# Function: Assign a random ECG-level split to rows.
# Inputs: Rows, validation fraction, and random seed.
# Outputs: None; mutates each row with a split value.
def assign_splits(
    rows: List[Dict[str, Any]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    train_ids, val_ids = split_ecg_ids(
        (row["ecg_id"] for row in rows),
        val_fraction=val_fraction,
        seed=seed,
    )

    for row in rows:
        row["split"] = "val" if row["ecg_id"] in val_ids else "train"

    assert not train_ids & val_ids


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
# Inputs: Output path and rows to serialize.
# Outputs: None; writes file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build the official-split ECG-QA SCP binary subset.
# Inputs: Local constants for input paths and selected SCP codes.
# Outputs: Train/validation JSONL files plus summary statistics.
def main() -> None:
    target_scp_codes = set(DEFAULT_TARGET_SCP_CODES)

    scp_statements = load_scp_statement_table(SCP_STATEMENTS_PATH)
    metadata_by_ecg_id = load_ptbxl_metadata_by_ecg_id(PTBXL_METADATA)

    rows, skipped = load_candidate_rows(
        [
            (INPUT_TRAIN_DATA_PATH, "train"),
            (INPUT_VAL_DATA_PATH, "val"),
        ],
        metadata_by_ecg_id=metadata_by_ecg_id,
        scp_statements=scp_statements,
        target_scp_codes=target_scp_codes,
    )

    if not rows:
        raise RuntimeError("No rows matched the ECG-QA SCP binary subset filters.")

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    stats = summarize_rows(rows, skipped)
    stats["target_scp_codes"] = sorted(target_scp_codes)
    stats["split_source"] = "official_ecgqa_template_split"
    stats["input_train_data_path"] = str(INPUT_TRAIN_DATA_PATH)
    stats["input_val_data_path"] = str(INPUT_VAL_DATA_PATH)
    stats["scp_statements_path"] = str(SCP_STATEMENTS_PATH)
    stats["ptbxl_metadata_path"] = str(PTBXL_METADATA)

    write_jsonl(OUTPUT_ALL_PATH, rows)
    write_jsonl(OUTPUT_TRAIN_PATH, train_rows)
    write_jsonl(OUTPUT_VAL_PATH, val_rows)

    OUTPUT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved all rows:", OUTPUT_ALL_PATH)
    print("Saved train rows:", OUTPUT_TRAIN_PATH)
    print("Saved val rows:", OUTPUT_VAL_PATH)
    print("Saved stats:", OUTPUT_STATS_PATH)
    print("Total questions:", stats["total_questions"])
    print("Unique ECGs:", stats["unique_ecgs"])
    print("Splits:", stats["splits"])
    print("Answers:", stats["answers"])
    print("Questions per code:", stats["questions_per_code"])


if __name__ == "__main__":
    main()
