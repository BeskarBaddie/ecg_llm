from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import PTBXL_METADATA, PTBXL_DIR


EMBEDDING_DATA_PATH = Path("outputs/ecgqa_csfm_preview_10000_sv.jsonl")
SCP_STATEMENTS_PATH = PTBXL_DIR / "scp_statements.csv"
OUTPUT_PATH = Path("outputs/unique_ecg_scp_dataset.jsonl")


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


def clean_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def load_scp_statement_map(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SCP statements file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    statement_map: Dict[str, Dict[str, Any]] = {}

    for code, row in df.iterrows():
        statement_map[str(code)] = {
            "description": clean_json_value(row.get("description")),
            "diagnostic": clean_json_value(row.get("diagnostic")),
            "form": clean_json_value(row.get("form")),
            "rhythm": clean_json_value(row.get("rhythm")),
            "diagnostic_class": clean_json_value(row.get("diagnostic_class")),
            "diagnostic_subclass": clean_json_value(row.get("diagnostic_subclass")),
        }

    return statement_map


def main() -> None:
    if not EMBEDDING_DATA_PATH.exists():
        raise FileNotFoundError(f"Embedding dataset not found: {EMBEDDING_DATA_PATH}")

    metadata = pd.read_csv(PTBXL_METADATA)
    metadata["scp_codes_parsed"] = metadata["scp_codes"].apply(parse_scp_codes)
    metadata_by_ecg_id = metadata.set_index("ecg_id")

    scp_statement_map = load_scp_statement_map(SCP_STATEMENTS_PATH)

    ecg_map: Dict[int, Dict[str, Any]] = {}
    missing_metadata = 0

    with EMBEDDING_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            ecg_id = row.get("ecg_id")
            if isinstance(ecg_id, list):
                if not ecg_id:
                    continue
                ecg_id = ecg_id[0]

            if ecg_id is None:
                continue

            ecg_id = int(ecg_id)

            if ecg_id in ecg_map:
                continue

            if ecg_id not in metadata_by_ecg_id.index:
                missing_metadata += 1
                continue

            metadata_row = metadata_by_ecg_id.loc[ecg_id]
            scp_codes = metadata_row["scp_codes_parsed"]

            scp_statements: List[Dict[str, Any]] = []
            for code, likelihood in sorted(scp_codes.items()):
                statement = dict(scp_statement_map.get(code, {}))
                statement["code"] = code
                statement["likelihood"] = likelihood
                scp_statements.append(statement)

            ecg_map[ecg_id] = {
                "ecg_id": ecg_id,
                "embedding": row["embedding"],
                "embedding_dim": row.get("embedding_dim"),
                "scp_codes": scp_codes,
                "scp_statements": scp_statements,
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in ecg_map.values():
            f.write(json.dumps(item) + "\n")

    print("Unique ECGs:", len(ecg_map))
    print("Missing PTB-XL metadata rows:", missing_metadata)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
