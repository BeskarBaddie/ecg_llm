from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def _load_single_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}, got {type(data)}")

    return data


def load_ecgqa_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load ECG-QA data from either:
    - a single JSON file, or
    - a directory containing multiple JSON files.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"ECG-QA path not found: {path}")

    if path.is_file():
        return _load_single_json(path)

    if path.is_dir():
        all_samples: List[Dict[str, Any]] = []
        json_files = sorted(path.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {path}")

        for json_file in json_files:
            all_samples.extend(_load_single_json(json_file))

        return all_samples

    raise ValueError(f"Unsupported ECG-QA path: {path}")


def inspect_ecgqa_sample(sample: Dict[str, Any]) -> None:
    keys_of_interest = [
        "template_id",
        "question_id",
        "sample_id",
        "question_type",
        "attribute_type",
        "question",
        "answer",
        "ecg_id",
        "attribute",
    ]

    for key in keys_of_interest:
        if key in sample:
            print(f"{key}: {sample[key]}")