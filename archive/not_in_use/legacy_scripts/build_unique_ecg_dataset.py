from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path("outputs/ecgqa_csfm_preview_10000_sv.jsonl")
OUTPUT_PATH = Path("outputs/unique_ecg_dataset.jsonl")


def normalize_attribute(attr):
    if isinstance(attr, list):
        if not attr:
            return ""
        return str(attr[0]).strip().lower()

    return str(attr).strip().lower()


def main():
    ecg_map = {}

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            ecg_id = row["ecg_id"]
            if isinstance(ecg_id, list):
                ecg_id = ecg_id[0]

            ecg_id = int(ecg_id)

            attribute = normalize_attribute(row.get("attribute", ""))

            if ecg_id not in ecg_map:
                ecg_map[ecg_id] = {
                    "ecg_id": ecg_id,
                    "embedding": row["embedding"],
                    "attributes": set(),
                }

            if attribute:
                ecg_map[ecg_id]["attributes"].add(attribute)

    print("Unique ECGs:", len(ecg_map))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in ecg_map.values():
            out_row = {
                "ecg_id": item["ecg_id"],
                "embedding": item["embedding"],
                "attributes": sorted(list(item["attributes"])),
            }

            f.write(json.dumps(out_row) + "\n")

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()