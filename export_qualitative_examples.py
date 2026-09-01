from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train_ecg_soft_prompt_adapter import load_jsonl, write_json


DEFAULT_OUTPUT_DIR = Path("outputs/qualitative_examples")
DEFAULT_FIGURE_DIR = Path("figures/qualitative_examples")


# Function: Write dictionaries to a CSV file.
# Inputs: output path and row dictionaries.
# Outputs: None; writes a CSV file.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Load PTB-XL metadata keyed by ECG ID.
# Inputs: path to ptbxl_database.csv.
# Outputs: dictionary keyed by ECG ID.
def load_ptbxl_metadata(path: Path | None) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row["ecg_id"]): dict(row) for _, row in df.iterrows()}


# Function: Load one ECG waveform from PTB-XL using wfdb.
# Inputs: PTB-XL root, metadata row, and sampling-rate choice.
# Outputs: tuple of signal array and sampling rate, or None when unavailable.
def load_waveform(ptbxl_root: Path, metadata_row: Dict[str, Any], sampling: str) -> tuple[Any, int] | None:
    try:
        import wfdb
    except ImportError:
        return None

    filename_key = "filename_hr" if sampling == "hr" else "filename_lr"
    sampling_rate = 500 if sampling == "hr" else 100
    record = str(metadata_row.get(filename_key, ""))
    if not record:
        return None
    record_path = ptbxl_root / record
    signal, _ = wfdb.rdsamp(str(record_path))
    return signal, sampling_rate


# Function: Select representative correct and incorrect examples.
# Inputs: prediction rows and examples per category.
# Outputs: selected rows tagged with selection category.
def select_examples(rows: List[Dict[str, Any]], per_bucket: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        true_answer = "yes" if int(row["true_label"]) == 1 else "no"
        pred_label = int(row.get("pred_label", row.get("default_pred_label", int((float(row["yes_score"]) - float(row["no_score"])) >= 0.0))))
        pred_answer = str(row.get("pred_answer", "yes" if pred_label == 1 else "no"))
        correct = bool(row.get("correct", pred_label == int(row["true_label"])))
        correctness = "correct" if correct else "incorrect"
        bucket = f"{correctness}_true_{true_answer}"
        score_diff = float(row["yes_score"]) - float(row["no_score"])
        confidence = abs(score_diff)
        buckets[bucket].append(
            {
                **row,
                "pred_label": pred_label,
                "pred_answer": pred_answer,
                "correct": correct,
                "score_diff": score_diff,
                "confidence": confidence,
            }
        )

    selected: List[Dict[str, Any]] = []
    for bucket, bucket_rows in sorted(buckets.items()):
        ranked = sorted(bucket_rows, key=lambda row: row["confidence"], reverse=True)
        for row in ranked[:per_bucket]:
            selected.append({**row, "selection_bucket": bucket})
    return selected


# Function: Plot a selected ECG waveform with the question and prediction metadata.
# Inputs: selected row, PTB-XL root, metadata lookup, figure directory, lead name, and sampling choice.
# Outputs: path to saved figure, or None when waveform plotting is unavailable.
def plot_example(
    row: Dict[str, Any],
    ptbxl_root: Path | None,
    metadata: Dict[int, Dict[str, Any]],
    figure_dir: Path,
    lead: str,
    sampling: str,
) -> str | None:
    if ptbxl_root is None:
        return None
    ecg_id = int(row["ecg_id"])
    meta = metadata.get(ecg_id)
    if meta is None:
        return None
    loaded = load_waveform(ptbxl_root, meta, sampling=sampling)
    if loaded is None:
        return None
    signal, sampling_rate = loaded
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lead_idx = lead_names.index(lead) if lead in lead_names else 1
    seconds = [idx / sampling_rate for idx in range(signal.shape[0])]

    figure_dir.mkdir(parents=True, exist_ok=True)
    safe_code = str(row["target_scp_code"]).replace("/", "_")
    filename = f"{row['selection_bucket']}_ecg{ecg_id}_{safe_code}.png"
    path = figure_dir / filename

    plt.figure(figsize=(10, 4))
    plt.plot(seconds, signal[:, lead_idx], linewidth=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel(f"Lead {lead}")
    plt.title(
        f"ECG {ecg_id} | {row['target_scp_code']} | true={row['answer']} pred={row['pred_answer']} | diff={row['score_diff']:.2f}"
    )
    plt.figtext(0.01, 0.01, str(row["question"])[:160], fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


# Function: Run qualitative example export from prediction rows.
# Inputs: command-line paths and plotting options.
# Outputs: CSV/JSONL selected examples plus optional ECG waveform PNGs.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--ptbxl-root", type=Path)
    parser.add_argument("--ptbxl-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--sampling", type=str, default="lr", choices=["lr", "hr"])
    args = parser.parse_args()

    rows = load_jsonl(args.predictions_path)
    selected = select_examples(rows, per_bucket=args.per_bucket)
    metadata = load_ptbxl_metadata(args.ptbxl_metadata)

    output_rows: List[Dict[str, Any]] = []
    for row in selected:
        figure_path = plot_example(
            row=row,
            ptbxl_root=args.ptbxl_root,
            metadata=metadata,
            figure_dir=args.figure_dir,
            lead=args.lead,
            sampling=args.sampling,
        )
        output_rows.append(
            {
                "selection_bucket": row["selection_bucket"],
                "ecg_id": row["ecg_id"],
                "target_scp_code": row["target_scp_code"],
                "question": row["question"],
                "answer": row["answer"],
                "pred_answer": row["pred_answer"],
                "correct": row["correct"],
                "yes_score": row["yes_score"],
                "no_score": row["no_score"],
                "score_diff": row["score_diff"],
                "confidence": row["confidence"],
                "figure_path": figure_path,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "qualitative_examples.csv", output_rows)
    with (args.output_dir / "qualitative_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    write_json(
        args.output_dir / "qualitative_examples_metadata.json",
        {
            "predictions_path": str(args.predictions_path),
            "ptbxl_root": str(args.ptbxl_root) if args.ptbxl_root else None,
            "ptbxl_metadata": str(args.ptbxl_metadata) if args.ptbxl_metadata else None,
            "lead": args.lead,
            "sampling": args.sampling,
            "n_examples": len(output_rows),
        },
    )

    print(f"Selected examples: {len(output_rows)}")
    print(f"Saved CSV: {args.output_dir / 'qualitative_examples.csv'}")
    print(f"Saved figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
