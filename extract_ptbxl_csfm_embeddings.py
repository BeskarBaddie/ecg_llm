from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.ptbxl_loader import load_ecg_signal


DEFAULT_OUTPUT_JSONL = Path("outputs/ptbxl_csfm_embeddings.jsonl")
DEFAULT_MANIFEST_PATH = Path("outputs/ptbxl_csfm_embeddings_manifest.json")
FS_ORIGINAL = 500


# Function: Choose the best available torch device for CSFM embedding extraction.
# Inputs: optional user-requested device string.
# Outputs: torch device string accepted by torch.Tensor.to().
def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Function: Load the pretrained frozen CSFM encoder from a local repo/checkpoint.
# Inputs: CSFM repository root, checkpoint path, CSFM variant name, and target device.
# Outputs: CSFM model and checkpoint loading metadata.
def load_pretrained_csfm(
    csfm_repo_root: Path,
    checkpoint_path: Path,
    csfm_variant: str,
    device: str,
) -> tuple[nn.Module, Dict[str, Any]]:
    if not csfm_repo_root.exists():
        raise FileNotFoundError(f"CSFM repo not found: {csfm_repo_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CSFM checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(csfm_repo_root))
    from network.model import CSFM_model

    model = CSFM_model(csfm_variant).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_state_dict = {
        key.replace("encoder.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("encoder.") and "mlp_head" not in key
    }
    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
    model.mlp_head = nn.Identity()
    model.eval()

    metadata = {
        "csfm_repo_root": str(csfm_repo_root),
        "checkpoint_path": str(checkpoint_path),
        "csfm_variant": csfm_variant,
        "device": device,
        "loaded_keys": len(encoder_state_dict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, metadata


# Function: Load ECG IDs that have already been written to an embedding JSONL.
# Inputs: output JSONL path.
# Outputs: set of completed ECG IDs for resumable extraction.
def load_completed_ecg_ids(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring invalid JSON line {line_no} in {path}", flush=True)
                continue
            if "ecg_id" in row:
                completed.add(int(row["ecg_id"]))

    return completed


# Function: Select ECG IDs to process from PTB-XL metadata.
# Inputs: metadata table, optional limit, and optional explicit ECG ID list.
# Outputs: ordered ECG IDs for extraction.
def select_ecg_ids(
    metadata: pd.DataFrame,
    limit: int | None,
    ecg_id_path: Path | None,
) -> List[int]:
    if ecg_id_path is not None:
        if not ecg_id_path.exists():
            raise FileNotFoundError(f"ECG ID file not found: {ecg_id_path}")
        ecg_ids = []
        with ecg_id_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ecg_ids.append(int(line))
    else:
        ecg_ids = [int(x) for x in metadata["ecg_id"].tolist()]

    ecg_ids = sorted(dict.fromkeys(ecg_ids))
    if limit is not None:
        ecg_ids = ecg_ids[:limit]
    return ecg_ids


# Function: Extract one CSFM embedding from one PTB-XL ECG.
# Inputs: ECG ID, PTB-XL metadata/root, CSFM model/device, preprocessing function, and signal preference.
# Outputs: JSON-serializable embedding row.
def extract_one_embedding(
    ecg_id: int,
    metadata: pd.DataFrame,
    ptbxl_root: Path,
    model: nn.Module,
    device: str,
    prefer: str,
) -> Dict[str, Any]:
    from utils.preprocess import preprocess_ecg

    signal = load_ecg_signal(
        ecg_id,
        metadata=metadata,
        ptbxl_root=ptbxl_root,
        prefer=prefer,
    )
    signal = preprocess_ecg(signal, fs=FS_ORIGINAL).astype(np.float32)
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    channels = np.arange(signal.shape[0])

    with torch.no_grad():
        embedding = model(x, channels)

    embedding_vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return {
        "ecg_id": int(ecg_id),
        "embedding": embedding_vector.tolist(),
        "embedding_dim": int(embedding_vector.size),
        "signal_shape": list(signal.shape),
        "prefer": prefer,
    }


# Function: Write extraction run metadata for reproducibility.
# Inputs: manifest path, run configuration, CSFM metadata, and extraction counts.
# Outputs: None; writes JSON manifest.
def write_manifest(
    path: Path,
    config: Dict[str, Any],
    csfm_metadata: Dict[str, Any],
    counts: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "csfm": csfm_metadata,
                "counts": counts,
            },
            f,
            indent=2,
        )


# Function: Run resumable CSFM embedding extraction for PTB-XL ECG records.
# Inputs: command-line arguments.
# Outputs: JSONL embedding bank plus manifest JSON.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-root", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--csfm-repo-root", type=Path, required=True)
    parser.add_argument("--csfm-checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--csfm-variant",
        type=str,
        default="Tiny",
        choices=["Tiny", "Base", "Large"],
        help="CSFM architecture variant. Must match the supplied pretrained checkpoint.",
    )
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--prefer", choices=["hr", "lr"], default="hr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ecg-id-path",
        type=Path,
        default=None,
        help="Optional text file containing one ecg_id per line. Defaults to all PTB-XL metadata ECGs.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    metadata_path = args.metadata_path or args.ptbxl_root / "ptbxl_database.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata not found: {metadata_path}")

    device = resolve_device(args.device)
    metadata = pd.read_csv(metadata_path)
    if "ecg_id" not in metadata.columns:
        raise ValueError("PTB-XL metadata must contain an ecg_id column.")

    ecg_ids = select_ecg_ids(metadata, limit=args.limit, ecg_id_path=args.ecg_id_path)
    completed = load_completed_ecg_ids(args.output_jsonl) if args.resume else set()
    pending_ecg_ids = [ecg_id for ecg_id in ecg_ids if ecg_id not in completed]

    print("PTB-XL root:", args.ptbxl_root, flush=True)
    print("Metadata:", metadata_path, flush=True)
    print("CSFM variant:", args.csfm_variant, flush=True)
    print("CSFM checkpoint:", args.csfm_checkpoint_path, flush=True)
    print("Output:", args.output_jsonl, flush=True)
    print("Device:", device, flush=True)
    print("Total selected ECGs:", len(ecg_ids), flush=True)
    print("Already completed:", len(completed), flush=True)
    print("Pending:", len(pending_ecg_ids), flush=True)

    model, csfm_metadata = load_pretrained_csfm(
        args.csfm_repo_root,
        args.csfm_checkpoint_path,
        csfm_variant=args.csfm_variant,
        device=device,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    extracted = 0
    skipped = 0
    failed_rows: List[Dict[str, Any]] = []

    with args.output_jsonl.open(mode, encoding="utf-8") as f:
        for idx, ecg_id in enumerate(pending_ecg_ids, start=1):
            try:
                row = extract_one_embedding(
                    ecg_id,
                    metadata=metadata,
                    ptbxl_root=args.ptbxl_root,
                    model=model,
                    device=device,
                    prefer=args.prefer,
                )
            except Exception as exc:
                skipped += 1
                failed_rows.append({"ecg_id": int(ecg_id), "error": repr(exc)})
                print(f"Skipped ecg_id={ecg_id}: {exc}", flush=True)
                continue

            f.write(json.dumps(row) + "\n")
            extracted += 1

            if extracted % args.progress_every == 0 or idx == len(pending_ecg_ids):
                f.flush()
                print(
                    f"[{idx}/{len(pending_ecg_ids)}] extracted={extracted} skipped={skipped}",
                    flush=True,
                )

    counts = {
        "selected_ecgs": len(ecg_ids),
        "completed_before_run": len(completed),
        "pending_at_start": len(pending_ecg_ids),
        "extracted_this_run": extracted,
        "skipped_this_run": skipped,
        "failed_rows": failed_rows[:100],
        "failed_rows_truncated": len(failed_rows) > 100,
    }
    config = {
        "ptbxl_root": str(args.ptbxl_root),
        "metadata_path": str(metadata_path),
        "csfm_variant": args.csfm_variant,
        "csfm_checkpoint_path": str(args.csfm_checkpoint_path),
        "output_jsonl": str(args.output_jsonl),
        "prefer": args.prefer,
        "device": device,
        "limit": args.limit,
        "ecg_id_path": str(args.ecg_id_path) if args.ecg_id_path else None,
        "resume": args.resume,
    }
    write_manifest(args.manifest_path, config, csfm_metadata, counts)

    print("Saved embeddings:", args.output_jsonl, flush=True)
    print("Saved manifest:", args.manifest_path, flush=True)
    print("Counts:", counts, flush=True)


if __name__ == "__main__":
    main()
