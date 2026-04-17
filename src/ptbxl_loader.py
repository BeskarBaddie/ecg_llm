from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from src.config import PTBXL_DIR, PTBXL_METADATA


def load_ptbxl_metadata(metadata_path: Path = PTBXL_METADATA) -> pd.DataFrame:
    """
    Load PTB-XL metadata CSV.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if "ecg_id" not in df.columns:
        raise ValueError("PTB-XL metadata does not contain an 'ecg_id' column.")

    return df


def get_ptbxl_row_by_ecg_id(ecg_id: int, metadata: pd.DataFrame) -> pd.Series:
    """
    Return the metadata row matching a given ecg_id.
    """
    matches = metadata[metadata["ecg_id"] == ecg_id]

    if matches.empty:
        raise KeyError(f"ecg_id {ecg_id} not found in PTB-XL metadata.")

    return matches.iloc[0]


def resolve_signal_base_path(
    row: pd.Series,
    ptbxl_root: Path = PTBXL_DIR,
    prefer: str = "hr",
) -> Path:
    """
    Build the base path to the WFDB record, without file extension.

    prefer:
        - "hr" -> use filename_hr if available
        - "lr" -> use filename_lr if available
    """
    ptbxl_root = Path(ptbxl_root)

    if prefer not in {"hr", "lr"}:
        raise ValueError("prefer must be either 'hr' or 'lr'")

    filename_col = "filename_hr" if prefer == "hr" else "filename_lr"

    if filename_col in row and isinstance(row[filename_col], str) and row[filename_col]:
        rel_path = row[filename_col]
    else:
        fallback_col = "filename_lr" if filename_col == "filename_hr" else "filename_hr"
        if fallback_col in row and isinstance(row[fallback_col], str) and row[fallback_col]:
            rel_path = row[fallback_col]
        else:
            raise ValueError("Could not find filename_hr or filename_lr in PTB-XL row.")

    return ptbxl_root / rel_path


def load_ecg_signal(
    ecg_id: int,
    metadata: Optional[pd.DataFrame] = None,
    ptbxl_root: Path = PTBXL_DIR,
    prefer: str = "hr",
) -> np.ndarray:
    """
    Load a PTB-XL ECG waveform for a single ecg_id.

    Returns:
        np.ndarray with shape [channels, time]
    """
    if metadata is None:
        metadata = load_ptbxl_metadata()

    row = get_ptbxl_row_by_ecg_id(ecg_id, metadata)
    signal_base = resolve_signal_base_path(row, ptbxl_root=ptbxl_root, prefer=prefer)

    hea_path = signal_base.with_suffix(".hea")
    dat_path = signal_base.with_suffix(".dat")

    if not hea_path.exists():
        raise FileNotFoundError(f"Missing .hea file: {hea_path}")
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing .dat file: {dat_path}")

    record = wfdb.rdrecord(str(signal_base))

    if record.p_signal is None:
        raise ValueError(f"No p_signal found for ecg_id {ecg_id}")

    # wfdb returns [time, channels]; we convert to [channels, time]
    signal = record.p_signal.T

    return signal


def describe_ecg_row(ecg_id: int, metadata: Optional[pd.DataFrame] = None) -> None:
    """
    Print a small summary for debugging.
    """
    if metadata is None:
        metadata = load_ptbxl_metadata()

    row = get_ptbxl_row_by_ecg_id(ecg_id, metadata)

    print(f"ecg_id: {ecg_id}")
    print(f"filename_lr: {row.get('filename_lr', 'N/A')}")
    print(f"filename_hr: {row.get('filename_hr', 'N/A')}")
    print(f"scp_codes: {row.get('scp_codes', 'N/A')}")
    print(f"patient_id: {row.get('patient_id', 'N/A')}")