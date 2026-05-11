from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import neurokit2 as nk


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def extract_basic_ecg_features(ecg_signal: np.ndarray, fs: int = 250) -> Dict[str, float]:
    """
    Extract a small set of clinically meaningful ECG features from one lead.
    This is the first version; we can expand it later to the full 54-feature set.
    """
    ecg_signal = np.asarray(ecg_signal)

    if ecg_signal.ndim != 1:
        raise ValueError(f"Expected 1D ECG signal, got shape {ecg_signal.shape}")

    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

    rpeaks = info.get("ECG_R_Peaks", [])
    rpeaks = np.asarray(rpeaks)

    features: Dict[str, float] = {
        "n_rpeaks": float(len(rpeaks)),
        "signal_mean": float(np.mean(ecg_clean)),
        "signal_std": float(np.std(ecg_clean)),
    }

    if len(rpeaks) > 1:
        rr_intervals = np.diff(rpeaks) / fs
        features["rr_mean"] = float(np.mean(rr_intervals))
        features["rr_std"] = float(np.std(rr_intervals))
        features["heart_rate_est"] = float(60.0 / np.mean(rr_intervals))
    else:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["heart_rate_est"] = np.nan

    return features


def extract_12lead_basic_features(ecg_12lead: np.ndarray, fs: int = 250) -> Dict[str, Dict[str, float]]:
    """
    Extract basic features for all 12 leads.
    Input shape should be (12, time).
    """
    ecg_12lead = np.asarray(ecg_12lead)

    if ecg_12lead.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got shape {ecg_12lead.shape}")

    all_features: Dict[str, Dict[str, float]] = {}

    for lead_name, lead_signal in zip(LEADS_12, ecg_12lead):
        all_features[lead_name] = extract_basic_ecg_features(lead_signal, fs=fs)

    return all_features