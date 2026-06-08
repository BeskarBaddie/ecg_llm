from __future__ import annotations

from typing import Any, Callable, Dict, Iterable
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


# Function: Safely compute a percentile from a numeric vector.
# Inputs: Numeric values and target percentile.
# Outputs: Percentile value as float, or NaN when no finite values exist.
def safe_percentile(values: Iterable[float], percentile: float) -> float:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


# Function: Safely compute a mean while ignoring NaN values.
# Inputs: Numeric values.
# Outputs: Mean of finite values, or NaN when no finite values exist.
def safe_nanmean(values: Iterable[float]) -> float:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


# Function: Extract a compact single-lead ECG domain feature set.
# Inputs: One ECG lead signal, sampling frequency, and optional maximum feature count.
# Outputs: Dictionary of interpretable rhythm, amplitude, and signal-shape features.
def extract_single_lead_domain_features(
    ecg_signal: np.ndarray,
    fs: int = 500,
    max_features: int | None = 10,
) -> Dict[str, float]:
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64)

    if ecg_signal.ndim != 1:
        raise ValueError(f"Expected 1D ECG signal, got shape {ecg_signal.shape}")

    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < len(ecg_clean))]

    duration_sec = len(ecg_clean) / float(fs) if fs > 0 else float("nan")
    rr_intervals = np.diff(rpeaks) / float(fs) if len(rpeaks) > 1 else np.asarray([])
    r_amplitudes = ecg_clean[rpeaks] if len(rpeaks) else np.asarray([])

    signal_std = float(np.std(ecg_clean))
    signal_mean = float(np.mean(ecg_clean))
    centered = ecg_clean - signal_mean
    if signal_std > 0:
        signal_skew = float(np.mean((centered / signal_std) ** 3))
        signal_kurtosis = float(np.mean((centered / signal_std) ** 4))
    else:
        signal_skew = float("nan")
        signal_kurtosis = float("nan")

    if rr_intervals.size:
        rr_mean = float(np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
        heart_rate_mean = float(60.0 / rr_mean) if rr_mean > 0 else float("nan")
        heart_rate_std = float(np.std(60.0 / rr_intervals)) if np.all(rr_intervals > 0) else float("nan")
        rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))) if rr_intervals.size > 1 else float("nan")
        pnn50 = float(np.mean(np.abs(np.diff(rr_intervals)) > 0.05)) if rr_intervals.size > 1 else float("nan")
    else:
        rr_mean = float("nan")
        rr_std = float("nan")
        heart_rate_mean = float("nan")
        heart_rate_std = float("nan")
        rmssd = float("nan")
        pnn50 = float("nan")

    features: Dict[str, float] = {
        "duration_sec": duration_sec,
        "n_rpeaks": float(len(rpeaks)),
        "heart_rate_mean_bpm": heart_rate_mean,
        "heart_rate_std_bpm": heart_rate_std,
        "rr_mean_sec": rr_mean,
        "rr_std_sec": rr_std,
        "rmssd_sec": rmssd,
        "pnn50": pnn50,
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "signal_rms": float(np.sqrt(np.mean(ecg_clean**2))),
        "signal_skew": signal_skew,
        "signal_kurtosis": signal_kurtosis,
        "r_amplitude_mean": float(np.mean(r_amplitudes)) if r_amplitudes.size else float("nan"),
        "r_amplitude_std": float(np.std(r_amplitudes)) if r_amplitudes.size else float("nan"),
        "r_amplitude_p05": safe_percentile(r_amplitudes, 5),
        "r_amplitude_p95": safe_percentile(r_amplitudes, 95),
    }

    if max_features is None:
        return features

    return dict(list(features.items())[:max_features])


# Function: Return ECG amplitude at the current R peak.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: R-peak amplitude.
def amplitude_R(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float(sig[peaks_locs[beatno]])


# Function: Return the RR interval before the current beat.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: Previous RR interval in seconds.
def RR0(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float((peaks_locs[beatno] - peaks_locs[beatno - 1]) / sampling_rate)


# Function: Return the RR interval after the current beat.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: Next RR interval in seconds.
def RR1(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float((peaks_locs[beatno + 1] - peaks_locs[beatno]) / sampling_rate)


# Function: Return the RR interval after the next beat.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: Second-next RR interval in seconds.
def RR2(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float((peaks_locs[beatno + 2] - peaks_locs[beatno + 1]) / sampling_rate)


# Function: Return a local mean RR interval around the current beat.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: Mean of RR0, RR1, and RR2.
def RRm(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float(np.mean([RR0(sig, sampling_rate, peaks_locs, beatno), RR1(sig, sampling_rate, peaks_locs, beatno), RR2(sig, sampling_rate, peaks_locs, beatno)]))


# Function: Return the previous-to-next RR ratio.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: RR0 divided by RR1.
def RR_0_1(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float(RR0(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno))


# Function: Return the second-next-to-next RR ratio.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: RR2 divided by RR1.
def RR_2_1(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float(RR2(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno))


# Function: Return the local-mean-to-next RR ratio.
# Inputs: ECG signal, sampling rate, R-peak locations, and beat index.
# Outputs: RRm divided by RR1.
def RR_m_1(sig: np.ndarray, sampling_rate: float, peaks_locs: np.ndarray, beatno: int) -> float:
    return float(RRm(sig, sampling_rate, peaks_locs, beatno) / RR1(sig, sampling_rate, peaks_locs, beatno))


FEATURES_RPEAKS: Dict[str, Callable[..., float]] = {
    "a_R": amplitude_R,
    "RR0": RR0,
    "RR1": RR1,
    "RR2": RR2,
    "RRm": RRm,
    "RR_0_1": RR_0_1,
    "RR_2_1": RR_2_1,
    "RR_m_1": RR_m_1,
}


# Function: Calculate averaged R-peak-based ECG features.
# Inputs: Clean ECG signal, R-peak locations, sampling rate, prefix, and averaging flag.
# Outputs: Dictionary of R-peak feature names to scalar values.
def from_Rpeaks(
    sig: np.ndarray,
    peaks_locs: np.ndarray,
    sampling_rate: float,
    prefix: str = "ecg",
    average: bool = False,
) -> Dict[str, float] | Dict[int, Dict[str, float]]:
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    peaks_locs = np.asarray(peaks_locs, dtype=np.int64)
    features_rpeaks: Dict[int, Dict[str, float]] = {}
    for beatno in range(1, len(peaks_locs) - 2):
        features: Dict[str, float] = {}
        for key, func in FEATURES_RPEAKS.items():
            try:
                features["_".join([prefix, key])] = func(
                    sig,
                    sampling_rate,
                    peaks_locs=peaks_locs,
                    beatno=beatno,
                )
            except Exception:
                features["_".join([prefix, key])] = float("nan")
        features_rpeaks[beatno] = features

    if not average:
        return features_rpeaks

    features_avg: Dict[str, float] = {}
    for key in [f"{prefix}_{name}" for name in FEATURES_RPEAKS]:
        values = [beat_features.get(key, np.nan) for beat_features in features_rpeaks.values()]
        features_avg[key] = safe_nanmean(values)
    return features_avg


# Function: Return a fiducial location when it exists and is finite.
# Inputs: Fiducial list/array and beat index.
# Outputs: Integer sample index, or None when unavailable.
def get_fiducial(fiducials: Any, beatno: int) -> int | None:
    if beatno >= len(fiducials):
        return None
    value = fiducials[beatno]
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        return None
    return int(value)


# Function: Compute elapsed time between two ECG fiducials.
# Inputs: Start/end sample positions and sampling rate.
# Outputs: Time difference in seconds.
def fiducial_time(start: int | None, end: int | None, sampling_rate: float) -> float:
    if start is None or end is None:
        return float("nan")
    return float((end - start) / sampling_rate)


# Function: Compute amplitude difference between two ECG fiducials.
# Inputs: ECG signal and two sample positions.
# Outputs: Amplitude at end minus amplitude at start.
def fiducial_amplitude(sig: np.ndarray, start: int | None, end: int | None) -> float:
    if start is None or end is None:
        return float("nan")
    if start < 0 or start >= len(sig) or end < 0 or end >= len(sig):
        return float("nan")
    return float(sig[end] - sig[start])


# Function: Safely divide two numeric feature values.
# Inputs: Numerator and denominator.
# Outputs: Ratio, or NaN when division is invalid.
def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


# Function: Calculate averaged waveform morphology features from ECG fiducials.
# Inputs: Clean ECG signal, R peaks, NeuroKit fiducials, sampling rate, prefix, and averaging flag.
# Outputs: Dictionary of waveform feature names to scalar values.
def from_waves(
    sig: np.ndarray,
    R_peaks: np.ndarray,
    fiducials: Dict[str, Any],
    sampling_rate: float,
    prefix: str = "ecg",
    average: bool = False,
) -> Dict[str, float] | Dict[int, Dict[str, float]]:
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    P_peaks = fiducials.get("ECG_P_Peaks", [])
    Q_peaks = fiducials.get("ECG_Q_Peaks", [])
    S_peaks = fiducials.get("ECG_S_Peaks", [])
    T_peaks = fiducials.get("ECG_T_Peaks", [])
    R_peaks = np.asarray(R_peaks, dtype=np.int64)

    features_waves: Dict[int, Dict[str, float]] = {}
    for beatno in range(len(R_peaks)):
        p = get_fiducial(P_peaks, beatno)
        q = get_fiducial(Q_peaks, beatno)
        r = int(R_peaks[beatno])
        s = get_fiducial(S_peaks, beatno)
        t = get_fiducial(T_peaks, beatno)

        a_qs = fiducial_amplitude(sig, q, s)
        a_qr = fiducial_amplitude(sig, q, r)
        features_waves[beatno] = {
            f"{prefix}_t_PR": fiducial_time(p, r, sampling_rate),
            f"{prefix}_t_QR": fiducial_time(q, r, sampling_rate),
            f"{prefix}_t_RS": fiducial_time(r, s, sampling_rate),
            f"{prefix}_t_RT": fiducial_time(r, t, sampling_rate),
            f"{prefix}_t_PQ": fiducial_time(p, q, sampling_rate),
            f"{prefix}_t_PS": fiducial_time(p, s, sampling_rate),
            f"{prefix}_t_PT": fiducial_time(p, t, sampling_rate),
            f"{prefix}_t_QS": fiducial_time(q, s, sampling_rate),
            f"{prefix}_t_QT": fiducial_time(q, t, sampling_rate),
            f"{prefix}_t_ST": fiducial_time(s, t, sampling_rate),
            f"{prefix}_a_PQ": fiducial_amplitude(sig, p, q),
            f"{prefix}_a_QR": a_qr,
            f"{prefix}_a_RS": fiducial_amplitude(sig, r, s),
            f"{prefix}_a_ST": fiducial_amplitude(sig, s, t),
            f"{prefix}_a_PS": fiducial_amplitude(sig, p, s),
            f"{prefix}_a_PT": fiducial_amplitude(sig, p, t),
            f"{prefix}_a_QS": a_qs,
            f"{prefix}_a_QT": fiducial_amplitude(sig, q, t),
            f"{prefix}_a_ST_QS": safe_ratio(fiducial_amplitude(sig, s, t), a_qs),
            f"{prefix}_a_RS_QR": safe_ratio(fiducial_amplitude(sig, r, s), a_qr),
        }

    if not average:
        return features_waves

    feature_names = [
        f"{prefix}_t_PR",
        f"{prefix}_t_QR",
        f"{prefix}_t_RS",
        f"{prefix}_t_RT",
        f"{prefix}_t_PQ",
        f"{prefix}_t_PS",
        f"{prefix}_t_PT",
        f"{prefix}_t_QS",
        f"{prefix}_t_QT",
        f"{prefix}_t_ST",
        f"{prefix}_a_PQ",
        f"{prefix}_a_QR",
        f"{prefix}_a_RS",
        f"{prefix}_a_ST",
        f"{prefix}_a_PS",
        f"{prefix}_a_PT",
        f"{prefix}_a_QS",
        f"{prefix}_a_QT",
        f"{prefix}_a_ST_QS",
        f"{prefix}_a_RS_QR",
    ]
    features_avg: Dict[str, float] = {}
    for key in feature_names:
        values = [beat_features.get(key, np.nan) for beat_features in features_waves.values()]
        features_avg[key] = safe_nanmean(values)
    return features_avg


# Function: Extract SignalMC-MED-style 54-dimensional single-lead ECG features.
# Inputs: One ECG lead signal, sampling frequency, and optional maximum feature count.
# Outputs: Dictionary of named ECG features matching the 54-feature design.
def extract_signal_mc_med_features(
    ecg_signal: np.ndarray,
    fs: int = 500,
    max_features: int | None = None,
) -> Dict[str, float]:
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64)

    if ecg_signal.ndim != 1:
        raise ValueError(f"Expected 1D ECG signal, got shape {ecg_signal.shape}")

    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    _, rpeaks_info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    rpeaks = np.asarray(rpeaks_info.get("ECG_R_Peaks", []), dtype=np.int64)

    feature_names = (
        [f"ecg_{name}" for name in FEATURES_RPEAKS]
        + [
            "ecg_t_PR",
            "ecg_t_QR",
            "ecg_t_RS",
            "ecg_t_RT",
            "ecg_t_PQ",
            "ecg_t_PS",
            "ecg_t_PT",
            "ecg_t_QS",
            "ecg_t_QT",
            "ecg_t_ST",
            "ecg_a_PQ",
            "ecg_a_QR",
            "ecg_a_RS",
            "ecg_a_ST",
            "ecg_a_PS",
            "ecg_a_PT",
            "ecg_a_QS",
            "ecg_a_QT",
            "ecg_a_ST_QS",
            "ecg_a_RS_QR",
        ]
    )

    if len(rpeaks) <= 3:
        base_features = {name: 0.0 for name in feature_names}
        hrv_features = {name: 0.0 for name in hrv_time_feature_names()}
        all_features = {**base_features, **hrv_features, "ecg_sqi_zhao2018": 0.0}
        return dict(list(all_features.items())[:max_features]) if max_features is not None else all_features

    try:
        _, waves_peak = nk.ecg_delineate(
            ecg_clean,
            rpeaks_info,
            sampling_rate=fs,
            method="peak",
        )
    except Exception:
        waves_peak = {}

    features_rpeaks = from_Rpeaks(ecg_clean, rpeaks, sampling_rate=fs, average=True)
    features_waveforms = from_waves(ecg_clean, rpeaks, waves_peak, sampling_rate=fs, average=True)
    features_hrv = extract_hrv_time_features(rpeaks, fs)
    sqi = extract_sqi_feature(ecg_clean, rpeaks, fs)

    all_features = {
        **{key: float(np.nan_to_num(value)) for key, value in features_rpeaks.items()},
        **{key: float(np.nan_to_num(value)) for key, value in features_waveforms.items()},
        **features_hrv,
        "ecg_sqi_zhao2018": sqi,
    }

    if max_features is not None:
        return dict(list(all_features.items())[:max_features])
    return all_features


# Function: Return the expected NeuroKit HRV time-domain feature names.
# Inputs: None.
# Outputs: Ordered list of HRV feature names.
def hrv_time_feature_names() -> list[str]:
    return [
        "HRV_MeanNN",
        "HRV_SDNN",
        "HRV_SDANN1",
        "HRV_SDNNI1",
        "HRV_SDANN2",
        "HRV_SDNNI2",
        "HRV_SDANN5",
        "HRV_SDNNI5",
        "HRV_RMSSD",
        "HRV_SDSD",
        "HRV_CVNN",
        "HRV_CVSD",
        "HRV_MedianNN",
        "HRV_MadNN",
        "HRV_MCVNN",
        "HRV_IQRNN",
        "HRV_SDRMSSD",
        "HRV_Prc20NN",
        "HRV_Prc80NN",
        "HRV_pNN50",
        "HRV_pNN20",
        "HRV_MinNN",
        "HRV_MaxNN",
        "HRV_HTI",
        "HRV_TINN",
    ]


# Function: Extract named NeuroKit HRV time-domain features.
# Inputs: R-peak sample locations and sampling frequency.
# Outputs: Dictionary of HRV feature names to numeric values.
def extract_hrv_time_features(rpeaks: np.ndarray, fs: int) -> Dict[str, float]:
    try:
        hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
    except Exception:
        return {name: 0.0 for name in hrv_time_feature_names()}

    values: Dict[str, float] = {}
    for name in hrv_time_feature_names():
        if name in hrv:
            values[name] = float(np.nan_to_num(hrv[name].iloc[0]))
        else:
            values[name] = 0.0
    return values


# Function: Extract a scalar Zhao 2018 ECG signal quality feature.
# Inputs: Clean ECG signal, R-peak locations, and sampling frequency.
# Outputs: Quality score where 1 is excellent, 0.5 acceptable, and 0 unacceptable.
def extract_sqi_feature(ecg_clean: np.ndarray, rpeaks: np.ndarray, fs: int) -> float:
    try:
        sqi = nk.ecg_quality(
            ecg_clean,
            rpeaks=rpeaks,
            sampling_rate=fs,
            method="zhao2018",
            approach="fuzzy",
        )
        return float({"Excellent": 1.0, "Unacceptable": 0.0, "Barely acceptable": 0.5}.get(sqi, 0.0))
    except Exception:
        return 0.0
