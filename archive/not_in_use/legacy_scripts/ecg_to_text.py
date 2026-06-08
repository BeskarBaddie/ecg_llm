import numpy as np
import neurokit2 as nk


def ecg_to_text(signal, fs=250):
    """
    Convert ECG signal into a more meaningful textual description.
    """

    description = []

    # Use lead II (common practice)
    ecg = signal[1]

    # Clean signal
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)

    # Detect peaks
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

    r_peaks = np.where(peaks["ECG_R_Peaks"] == 1)[0]

    if len(r_peaks) > 1:
        # Heart rate
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals)

        description.append(f"Estimated heart rate is {heart_rate:.1f} beats per minute")

        # Rhythm regularity
        rr_std = np.std(rr_intervals)

        if rr_std > 0.1:
            description.append("The rhythm appears irregular")
        else:
            description.append("The rhythm appears regular")

    else:
        description.append("Unable to reliably detect heart rhythm")

    # Signal variability
    signal_std = np.std(ecg_clean)

    if signal_std > 1.0:
        description.append("The ECG signal shows high variability")
    else:
        description.append("The ECG signal appears relatively stable")

    return ". ".join(description)