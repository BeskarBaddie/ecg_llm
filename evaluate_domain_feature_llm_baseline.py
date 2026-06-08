from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.ecg_feature_extractor import (
    LEADS_12,
    extract_signal_mc_med_features,
    extract_single_lead_domain_features,
)
from src.ptbxl_loader import load_ecg_signal, load_ptbxl_metadata


VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/domain_feature_llm_baseline_results.json")
PREDICTIONS_PATH = Path("outputs/domain_feature_llm_baseline_predictions.jsonl")
DOMAIN_CLASSIFIER_BUNDLE_PATH = Path("outputs/domain_feature_classifier_signalmc_leadII_54_logreg_models.joblib")

FEATURE_DESCRIPTIONS = {
    "ecg_a_R": "Average R-peak amplitude in the selected lead.",
    "ecg_RR0": "RR interval before the current beat, in seconds.",
    "ecg_RR1": "RR interval after the current beat, in seconds.",
    "ecg_RR2": "Following RR interval after RR1, in seconds.",
    "ecg_RRm": "Mean of neighboring RR intervals, in seconds.",
    "ecg_RR_0_1": "Ratio RR0/RR1; values near 1 suggest local rhythm regularity.",
    "ecg_RR_2_1": "Ratio RR2/RR1; values near 1 suggest local rhythm regularity.",
    "ecg_RR_m_1": "Ratio mean RR/RR1; values near 1 suggest local rhythm regularity.",
    "ecg_t_PR": "Time from P peak to R peak, in seconds; approximates PR timing.",
    "ecg_t_QR": "Time from Q peak to R peak, in seconds.",
    "ecg_t_RS": "Time from R peak to S peak, in seconds.",
    "ecg_t_RT": "Time from R peak to T peak, in seconds.",
    "ecg_t_PQ": "Time from P peak to Q peak, in seconds.",
    "ecg_t_PS": "Time from P peak to S peak, in seconds.",
    "ecg_t_PT": "Time from P peak to T peak, in seconds.",
    "ecg_t_QS": "Time from Q peak to S peak, in seconds; partial QRS timing.",
    "ecg_t_QT": "Time from Q peak to T peak, in seconds; approximates QT timing.",
    "ecg_t_ST": "Time from S peak to T peak, in seconds.",
    "ecg_a_PQ": "Amplitude difference from P peak to Q peak.",
    "ecg_a_QR": "Amplitude difference from Q peak to R peak.",
    "ecg_a_RS": "Amplitude difference from R peak to S peak.",
    "ecg_a_ST": "Amplitude difference from S peak to T peak.",
    "ecg_a_PS": "Amplitude difference from P peak to S peak.",
    "ecg_a_PT": "Amplitude difference from P peak to T peak.",
    "ecg_a_QS": "Amplitude difference from Q peak to S peak.",
    "ecg_a_QT": "Amplitude difference from Q peak to T peak.",
    "ecg_a_ST_QS": "Ratio of ST amplitude difference to QS amplitude difference.",
    "ecg_a_RS_QR": "Ratio of RS amplitude difference to QR amplitude difference.",
    "HRV_MeanNN": "Mean normal-to-normal interval in milliseconds.",
    "HRV_SDNN": "Standard deviation of normal-to-normal intervals in milliseconds.",
    "HRV_SDANN1": "SDANN HRV statistic over 1-minute segments; may be zero for short recordings.",
    "HRV_SDNNI1": "SDNN index over 1-minute segments; may be zero for short recordings.",
    "HRV_SDANN2": "SDANN HRV statistic over 2-minute segments; may be zero for short recordings.",
    "HRV_SDNNI2": "SDNN index over 2-minute segments; may be zero for short recordings.",
    "HRV_SDANN5": "SDANN HRV statistic over 5-minute segments; may be zero for short recordings.",
    "HRV_SDNNI5": "SDNN index over 5-minute segments; may be zero for short recordings.",
    "HRV_RMSSD": "Root mean square of successive NN interval differences in milliseconds.",
    "HRV_SDSD": "Standard deviation of successive NN interval differences in milliseconds.",
    "HRV_CVNN": "Coefficient of variation of NN intervals.",
    "HRV_CVSD": "Coefficient of variation based on RMSSD.",
    "HRV_MedianNN": "Median normal-to-normal interval in milliseconds.",
    "HRV_MadNN": "Median absolute deviation of NN intervals.",
    "HRV_MCVNN": "Median-based coefficient of variation of NN intervals.",
    "HRV_IQRNN": "Interquartile range of NN intervals.",
    "HRV_SDRMSSD": "Ratio of SDNN to RMSSD.",
    "HRV_Prc20NN": "20th percentile of NN intervals in milliseconds.",
    "HRV_Prc80NN": "80th percentile of NN intervals in milliseconds.",
    "HRV_pNN50": "Percentage of successive NN interval differences greater than 50 ms.",
    "HRV_pNN20": "Percentage of successive NN interval differences greater than 20 ms.",
    "HRV_MinNN": "Minimum NN interval in milliseconds.",
    "HRV_MaxNN": "Maximum NN interval in milliseconds.",
    "HRV_HTI": "HRV triangular index.",
    "HRV_TINN": "Triangular interpolation of NN interval histogram.",
    "ecg_sqi_zhao2018": "Signal quality index from Zhao 2018; higher values indicate better quality.",
}


# Function: Load a JSONL dataset into memory.
# Inputs: Path to a JSONL file.
# Outputs: List of parsed row dictionaries.
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc

    return rows


# Function: Write prediction rows to a JSONL file.
# Inputs: Output path and serializable prediction dictionaries.
# Outputs: None; writes the file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Parse an LLM response into a binary yes/no prediction.
# Inputs: Raw model response text.
# Outputs: Tuple of numeric label and normalized answer string.
def parse_yes_no_response(text: str) -> tuple[int, str]:
    cleaned = text.strip().lower()

    if cleaned.startswith("yes"):
        return 1, "yes"

    if cleaned.startswith("no"):
        return 0, "no"

    tokens = cleaned.replace(".", " ").replace(",", " ").split()
    if tokens:
        if tokens[0] == "yes":
            return 1, "yes"
        if tokens[0] == "no":
            return 0, "no"

    return -1, "unknown"


# Function: Load an ECG and extract selected single-lead domain features.
# Inputs: ECG ID, PTB-XL metadata, lead name, feature set name, feature count, and cache.
# Outputs: Feature payload with features, source sampling rate, and signal metadata.
def load_or_extract_domain_features(
    ecg_id: int,
    metadata: Any,
    lead_name: str,
    feature_set: str,
    max_features: int,
    feature_cache: Dict[tuple[int, str, str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = (ecg_id, lead_name, feature_set, max_features)
    if cache_key in feature_cache:
        return feature_cache[cache_key]

    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="hr")
        fs = 500
        source = "hr"
    except Exception:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
        fs = 100
        source = "lr"

    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] != len(LEADS_12):
        raise ValueError(f"Expected 12-lead ECG, got shape {signal.shape}")

    lead_idx = LEADS_12.index(lead_name)
    if feature_set == "signalmc":
        features = extract_signal_mc_med_features(
            signal[lead_idx],
            fs=fs,
            max_features=max_features,
        )
    elif feature_set == "compact":
        features = extract_single_lead_domain_features(
            signal[lead_idx],
            fs=fs,
            max_features=max_features,
        )
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    feature_cache[cache_key] = {
        "features": features,
        "lead_name": lead_name,
        "feature_set": feature_set,
        "fs": fs,
        "source": source,
        "signal_shape": list(signal.shape),
    }
    return feature_cache[cache_key]


# Function: Load trained domain-feature classifiers from a joblib bundle.
# Inputs: Path to a bundle produced by train_domain_feature_classifiers.py, or None.
# Outputs: Bundle dictionary, or None when no path is supplied.
def load_domain_classifier_bundle(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Domain classifier bundle not found: {path}")
    bundle = joblib.load(path)
    required_keys = {"models", "feature_names"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Domain classifier bundle is missing keys: {sorted(missing)}")
    return bundle


# Function: Convert extracted feature dictionary into the matrix expected by a classifier bundle.
# Inputs: Feature payload and bundle feature names.
# Outputs: Single-row float feature matrix.
def feature_item_to_classifier_matrix(
    feature_item: Dict[str, Any],
    feature_names: List[str],
) -> np.ndarray:
    values = []
    features = feature_item["features"]
    for name in feature_names:
        try:
            value = float(features.get(name, 0.0))
        except Exception:
            value = 0.0
        values.append(float(np.nan_to_num(value)))
    return np.array([values], dtype=np.float32)


# Function: Score all available domain-feature classifiers for one ECG.
# Inputs: Feature payload and loaded classifier bundle.
# Outputs: Mapping from SCP code to positive-class probability.
def score_domain_feature_classifiers(
    feature_item: Dict[str, Any],
    classifier_bundle: Dict[str, Any] | None,
) -> Dict[str, float]:
    if classifier_bundle is None:
        return {}

    feature_names = list(classifier_bundle["feature_names"])
    X = feature_item_to_classifier_matrix(feature_item, feature_names)
    probabilities: Dict[str, float] = {}

    for code, model in sorted(classifier_bundle["models"].items()):
        if not hasattr(model, "predict_proba"):
            continue
        probabilities[str(code)] = float(model.predict_proba(X)[0, 1])

    return probabilities


# Function: Convert ECG domain features into an LLM prompt.
# Inputs: Feature payload, ECG-QA question, and feature provenance text.
# Outputs: Strict binary yes/no prompt string.
def build_domain_feature_prompt(
    feature_item: Dict[str, Any],
    question: str,
) -> str:
    lines = [
        "You are answering a binary ECG question.",
        "Use the extracted single-lead ECG domain features below as the ECG representation.",
        "The features are a limited baseline, not a full diagnostic ECG interpretation.",
        "",
        f"Lead used: {feature_item['lead_name']}",
        f"Feature set: {feature_item.get('feature_set', 'unknown')}",
        f"Sampling frequency: {feature_item['fs']} Hz",
        "",
        "ECG domain features:",
    ]

    for name, value in feature_item["features"].items():
        try:
            numeric_value = float(value)
            formatted = "NA" if not np.isfinite(numeric_value) else f"{numeric_value:.4f}"
        except Exception:
            formatted = "NA"
        lines.append(f"- {name}: {formatted}")

    lines.extend(
        [
            "",
            f"Question: {question}",
            "Valid answers: yes, no.",
            "Return exactly one word: yes or no.",
            "Do not include any explanation.",
        ]
    )
    return "\n".join(lines)


# Function: Format a numeric feature value for clinical text.
# Inputs: Feature dictionary, feature name, unit suffix, and decimal precision.
# Outputs: Readable string value, or "not available" when missing/non-finite.
def format_feature_value(
    features: Dict[str, Any],
    name: str,
    unit: str = "",
    precision: int = 3,
) -> str:
    try:
        value = float(features[name])
    except Exception:
        return "not available"

    if not np.isfinite(value):
        return "not available"

    return f"{value:.{precision}f}{unit}"


# Function: Convert the first 10 SignalMC-style features into clinical context sentences.
# Inputs: Feature payload containing extracted ECG domain features.
# Outputs: List of plain-language clinical interpretation sentences.
def interpret_signal_mc_first_10_features(feature_item: Dict[str, Any]) -> List[str]:
    features = feature_item["features"]
    sentences: List[str] = []

    rr_values = []
    for name in ["ecg_RR0", "ecg_RR1", "ecg_RR2"]:
        try:
            value = float(features[name])
            if np.isfinite(value) and value > 0:
                rr_values.append(value)
        except Exception:
            pass

    if rr_values:
        mean_rr = float(np.mean(rr_values))
        heart_rate = 60.0 / mean_rr
        if heart_rate < 60:
            heart_rate_text = "slow"
        elif heart_rate > 100:
            heart_rate_text = "fast"
        else:
            heart_rate_text = "within the usual adult resting range"
        sentences.append(
            f"The local RR intervals are around {mean_rr:.3f} seconds, corresponding to an estimated heart rate of {heart_rate:.1f} bpm, which is {heart_rate_text}."
        )
    else:
        sentences.append("The RR interval features are not available, so rhythm rate cannot be estimated from these features.")

    rr_ratio_names = ["ecg_RR_0_1", "ecg_RR_2_1", "ecg_RR_m_1"]
    rr_ratios = []
    for name in rr_ratio_names:
        try:
            value = float(features[name])
            if np.isfinite(value):
                rr_ratios.append(value)
        except Exception:
            pass

    if rr_ratios:
        max_ratio_deviation = max(abs(value - 1.0) for value in rr_ratios)
        rhythm_text = "fairly regular" if max_ratio_deviation < 0.10 else "irregular"
        sentences.append(
            f"The RR interval ratios are close to 1 when regular; here the largest deviation from 1 is {max_ratio_deviation:.3f}, suggesting a {rhythm_text} rhythm over the sampled beats."
        )
    else:
        sentences.append("The RR ratio features are not available, so rhythm regularity cannot be assessed from these features.")

    pr_value = features.get("ecg_t_PR")
    try:
        pr_float = float(pr_value)
        if np.isfinite(pr_float):
            if pr_float < 0.12:
                pr_text = "shorter than the usual adult PR interval range"
            elif pr_float > 0.20:
                pr_text = "longer than the usual adult PR interval range"
            else:
                pr_text = "within the usual adult PR interval range"
            sentences.append(f"The P-to-R timing feature is {pr_float:.3f} seconds, which is {pr_text}.")
    except Exception:
        sentences.append("The P-to-R timing feature is not available.")

    qr_value = features.get("ecg_t_QR")
    try:
        qr_float = float(qr_value)
        if np.isfinite(qr_float):
            sentences.append(
                f"The Q-to-R timing feature is {qr_float:.3f} seconds; this gives partial QRS morphology context but does not measure the full QRS duration by itself."
            )
    except Exception:
        sentences.append("The Q-to-R timing feature is not available.")

    sentences.append(
        f"The average R-peak amplitude in lead {feature_item['lead_name']} is {format_feature_value(features, 'ecg_a_R', precision=3)} in the signal's amplitude units."
    )
    sentences.append(
        "These 10 single-lead features mainly describe rhythm regularity, local timing, and R-wave amplitude; they are limited and may not capture multi-lead diagnoses such as infarction, LVH, or bundle branch block."
    )

    return sentences


# Function: Convert interpreted ECG domain features into an LLM prompt.
# Inputs: Feature payload and ECG-QA question.
# Outputs: Strict binary yes/no prompt string using clinical sentences rather than raw-only values.
def build_interpreted_domain_feature_prompt(
    feature_item: Dict[str, Any],
    question: str,
) -> str:
    lines = [
        "You are answering a binary ECG question.",
        "Use the interpreted single-lead ECG domain features below as the ECG representation.",
        "The interpretation is a limited baseline and is not a full 12-lead clinical ECG report.",
        "",
        f"Lead used: {feature_item['lead_name']}",
        f"Feature set: {feature_item.get('feature_set', 'unknown')}",
        f"Sampling frequency: {feature_item['fs']} Hz",
        "",
        "Interpreted ECG feature summary:",
    ]

    for sentence in interpret_signal_mc_first_10_features(feature_item):
        lines.append(f"- {sentence}")

    lines.extend(
        [
            "",
            f"Question: {question}",
            "Valid answers: yes, no.",
            "Return exactly one word: yes or no.",
            "Do not include any explanation.",
        ]
    )
    return "\n".join(lines)


# Function: Convert described ECG domain features into a zero-shot LLM prompt.
# Inputs: Feature payload and ECG-QA question.
# Outputs: Strict binary yes/no prompt listing feature descriptions and values, with no classifier evidence.
def build_described_domain_feature_prompt(
    feature_item: Dict[str, Any],
    question: str,
) -> str:
    lines = [
        "You are answering a binary ECG question.",
        "Use only the single-lead ECG domain features below as the ECG representation.",
        "No trained ECG classifier output is provided.",
        "The ECG representation is limited to one lead and may miss findings that require other leads.",
        "",
        f"Lead used: {feature_item['lead_name']}",
        f"Feature set: {feature_item.get('feature_set', 'unknown')}",
        f"Sampling frequency: {feature_item['fs']} Hz",
        "",
        "Feature descriptions and values:",
    ]

    for name, value in feature_item["features"].items():
        description = FEATURE_DESCRIPTIONS.get(name, "Extracted ECG domain feature.")
        try:
            numeric_value = float(value)
            formatted = "NA" if not np.isfinite(numeric_value) else f"{numeric_value:.4f}"
        except Exception:
            formatted = "NA"
        lines.append(f"- {name}: {formatted}. Meaning: {description}")

    lines.extend(
        [
            "",
            f"Question: {question}",
            "Valid answers: yes, no.",
            "Return exactly one word: yes or no.",
            "Do not include any explanation.",
        ]
    )
    return "\n".join(lines)


# Function: Convert domain-feature classifier scores and interpreted features into an LLM prompt.
# Inputs: Feature payload, ECG-QA row, and per-code classifier probabilities.
# Outputs: Strict binary yes/no prompt string grounded in handcrafted ECG features.
def build_classifier_interpreted_domain_prompt(
    feature_item: Dict[str, Any],
    row: Dict[str, Any],
    classifier_probabilities: Dict[str, float],
) -> str:
    target_code = str(row["target_scp_code"])
    target_probability = classifier_probabilities.get(target_code)
    if target_probability is None:
        target_probability_text = "not available"
        target_prediction_text = "not available"
    else:
        target_probability_text = f"{target_probability:.3f}"
        target_prediction_text = "present" if target_probability >= 0.5 else "absent"

    lines = [
        "You are answering a binary ECG question.",
        "Use the handcrafted ECG domain-feature evidence below as the ECG representation.",
        "The ECG representation is limited to one lead and derived features, not the full raw 12-lead ECG.",
        "",
        f"Lead used: {feature_item['lead_name']}",
        f"Feature set: {feature_item.get('feature_set', 'unknown')}",
        f"Sampling frequency: {feature_item['fs']} Hz",
        f"Question target SCP code: {target_code}",
        "",
        "Domain-feature classifier evidence:",
        f"- Target classifier probability for {target_code}: {target_probability_text}",
        f"- At threshold 0.5, the target classifier predicts: {target_prediction_text}",
    ]

    if classifier_probabilities:
        lines.append("- Other classifier probabilities:")
        for code, probability in sorted(classifier_probabilities.items()):
            if code == target_code:
                continue
            lines.append(f"  - {code}: {probability:.3f}")

    lines.extend(
        [
            "",
            "Interpreted ECG feature summary:",
        ]
    )
    for sentence in interpret_signal_mc_first_10_features(feature_item):
        lines.append(f"- {sentence}")

    lines.extend(
        [
            "",
            f"Question: {row['question']}",
            "Valid answers: yes, no.",
            "Return exactly one word: yes or no.",
            "Do not include any explanation.",
        ]
    )
    return "\n".join(lines)


# Function: Call Ollama chat completion for one prompt.
# Inputs: Ollama model name, prompt, temperature, and max generated tokens.
# Outputs: Raw assistant response text.
def call_ollama(
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    host: str,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to Ollama at {host}") from exc

    return str(response_data["message"]["content"])


# Function: Compute validation metrics separately for each SCP code.
# Inputs: Validation rows, true labels, and predicted labels.
# Outputs: Nested dictionary of per-code metrics and counts.
def per_code_metrics(
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    by_code: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_code[str(row.get("target_scp_code", "UNKNOWN"))].append(idx)

    metrics: Dict[str, Dict[str, Any]] = {}
    for code, indices in sorted(by_code.items()):
        code_true = y_true[indices]
        code_pred = y_pred[indices]
        metrics[code] = {
            "n": int(len(indices)),
            "accuracy": float(accuracy_score(code_true, code_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(code_true, code_pred))
            if len(np.unique(code_true)) > 1
            else None,
            "macro_f1": float(f1_score(code_true, code_pred, average="macro", zero_division=0)),
            "label_counts": dict(Counter(int(x) for x in code_true)),
            "prediction_counts": dict(Counter(int(x) for x in code_pred)),
        }

    return metrics


# Function: Compute overall and per-code classification metrics.
# Inputs: Validation rows, true labels, and predicted labels.
# Outputs: Dictionary of metrics, reports, and confusion matrix.
def evaluate_predictions(
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "invalid_predictions": int(np.sum(y_pred == -1)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "confusion_matrix_labels": ["no", "yes", "unknown"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, -1]).tolist(),
        "labels": {
            "no": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "yes": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": report,
        "per_code": per_code_metrics(rows, y_true, y_pred),
    }


# Function: Run the single-lead domain-feature-to-LLM baseline.
# Inputs: Command-line arguments.
# Outputs: Result and prediction files written to disk.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=4)
    parser.add_argument("--lead", type=str, default="II", choices=LEADS_12)
    parser.add_argument("--feature-set", type=str, default="signalmc", choices=["signalmc", "compact"])
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="raw",
        choices=["raw", "interpreted", "described_features", "classifier_interpreted"],
    )
    parser.add_argument(
        "--domain-classifier-bundle-path",
        type=Path,
        default=None,
        help=(
            "Optional trained domain-feature classifier bundle. Required for "
            "--prompt-style classifier_interpreted."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()

    rows = load_jsonl(args.val_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        raise RuntimeError("No validation rows to evaluate.")

    if args.prompt_style == "classifier_interpreted" and args.domain_classifier_bundle_path is None:
        args.domain_classifier_bundle_path = DOMAIN_CLASSIFIER_BUNDLE_PATH

    classifier_bundle = load_domain_classifier_bundle(args.domain_classifier_bundle_path)

    metadata = load_ptbxl_metadata()
    feature_cache: Dict[tuple[int, str, str, int], Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = args.predictions_path.open("w", encoding="utf-8")

    try:
        for idx, row in enumerate(rows, start=1):
            ecg_id = int(row["ecg_id"])
            true_label = int(row["label"])

            try:
                feature_item = load_or_extract_domain_features(
                    ecg_id,
                    metadata,
                    lead_name=args.lead,
                    feature_set=args.feature_set,
                    max_features=args.max_features,
                    feature_cache=feature_cache,
                )
                classifier_probabilities = score_domain_feature_classifiers(feature_item, classifier_bundle)
                if args.prompt_style == "classifier_interpreted":
                    prompt = build_classifier_interpreted_domain_prompt(
                        feature_item,
                        row,
                        classifier_probabilities,
                    )
                elif args.prompt_style == "described_features":
                    prompt = build_described_domain_feature_prompt(feature_item, row["question"])
                elif args.prompt_style == "interpreted":
                    prompt = build_interpreted_domain_feature_prompt(feature_item, row["question"])
                else:
                    prompt = build_domain_feature_prompt(feature_item, row["question"])
                raw_response = call_ollama(
                    args.model,
                    prompt,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    host=args.ollama_host,
                )
                pred_label, pred_answer = parse_yes_no_response(raw_response)
                error = None
            except Exception as exc:
                feature_item = {}
                classifier_probabilities = {}
                prompt = None
                raw_response = ""
                pred_label = -1
                pred_answer = "unknown"
                error = str(exc)

            prediction = {
                "method": "domain_feature_llm",
                "model": args.model,
                "ecg_id": ecg_id,
                "split": row.get("split"),
                "target_scp_code": row.get("target_scp_code"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "true_label": true_label,
                "pred_label": int(pred_label),
                "pred_answer": pred_answer,
                "raw_response": raw_response,
                "correct": bool(pred_label == true_label),
                "lead": feature_item.get("lead_name"),
                "feature_set": feature_item.get("feature_set"),
                "prompt_style": args.prompt_style,
                "feature_source": feature_item.get("source"),
                "fs": feature_item.get("fs"),
                "features": feature_item.get("features"),
                "domain_classifier_probabilities": classifier_probabilities,
                "signal_shape": feature_item.get("signal_shape"),
                "prompt": prompt,
                "error": error,
            }
            predictions.append(prediction)
            y_true.append(true_label)
            y_pred.append(int(pred_label))

            prediction_file.write(json.dumps(prediction) + "\n")
            prediction_file.flush()

            print(
                f"[{idx}/{len(rows)}] ecg_id={ecg_id} code={row.get('target_scp_code')} "
                f"true={row.get('answer')} pred={pred_answer}",
                flush=True,
            )
    finally:
        prediction_file.close()

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    results = {
        "method": "domain_feature_llm",
        "model": args.model,
        "ollama_host": args.ollama_host,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
        "val_path": str(args.val_path),
        "n_rows": len(rows),
        "n_unique_ecgs": len({row["ecg_id"] for row in rows}),
        "feature_cache_size": len(feature_cache),
        "lead": args.lead,
        "max_features": args.max_features,
        "feature_set": args.feature_set,
        "prompt_style": args.prompt_style,
        "domain_classifier_bundle_path": str(args.domain_classifier_bundle_path)
        if args.domain_classifier_bundle_path is not None
        else None,
        "metrics": evaluate_predictions(rows, y_true_arr, y_pred_arr),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_jsonl(args.predictions_path, predictions)

    metrics = results["metrics"]
    print("\n=== Domain Feature LLM Baseline ===")
    print("Accuracy:", metrics["accuracy"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Invalid predictions:", metrics["invalid_predictions"])
    print(f"\nSaved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
