from __future__ import annotations

from typing import Dict


def format_feature_value(x, decimals=4):
    """
    Format numeric ECG features cleanly for prompting.
    """
    try:
        x = float(x)
    except Exception:
        return "NA"

    if x != x:  # NaN check
        return "NA"

    return f"{x:.{decimals}f}"


def build_ecg_prompt(
    feature_dict: Dict[str, Dict[str, float]],
    question: str,
    allowed_answers=("yes", "no", "not sure"),
):
    """
    Build a structured ECG prompt for the LLM.
    """

    lines = []

    lines.append("You are a cardiology assistant.")
    lines.append("You are answering a question about a 12-lead ECG.")
    lines.append("The ECG is represented using extracted ECG features.")
    lines.append("")

    lines.append("ECG FEATURES:")
    lines.append("")

    for lead_name, features in feature_dict.items():
        lines.append(f"Lead {lead_name}:")

        for feat_name, feat_value in features.items():
            value = format_feature_value(feat_value)
            lines.append(f"  {feat_name} = {value}")

        lines.append("")

    lines.append(f"Question: {question}")

    allowed = ", ".join(allowed_answers)
    lines.append(f"Answer using exactly one of: {allowed}.")

    return "\n".join(lines)