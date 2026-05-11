from __future__ import annotations

import random
from typing import Any

import numpy as np
import ollama

from src.config import ECG_QA_TRAIN
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal
from src.ecg_feature_extractor import extract_12lead_basic_features
from src.ecg_prompt_builder import build_ecg_prompt


# -----------------------------
# LOAD DATA
# -----------------------------
samples = load_ecgqa_json(ECG_QA_TRAIN)
metadata = load_ptbxl_metadata()

# -----------------------------
# FILTER (clean task)
# -----------------------------
valid_answers = {"yes", "no", "not sure"}
filtered_samples = []

for s in samples:
    if s.get("question_type") != "single-verify":
        continue

    answer = s.get("answer")
    if isinstance(answer, list):
        if len(answer) == 0:
            continue
        answer = answer[0]

    answer = str(answer).strip().lower()

    if answer in valid_answers:
        s["clean_answer"] = answer
        filtered_samples.append(s)

print("Filtered dataset size:", len(filtered_samples))

if len(filtered_samples) == 0:
    raise RuntimeError("No valid samples found after filtering.")


# -----------------------------
# RANDOM SAMPLE
# -----------------------------
N = 50  # number of test samples
N = min(N, len(filtered_samples))

random.seed(42)
test_samples = random.sample(filtered_samples, N)


# -----------------------------
# EVALUATION LOOP
# -----------------------------
correct = 0
total = 0

for i, sample in enumerate(test_samples, start=1):
    ecg_id = int(sample["ecg_id"][0])
    question = sample["question"]
    true_answer = sample["clean_answer"]

    # Try HR first, then LR fallback
    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="hr")
        fs = 500
        print(f"\n[{i}] Loaded HR signal for ECG ID {ecg_id}")
    except Exception as e_hr:
        print(f"\n[{i}] HR load failed, trying LR fallback for ECG ID {ecg_id}: {e_hr}")
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
        fs = 100
        print(f"[{i}] Loaded LR signal for ECG ID {ecg_id}")

    signal = np.asarray(signal, dtype=np.float32)

    # Extract structured ECG features
    feature_dict = extract_12lead_basic_features(signal, fs=fs)

    # Build prompt
    prompt = build_ecg_prompt(
        feature_dict=feature_dict,
        question=question,
        allowed_answers=("yes", "no", "not sure"),
    )

    print("\n================ PROMPT ================\n")
    print(prompt[:3000])
    print("\n========================================\n")

    # Ask LLM
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
    )

    pred = response["message"]["content"].strip().lower()

    # Clean prediction
    if pred.startswith("yes"):
        pred = "yes"
    elif pred.startswith("no"):
        pred = "no"
    elif "not sure" in pred or pred.startswith("not"):
        pred = "not sure"
    else:
        pred = "unknown"

    print(f"Q: {question}")
    print("True:", true_answer)
    print("Pred:", pred)

    if pred == true_answer:
        correct += 1

    total += 1


# -----------------------------
# FINAL METRICS
# -----------------------------
accuracy = correct / total if total > 0 else 0.0

print("\n======================")
print("FINAL RESULTS")
print("======================")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)