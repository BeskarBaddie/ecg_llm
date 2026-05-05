import random
import ollama
import numpy as np

from src.config import ECG_QA_TRAIN
from src.ecgqa_loader import load_ecgqa_json
from src.ptbxl_loader import load_ptbxl_metadata, load_ecg_signal
from ecg_to_text import ecg_to_text


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


# -----------------------------
# RANDOM SAMPLE
# -----------------------------
N = 50  # number of test samples

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

    try:
        signal = load_ecg_signal(ecg_id, metadata=metadata, prefer="lr")
    except:
        continue

    ecg_description = ecg_to_text(signal)

    prompt = f"""
You are a cardiology assistant.

ECG summary:
{ecg_description}

Question:
{question}

Answer strictly with one word:
yes, no, or not sure.
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )

    pred = response["message"]["content"].strip().lower()

    # Clean prediction (sometimes LLM adds extra words)
    if "yes" in pred:
        pred = "yes"
    elif "no" in pred:
        pred = "no"
    elif "not" in pred:
        pred = "not sure"
    else:
        pred = "unknown"

    print(f"\n[{i}] ECG ID: {ecg_id}")
    print("Q:", question)
    print("ECG Summary:", ecg_description)
    print("True:", true_answer)
    print("Pred:", pred)

    if pred == true_answer:
        correct += 1

    total += 1


# -----------------------------
# FINAL METRICS
# -----------------------------
accuracy = correct / total if total > 0 else 0

print("\n======================")
print("FINAL RESULTS")
print("======================")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)