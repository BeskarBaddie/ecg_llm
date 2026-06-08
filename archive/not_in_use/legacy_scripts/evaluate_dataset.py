import json
import numpy as np
from collections import Counter

DATA_PATH = "outputs/ecgqa_csfm_preview_1000.jsonl"

total = 0
yes = 0
no = 0
invalid = 0

embedding_lengths = []
question_types = Counter()
attribute_types = Counter()

with open(DATA_PATH, "r") as f:
    for line in f:
        row = json.loads(line)
        total += 1

        # Answer stats
        ans = row["answer"]
        if isinstance(ans, list):
            ans = ans[0]

        if ans == "yes":
            yes += 1
        elif ans == "no":
            no += 1
        else:
            invalid += 1

        # Embedding stats
        emb = np.array(row["embedding"])
        embedding_lengths.append(len(emb))

        if np.isnan(emb).any() or np.isinf(emb).any():
            print(f"⚠️ Bad embedding at ecg_id {row['ecg_id']}")

        # Distribution stats
        question_types[row["question_type"]] += 1
        attribute_types[row["attribute_type"]] += 1


print("\n=== DATASET SUMMARY ===")
print("Total samples:", total)
print("Yes:", yes)
print("No:", no)
print("Invalid:", invalid)

print("\n=== EMBEDDINGS ===")
print("Embedding dim (unique):", set(embedding_lengths))

print("\n=== QUESTION TYPES ===")
print(question_types)

print("\n=== ATTRIBUTE TYPES ===")
print(attribute_types)