import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "outputs/ecgqa_csfm_preview_2000_sv.jsonl"

# -----------------------------
# LOAD DATA
# -----------------------------
embeddings = []
questions = []
labels = []

label_map = {
    "no": 0,
    "yes": 1,
    "not sure": 2
}

with open(DATA_PATH, "r") as f:
    for line in f:
        row = json.loads(line)

        # Get label
        answer = row["answer"]
        if isinstance(answer, list):
            answer = answer[0]

        answer = answer.lower()

        if answer not in label_map:
            continue

        if answer == "not sure":
            continue

        embeddings.append(row["embedding"])
        questions.append(row["question"])
        labels.append(label_map[answer])

X_emb = np.array(embeddings)
y = np.array(labels)

print("Dataset size:", X_emb.shape)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train_emb, X_test_emb, y_train, y_test, q_train, q_test = train_test_split(
    X_emb, y, questions, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL 1: EMBEDDING ONLY
# -----------------------------
print("\n=== MODEL 1: EMBEDDING ONLY ===")

clf_emb = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_emb.fit(X_train_emb, y_train)

y_pred_emb = clf_emb.predict(X_test_emb)

print("Accuracy:", accuracy_score(y_test, y_pred_emb))
print(classification_report(y_test, y_pred_emb))

# -----------------------------

# MODEL 2: EMBEDDING + QUESTION (Sentence Embeddings)

# -----------------------------

print("\n=== MODEL 2: EMBEDDING + QUESTION (SentenceTransformer) ===")

print("Loading sentence transformer model...")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding training questions...")

X_train_q = sentence_model.encode(q_train)

print("Encoding test questions...")

X_test_q = sentence_model.encode(q_test)

# Concatenate
X_train_combined = np.concatenate([X_train_emb, X_train_q], axis=1)
X_test_combined = np.concatenate([X_test_emb, X_test_q], axis=1)

clf_combined = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_combined.fit(X_train_combined, y_train)

y_pred_combined = clf_combined.predict(X_test_combined)

print("Accuracy:", accuracy_score(y_test, y_pred_combined))
print(classification_report(y_test, y_pred_combined))
print(confusion_matrix(y_test, y_pred_combined))