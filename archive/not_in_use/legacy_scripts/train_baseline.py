from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = Path("outputs/ecgqa_csfm_preview_2000_sv.jsonl")


# -----------------------------
# CLASSIFIER FACTORY
# -----------------------------
def build_classifier(name: str):
    name = name.lower().strip()

    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    if name == "mlp":
        # Simple nonlinear classifier; good next step after logistic regression.
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )

    if name == "svm":
        return LinearSVC(class_weight="balanced")

    raise ValueError(f"Unknown classifier '{name}'. Use one of: logreg, mlp, svm.")


def print_metrics(title: str, y_true, y_pred) -> None:
    print(f"\n=== {title} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset(path: Path):
    embeddings = []
    questions = []
    labels = []

    label_map = {
        "no": 0,
        "yes": 1,
    }

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            answer = row["answer"]
            if isinstance(answer, list):
                if len(answer) == 0:
                    continue
                answer = answer[0]

            answer = str(answer).strip().lower()

            if answer not in label_map:
                continue

            embeddings.append(row["embedding"])
            questions.append(row["question"])
            labels.append(label_map[answer])

    X_emb = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X_emb, questions, y


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        choices=["logreg", "mlp", "svm"],
        help="Classifier to use on top of the features.",
    )
    args = parser.parse_args()

    print(f"Using classifier: {args.classifier}")

    # Load data
    X_emb, questions, y = load_dataset(DATA_PATH)
    print("Dataset size:", X_emb.shape)

    # Train / test split
    X_train_emb, X_test_emb, y_train, y_test, q_train, q_test = train_test_split(
        X_emb,
        y,
        questions,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # MODEL 1: EMBEDDING ONLY
    # --------------------------------------------------
    scaler_emb = StandardScaler()
    X_train_emb_scaled = scaler_emb.fit_transform(X_train_emb)
    X_test_emb_scaled = scaler_emb.transform(X_test_emb)

    clf_emb = build_classifier(args.classifier)
    clf_emb.fit(X_train_emb_scaled, y_train)

    y_pred_emb = clf_emb.predict(X_test_emb_scaled)
    print_metrics("MODEL 1: EMBEDDING ONLY (TEST)", y_test, y_pred_emb)

    # --------------------------------------------------
    # MODEL 2: EMBEDDING + QUESTION
    # --------------------------------------------------
    print("\nLoading sentence transformer model...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding training questions...")
    X_train_q = sentence_model.encode(
        q_train,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Encoding test questions...")
    X_test_q = sentence_model.encode(
        q_test,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Question embedding shape:", X_train_q.shape)

    X_train_combined = np.concatenate([X_train_emb, X_train_q], axis=1)
    X_test_combined = np.concatenate([X_test_emb, X_test_q], axis=1)

    scaler_combined = StandardScaler()
    X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)
    X_test_combined_scaled = scaler_combined.transform(X_test_combined)

    clf_combined = build_classifier(args.classifier)
    clf_combined.fit(X_train_combined_scaled, y_train)

    y_pred_combined_test = clf_combined.predict(X_test_combined_scaled)
    y_pred_combined_train = clf_combined.predict(X_train_combined_scaled)

    print_metrics("MODEL 2: EMBEDDING + QUESTION (TEST)", y_test, y_pred_combined_test)
    print_metrics("MODEL 2: EMBEDDING + QUESTION (TRAIN)", y_train, y_pred_combined_train)

    print("\nNumber of test questions:", len(q_test))
    print("Number of train questions:", len(q_train))


if __name__ == "__main__":
    main()

#check the embeddings to see if they are normalised to see if we actually need standard scaling 
# might need to train a projector 