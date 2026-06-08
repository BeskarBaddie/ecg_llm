from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = Path("outputs/ecgqa_scp_binary_train.jsonl")
VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/ecgqa_embedding_baseline_results.json")
PREDICTIONS_PATH = Path("outputs/ecgqa_embedding_baseline_predictions.jsonl")
MODEL_BUNDLE_PATH = Path("outputs/ecgqa_embedding_baseline_models.joblib")


# Function: Load newline-delimited JSON rows from disk.
# Inputs: path to a JSONL dataset file.
# Outputs: list of parsed row dictionaries.
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


# Function: Keep only rows for one SCP code when requested.
# Inputs: dataset rows and optional target SCP code.
# Outputs: filtered rows, preserving original order.
def filter_rows_by_scp_code(
    rows: List[Dict[str, Any]],
    target_scp_code: str | None,
) -> List[Dict[str, Any]]:
    if target_scp_code is None:
        return rows

    target_scp_code = target_scp_code.strip().upper()
    return [row for row in rows if str(row.get("target_scp_code", "")).upper() == target_scp_code]


# Function: Limit training rows per exact question while preserving the validation set.
# Inputs: dataset rows, optional max rows per question, and random seed.
# Outputs: sampled rows with at most max rows for each question.
def limit_rows_per_question(
    rows: List[Dict[str, Any]],
    max_rows_per_question: int | None,
    random_state: int,
) -> List[Dict[str, Any]]:
    if max_rows_per_question is None:
        return rows
    if max_rows_per_question <= 0:
        raise ValueError("--max-train-per-question must be positive when supplied.")

    rng = np.random.default_rng(random_state)
    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[str(row.get("question", ""))].append(row)

    sampled_rows: List[Dict[str, Any]] = []
    for question in sorted(by_question):
        question_rows = by_question[question]
        if len(question_rows) <= max_rows_per_question:
            sampled_rows.extend(question_rows)
            continue
        selected_indices = sorted(
            rng.choice(len(question_rows), size=max_rows_per_question, replace=False).tolist()
        )
        sampled_rows.extend(question_rows[idx] for idx in selected_indices)

    return sampled_rows


# Function: Convert ECG-QA rows into model arrays.
# Inputs: rows containing "embedding", "question", and "label" fields.
# Outputs: ECG embedding matrix, question list, and binary label vector.
def rows_to_arrays(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, List[str], np.ndarray]:
    ecg_embeddings = []
    questions = []
    labels = []

    for row in rows:
        embedding = row.get("embedding")
        question = row.get("question")
        label = row.get("label")

        if embedding is None or question is None or label is None:
            continue

        ecg_embeddings.append(embedding)
        questions.append(str(question))
        labels.append(int(label))

    X_ecg = np.array(ecg_embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X_ecg, questions, y


# Function: Build question representations for train and validation rows.
# Inputs: train/validation question text, encoder type, and optional model name.
# Outputs: train question matrix, validation question matrix, and encoder artifact.
def encode_questions(
    train_questions: List[str],
    val_questions: List[str],
    encoder: str,
    model_name: str,
    allow_download: bool,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if encoder == "onehot":
        question_to_index = {
            question: idx for idx, question in enumerate(sorted(set(train_questions)))
        }
        unknown_index = len(question_to_index)

        def transform(questions: List[str]) -> np.ndarray:
            X = np.zeros((len(questions), len(question_to_index) + 1), dtype=np.float32)
            for row_idx, question in enumerate(questions):
                col_idx = question_to_index.get(question, unknown_index)
                X[row_idx, col_idx] = 1.0
            return X

        artifact = {
            "question_to_index": question_to_index,
            "unknown_index": unknown_index,
        }
        return transform(train_questions), transform(val_questions), artifact

    if encoder == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        X_train_text = vectorizer.fit_transform(train_questions).toarray()
        X_val_text = vectorizer.transform(val_questions).toarray()
        return X_train_text.astype(np.float32), X_val_text.astype(np.float32), vectorizer

    if encoder != "sentence-transformer":
        if encoder != "medcpt":
            raise ValueError("Unknown question encoder. Use: onehot, tfidf, medcpt, or sentence-transformer.")

    try:
        if encoder == "sentence-transformer":
            from sentence_transformers import SentenceTransformer
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Required text embedding dependencies are not installed. Install sentence_transformers "
            "for --question-encoder sentence-transformer, or transformers/torch for "
            "--question-encoder medcpt. Use onehot or tfidf for dependency-light baselines."
        ) from exc

    if encoder == "sentence-transformer":
        model = SentenceTransformer(model_name, local_files_only=not allow_download)

        X_train_text = model.encode(
            train_questions,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        X_val_text = model.encode(
            val_questions,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return X_train_text.astype(np.float32), X_val_text.astype(np.float32), model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download)
    model = AutoModel.from_pretrained(model_name, local_files_only=not allow_download)
    model.eval()

    # Function: Encode questions using MedCPT-style CLS embeddings.
    # Inputs: Question strings, tokenizer/model, and batch size.
    # Outputs: Dense 768-dimensional question embedding matrix.
    def encode_medcpt_questions(questions: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(questions), batch_size):
                batch_questions = questions[start : start + batch_size]
                encoded = tokenizer(
                    batch_questions,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=64,
                )
                batch_embeddings = model(**encoded).last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    return encode_medcpt_questions(train_questions), encode_medcpt_questions(val_questions), model_name


# Function: Optionally reduce ECG and question embeddings to a shared PCA dimension.
# Inputs: train/validation ECG matrices, train/validation text matrices, optional PCA dimension, and seed.
# Outputs: transformed matrices plus fitted PCA artifacts, or original matrices when PCA is disabled.
def maybe_apply_pca(
    X_train_ecg: np.ndarray,
    X_val_ecg: np.ndarray,
    X_train_text: np.ndarray,
    X_val_text: np.ndarray,
    pca_dim: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if pca_dim is None:
        return X_train_ecg, X_val_ecg, X_train_text, X_val_text, {}

    max_components = min(
        pca_dim,
        X_train_ecg.shape[0],
        X_train_ecg.shape[1],
        X_train_text.shape[0],
        X_train_text.shape[1],
    )
    if max_components < 1:
        raise ValueError("PCA dimension must be at least 1 after checking data shapes.")

    ecg_pca = PCA(n_components=max_components, random_state=random_state)
    text_pca = PCA(n_components=max_components, random_state=random_state)

    X_train_ecg_pca = ecg_pca.fit_transform(X_train_ecg).astype(np.float32)
    X_val_ecg_pca = ecg_pca.transform(X_val_ecg).astype(np.float32)
    X_train_text_pca = text_pca.fit_transform(X_train_text).astype(np.float32)
    X_val_text_pca = text_pca.transform(X_val_text).astype(np.float32)

    artifacts = {
        "requested_pca_dim": int(pca_dim),
        "actual_pca_dim": int(max_components),
        "ecg_pca": ecg_pca,
        "text_pca": text_pca,
        "ecg_explained_variance_ratio_sum": float(np.sum(ecg_pca.explained_variance_ratio_)),
        "text_explained_variance_ratio_sum": float(np.sum(text_pca.explained_variance_ratio_)),
    }

    return X_train_ecg_pca, X_val_ecg_pca, X_train_text_pca, X_val_text_pca, artifacts


# Function: Build multimodal fusion matrices from ECG and text embeddings.
# Inputs: ECG/text train and validation matrices plus fusion strategy.
# Outputs: fused train and validation feature matrices.
def fuse_embeddings(
    X_train_ecg: np.ndarray,
    X_val_ecg: np.ndarray,
    X_train_text: np.ndarray,
    X_val_text: np.ndarray,
    fusion: str,
) -> tuple[np.ndarray, np.ndarray]:
    if fusion == "concat":
        return (
            np.concatenate([X_train_ecg, X_train_text], axis=1),
            np.concatenate([X_val_ecg, X_val_text], axis=1),
        )

    if fusion == "mean":
        if X_train_ecg.shape[1] != X_train_text.shape[1]:
            raise ValueError(
                "Mean fusion requires ECG and text embeddings with the same dimension. "
                "Use --pca-dim 256 to project both modalities first."
            )
        return (
            ((X_train_ecg + X_train_text) / 2.0).astype(np.float32),
            ((X_val_ecg + X_val_text) / 2.0).astype(np.float32),
        )

    raise ValueError("Unknown fusion strategy. Use: concat or mean.")


# Function: Construct the simple classifier used by all embedding baselines.
# Inputs: random seed.
# Outputs: sklearn pipeline with scaling and class-balanced logistic regression.
def build_classifier(random_state: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )


# Function: Extract positive-class probabilities when the model supports them.
# Inputs: fitted model and feature matrix.
# Outputs: probability vector for label 1, or None if unavailable.
def positive_prob(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return None


# Function: Compute ROC AUC only when it is mathematically defined.
# Inputs: true labels and optional positive-class probabilities.
# Outputs: ROC AUC float, or None for unavailable/degenerate cases.
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray | None) -> float | None:
    if y_prob is None or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


# Function: Compute average precision only when it is mathematically defined.
# Inputs: true labels and optional positive-class probabilities.
# Outputs: average precision float, or None for unavailable/degenerate cases.
def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray | None) -> float | None:
    if y_prob is None or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


# Function: Compute validation metrics separately for each target SCP code.
# Inputs: validation rows, true labels, and predicted labels.
# Outputs: mapping from SCP code to per-code metric dictionary.
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
        }

    return metrics


# Function: Compute standard binary classification metrics for one method.
# Inputs: method name, validation rows, true labels, predictions, and probabilities.
# Outputs: metrics dictionary suitable for JSON serialization.
def evaluate_predictions(
    method: str,
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
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
        "method": method,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
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


# Function: Build row-level prediction records for error analysis.
# Inputs: method name, validation rows, true labels, predictions, and probabilities.
# Outputs: list of JSON-serializable prediction dictionaries.
def make_prediction_rows(
    method: str,
    rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> List[Dict[str, Any]]:
    prediction_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        pred_prob = None if y_prob is None else float(y_prob[idx])
        prediction_rows.append(
            {
                "method": method,
                "ecg_id": row.get("ecg_id"),
                "split": row.get("split"),
                "target_scp_code": row.get("target_scp_code"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "true_label": int(y_true[idx]),
                "pred_label": int(y_pred[idx]),
                "pred_answer": "yes" if int(y_pred[idx]) == 1 else "no",
                "pred_prob_yes": pred_prob,
                "correct": bool(int(y_true[idx]) == int(y_pred[idx])),
            }
        )

    return prediction_rows


# Function: Train one baseline model and evaluate it on validation data.
# Inputs: method name, train/validation matrices, labels, validation rows, and seed.
# Outputs: fitted model, metrics dictionary, and prediction rows.
def fit_and_evaluate(
    method: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    val_rows: List[Dict[str, Any]],
    random_state: int,
) -> tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    clf = build_classifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_prob = positive_prob(clf, X_val)

    metrics = evaluate_predictions(method, val_rows, y_val, y_pred, y_prob)
    predictions = make_prediction_rows(method, val_rows, y_val, y_pred, y_prob)

    return clf, metrics, predictions


# Function: Evaluate the majority-class baseline.
# Inputs: train/validation rows, labels, and random seed.
# Outputs: fitted dummy model, metrics dictionary, and prediction rows.
def evaluate_majority_baseline(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    clf.fit(np.zeros((len(y_train), 1), dtype=np.float32), y_train)

    X_dummy_val = np.zeros((len(y_val), 1), dtype=np.float32)
    y_pred = clf.predict(X_dummy_val)
    y_prob = positive_prob(clf, X_dummy_val)

    metrics = evaluate_predictions("majority", val_rows, y_val, y_pred, y_prob)
    predictions = make_prediction_rows("majority", val_rows, y_val, y_pred, y_prob)

    return clf, metrics, predictions


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: output path and rows to serialize.
# Outputs: None; writes file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Build target-specific output paths without overwriting all-code results.
# Inputs: default output path and optional target SCP code.
# Outputs: original path when no target is set, otherwise path with target suffix.
def make_target_output_path(path: Path, target_scp_code: str | None) -> Path:
    if target_scp_code is None:
        return path

    suffix = target_scp_code.strip().lower()
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


# Function: Run the embedding baseline experiment from command-line arguments.
# Inputs: CLI arguments for dataset paths, question encoder, target code, and outputs.
# Outputs: None; writes metrics, predictions, and model bundle to disk.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument(
        "--question-encoder",
        type=str,
        default="tfidf",
        choices=["onehot", "tfidf", "sentence-transformer", "medcpt"],
        help=(
            "Question representation to use. onehot tests exact question identity; "
            "medcpt uses ncbi/MedCPT-style biomedical CLS embeddings; "
            "sentence-transformer uses dense question embeddings; tfidf is a lightweight fallback."
        ),
    )
    parser.add_argument("--question-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--allow-question-model-download",
        action="store_true",
        help="Allow sentence-transformers to download model files from Hugging Face.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="Optional PCA dimension applied separately to ECG and text embeddings, e.g. 256.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "mean", "both"],
        help="How to combine ECG and text embeddings for the multimodal baseline.",
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-bundle-path", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument(
        "--target-scp-code",
        type=str,
        default=None,
        help="Optional single SCP code subset, e.g. AFIB.",
    )
    parser.add_argument(
        "--max-train-per-question",
        type=int,
        default=None,
        help="Optional cap on train rows per exact question. Validation rows are never capped.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    train_rows = filter_rows_by_scp_code(train_rows, args.target_scp_code)
    val_rows = filter_rows_by_scp_code(val_rows, args.target_scp_code)
    train_rows = limit_rows_per_question(
        train_rows,
        max_rows_per_question=args.max_train_per_question,
        random_state=args.seed,
    )

    if not train_rows or not val_rows:
        raise RuntimeError("Train and validation rows must both be non-empty after filtering.")

    X_train_ecg, train_questions, y_train = rows_to_arrays(train_rows)
    X_val_ecg, val_questions, y_val = rows_to_arrays(val_rows)

    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("ECG embedding dim:", X_train_ecg.shape[1])
    print("Train label counts:", dict(Counter(int(x) for x in y_train)))
    print("Val label counts:", dict(Counter(int(x) for x in y_val)))
    print(f"Target SCP code: {args.target_scp_code or 'ALL'}")
    print(f"Question encoder: {args.question_encoder}")
    print(f"Question model: {args.question_model}")
    print(f"PCA dim: {args.pca_dim if args.pca_dim is not None else 'none'}")
    print(f"Fusion: {args.fusion}")
    print(f"Max train per question: {args.max_train_per_question or 'none'}")

    X_train_text, X_val_text, question_encoder_artifact = encode_questions(
        train_questions,
        val_questions,
        encoder=args.question_encoder,
        model_name=args.question_model,
        allow_download=args.allow_question_model_download,
    )
    print("Question embedding dim:", X_train_text.shape[1])

    X_train_ecg_model, X_val_ecg_model, X_train_text_model, X_val_text_model, pca_artifacts = maybe_apply_pca(
        X_train_ecg,
        X_val_ecg,
        X_train_text,
        X_val_text,
        pca_dim=args.pca_dim,
        random_state=args.seed,
    )
    print("Model ECG embedding dim:", X_train_ecg_model.shape[1])
    print("Model question embedding dim:", X_train_text_model.shape[1])
    if pca_artifacts:
        print(
            "PCA explained variance:",
            {
                "ecg": round(pca_artifacts["ecg_explained_variance_ratio_sum"], 4),
                "text": round(pca_artifacts["text_explained_variance_ratio_sum"], 4),
            },
        )

    models: Dict[str, Any] = {}
    results: Dict[str, Any] = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "target_scp_code": args.target_scp_code,
        "question_encoder": args.question_encoder,
        "question_model": args.question_model,
        "pca_dim": args.pca_dim,
        "fusion": args.fusion,
        "max_train_per_question": args.max_train_per_question,
        "model_ecg_embedding_dim": int(X_train_ecg_model.shape[1]),
        "model_text_embedding_dim": int(X_train_text_model.shape[1]),
        "pca": {
            key: value
            for key, value in pca_artifacts.items()
            if key not in {"ecg_pca", "text_pca"}
        },
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_label_counts": dict(Counter(int(x) for x in y_train)),
        "val_label_counts": dict(Counter(int(x) for x in y_val)),
        "methods": {},
    }
    all_predictions: List[Dict[str, Any]] = []

    model, metrics, predictions = evaluate_majority_baseline(
        train_rows,
        val_rows,
        y_train,
        y_val,
        random_state=args.seed,
    )
    models["majority"] = model
    results["methods"]["majority"] = metrics
    all_predictions.extend(predictions)

    for method, X_train, X_val in [
        ("ecg_only", X_train_ecg_model, X_val_ecg_model),
        ("text_only", X_train_text_model, X_val_text_model),
    ]:
        model, metrics, predictions = fit_and_evaluate(
            method,
            X_train,
            X_val,
            y_train,
            y_val,
            val_rows,
            random_state=args.seed,
        )
        models[method] = model
        results["methods"][method] = metrics
        all_predictions.extend(predictions)

    fusion_methods = ["concat", "mean"] if args.fusion == "both" else [args.fusion]
    for fusion in fusion_methods:
        X_train_combined, X_val_combined = fuse_embeddings(
            X_train_ecg_model,
            X_val_ecg_model,
            X_train_text_model,
            X_val_text_model,
            fusion=fusion,
        )
        method = "combined" if fusion == "concat" else "combined_mean"
        model, metrics, predictions = fit_and_evaluate(
            method,
            X_train_combined,
            X_val_combined,
            y_train,
            y_val,
            val_rows,
            random_state=args.seed,
        )
        models[method] = model
        results["methods"][method] = metrics
        all_predictions.extend(predictions)

    results_path = make_target_output_path(args.results_path, args.target_scp_code)
    predictions_path = make_target_output_path(args.predictions_path, args.target_scp_code)
    model_bundle_path = make_target_output_path(args.model_bundle_path, args.target_scp_code)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_jsonl(predictions_path, all_predictions)

    model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "question_encoder": args.question_encoder,
            "question_encoder_artifact": question_encoder_artifact,
            "question_model": args.question_model,
            "pca_artifacts": pca_artifacts,
            "results": results,
        },
        model_bundle_path,
    )

    print("\n=== Validation Results ===")
    for method, metrics in results["methods"].items():
        print(
            f"{method:10s} "
            f"acc={metrics['accuracy']:.3f} "
            f"bal_acc={metrics['balanced_accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
        )

    print(f"\nSaved results: {results_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved model bundle: {model_bundle_path}")


if __name__ == "__main__":
    main()
