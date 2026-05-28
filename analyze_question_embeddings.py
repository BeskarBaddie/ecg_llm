from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


TRAIN_PATH = Path("outputs/ecgqa_scp_binary_train.jsonl")
VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
RESULTS_PATH = Path("outputs/question_embedding_diagnostics.json")
COORDS_PATH = Path("outputs/question_embedding_coordinates.csv")
CODE_PLOT_PATH = Path("outputs/question_embedding_by_code.png")
LABEL_PLOT_PATH = Path("outputs/question_embedding_by_label.png")


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


# Function: Build question embeddings using a local text encoder.
# Inputs: question strings, encoder type, and optional sentence-transformer model name.
# Outputs: dense question embedding matrix and encoder metadata.
def encode_questions(
    questions: List[str],
    encoder: str,
    model_name: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if encoder == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        X = vectorizer.fit_transform(questions).toarray().astype(np.float32)
        return X, {
            "encoder": "tfidf",
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        }

    if encoder != "sentence-transformer":
        raise ValueError("Unknown question encoder. Use: tfidf or sentence-transformer.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is not installed. Install it or run with "
            "--question-encoder tfidf."
        ) from exc

    model = SentenceTransformer(model_name)
    X = model.encode(
        questions,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    return X, {
        "encoder": "sentence-transformer",
        "model_name": model_name,
    }


# Function: Project high-dimensional question embeddings into two dimensions.
# Inputs: embedding matrix, reducer name, random seed, and UMAP neighbor/min-distance settings.
# Outputs: 2D coordinate matrix and reducer metadata.
def reduce_to_2d(
    X: np.ndarray,
    reducer: str,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    X_scaled = StandardScaler().fit_transform(X)

    if reducer == "umap":
        try:
            import umap
        except ImportError:
            reducer = "pca"
        else:
            coordinates = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="cosine",
                random_state=random_state,
            ).fit_transform(X_scaled)
            return coordinates.astype(np.float32), {
                "reducer": "umap",
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": "cosine",
            }

    coordinates = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)
    return coordinates.astype(np.float32), {
        "reducer": "pca",
        "fallback_reason": "umap_not_installed" if reducer == "pca" else None,
    }


# Function: Compute silhouette separability for a categorical grouping.
# Inputs: embedding matrix and string/int group labels.
# Outputs: silhouette score, or None when the grouping is degenerate.
def safe_silhouette(X: np.ndarray, labels: List[Any]) -> float | None:
    encoded = LabelEncoder().fit_transform([str(label) for label in labels])
    if len(np.unique(encoded)) < 2 or len(np.unique(encoded)) >= len(encoded):
        return None

    return float(silhouette_score(X, encoded, metric="cosine"))


# Function: Save 2D embedding coordinates and row metadata as CSV.
# Inputs: output path, dataset rows, and 2D coordinates.
# Outputs: None; writes the CSV file to disk.
def write_coordinates_csv(path: Path, rows: List[Dict[str, Any]], coordinates: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("x,y,split,ecg_id,target_scp_code,label,answer,question\n")
        for row, (x_coord, y_coord) in zip(rows, coordinates):
            question = str(row.get("question", "")).replace('"', '""')
            f.write(
                f"{float(x_coord)},{float(y_coord)},"
                f"{row.get('split')},{row.get('ecg_id')},"
                f"{row.get('target_scp_code')},{int(row.get('label'))},"
                f"{row.get('answer')},\"{question}\"\n"
            )


# Function: Plot 2D question embeddings colored by a row metadata field.
# Inputs: output path, coordinates, rows, metadata field, and plot title.
# Outputs: None; writes a PNG image to disk.
def plot_projection(
    path: Path,
    coordinates: np.ndarray,
    rows: List[Dict[str, Any]],
    color_field: str,
    title: str,
) -> None:
    labels = [str(row.get(color_field)) for row in rows]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    for idx, label in enumerate(unique_labels):
        mask = np.array([item == label for item in labels])
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.78,
            color=cmap(idx % 10),
            label=label,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# Function: Run the question embedding diagnostic from command-line arguments.
# Inputs: CLI arguments for dataset paths, encoder, reducer, and output files.
# Outputs: JSON diagnostics, coordinate CSV, and projection PNG files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument(
        "--question-encoder",
        choices=["tfidf", "sentence-transformer"],
        default="sentence-transformer",
    )
    parser.add_argument("--question-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--reducer", choices=["umap", "pca"], default="umap")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--coords-path", type=Path, default=COORDS_PATH)
    parser.add_argument("--code-plot-path", type=Path, default=CODE_PLOT_PATH)
    parser.add_argument("--label-plot-path", type=Path, default=LABEL_PLOT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    rows = train_rows + val_rows

    if not rows:
        raise RuntimeError("No rows available for question embedding diagnostics.")

    questions = [str(row["question"]) for row in rows]
    codes = [str(row["target_scp_code"]) for row in rows]
    labels = [int(row["label"]) for row in rows]

    X_questions, encoder_info = encode_questions(
        questions,
        encoder=args.question_encoder,
        model_name=args.question_model,
    )
    coordinates, reducer_info = reduce_to_2d(
        X_questions,
        reducer=args.reducer,
        random_state=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    write_coordinates_csv(args.coords_path, rows, coordinates)
    plot_projection(
        args.code_plot_path,
        coordinates,
        rows,
        color_field="target_scp_code",
        title="Question embeddings colored by SCP code",
    )
    plot_projection(
        args.label_plot_path,
        coordinates,
        rows,
        color_field="label",
        title="Question embeddings colored by yes/no label",
    )

    results = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "n_rows": int(len(rows)),
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "encoder": encoder_info,
        "question_embedding_dim": int(X_questions.shape[1]),
        "reducer": reducer_info,
        "scp_code_counts": dict(Counter(codes)),
        "label_counts": dict(Counter(labels)),
        "silhouette_by_scp_code": safe_silhouette(X_questions, codes),
        "silhouette_by_yes_no_label": safe_silhouette(X_questions, labels),
        "outputs": {
            "coordinates_csv": str(args.coords_path),
            "code_plot": str(args.code_plot_path),
            "label_plot": str(args.label_plot_path),
        },
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Rows:", len(rows))
    print("Question embedding dim:", X_questions.shape[1])
    print("Reducer:", reducer_info["reducer"])
    print("Silhouette by SCP code:", results["silhouette_by_scp_code"])
    print("Silhouette by yes/no label:", results["silhouette_by_yes_no_label"])
    print(f"Saved results: {args.results_path}")
    print(f"Saved coordinates: {args.coords_path}")
    print(f"Saved code plot: {args.code_plot_path}")
    print(f"Saved label plot: {args.label_plot_path}")


if __name__ == "__main__":
    main()
