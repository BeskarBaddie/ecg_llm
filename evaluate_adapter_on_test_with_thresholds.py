from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from evaluate_ecg_question_swap import load_model_components, predict_pair
from train_ecg_soft_prompt_adapter import load_embedding_bank, load_jsonl, write_json, write_jsonl


DEFAULT_TEST_PATH = Path("outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_test.jsonl")
DEFAULT_RESULTS_PATH = Path("outputs/adapter_3b_test_threshold_evaluation_results.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/adapter_3b_test_threshold_evaluation_predictions.jsonl")


# Function: Load validation-tuned threshold configuration from disk.
# Inputs: threshold-tuning JSON path.
# Outputs: dictionary containing global and per-code threshold settings.
def load_threshold_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Calculate binary classification metrics from true and predicted labels.
# Inputs: true labels and predicted labels.
# Outputs: dictionary of accuracy, balanced accuracy, recall, precision, F1, and confusion counts.
def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    counts = Counter(zip(y_true, y_pred))
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    n = tn + fp + fn + tp

    no_recall = tn / (tn + fp) if (tn + fp) else None
    yes_recall = tp / (tp + fn) if (tp + fn) else None
    no_precision = tn / (tn + fn) if (tn + fn) else None
    yes_precision = tp / (tp + fp) if (tp + fp) else None
    no_f1 = (
        2 * no_precision * no_recall / (no_precision + no_recall)
        if no_precision is not None and no_recall is not None and no_precision + no_recall > 0
        else 0.0
    )
    yes_f1 = (
        2 * yes_precision * yes_recall / (yes_precision + yes_recall)
        if yes_precision is not None and yes_recall is not None and yes_precision + yes_recall > 0
        else 0.0
    )

    return {
        "n": n,
        "accuracy": (tn + tp) / n if n else None,
        "balanced_accuracy": (yes_recall + no_recall) / 2
        if yes_recall is not None and no_recall is not None
        else None,
        "macro_f1": (yes_f1 + no_f1) / 2,
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "yes_precision": yes_precision,
        "no_precision": no_precision,
        "yes_f1": yes_f1,
        "no_f1": no_f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "prediction_counts": dict(Counter(y_pred)),
        "label_counts": dict(Counter(y_true)),
    }


# Function: Apply default, global, and per-code thresholds to scored prediction rows.
# Inputs: prediction rows with yes_score/no_score and threshold configuration.
# Outputs: rows augmented with thresholded predictions.
def apply_thresholds(
    rows: List[Dict[str, Any]],
    threshold_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    global_threshold = float(threshold_config["global"]["best_threshold"])
    per_code_thresholds = {
        str(code): float(metrics["threshold"])
        for code, metrics in threshold_config["per_code"]["per_code"].items()
    }

    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        margin = float(row["yes_score"]) - float(row["no_score"])
        code = str(row["target_scp_code"])
        default_pred = int(margin >= 0.0)
        global_pred = int(margin >= global_threshold)
        code_threshold = per_code_thresholds.get(code, global_threshold)
        per_code_pred = int(margin >= code_threshold)
        output_rows.append(
            {
                **row,
                "score_margin": margin,
                "default_threshold": 0.0,
                "global_threshold": global_threshold,
                "per_code_threshold": code_threshold,
                "default_pred_label": default_pred,
                "global_threshold_pred_label": global_pred,
                "per_code_threshold_pred_label": per_code_pred,
                "default_correct": bool(default_pred == int(row["true_label"])),
                "global_threshold_correct": bool(global_pred == int(row["true_label"])),
                "per_code_threshold_correct": bool(per_code_pred == int(row["true_label"])),
            }
        )
    return output_rows


# Function: Compute overall and per-code metrics for all thresholding methods.
# Inputs: thresholded prediction rows.
# Outputs: nested metrics dictionary.
def evaluate_thresholded_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [int(row["true_label"]) for row in rows]
    methods = {
        "default_threshold": "default_pred_label",
        "global_tuned_threshold": "global_threshold_pred_label",
        "per_code_tuned_thresholds": "per_code_threshold_pred_label",
    }

    overall: Dict[str, Any] = {}
    for method, field in methods.items():
        overall[method] = binary_metrics(y_true, [int(row[field]) for row in rows])

    by_code_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_code_rows[str(row["target_scp_code"])].append(row)

    by_code: Dict[str, Any] = {}
    for code, code_rows in sorted(by_code_rows.items()):
        code_true = [int(row["true_label"]) for row in code_rows]
        by_code[code] = {
            method: binary_metrics(code_true, [int(row[field]) for row in code_rows])
            for method, field in methods.items()
        }

    return {
        "overall": overall,
        "per_code": by_code,
    }


# Function: Score each test ECG-QA example with the trained adapter-LLM.
# Inputs: model components, embedding bank, test rows, device, and max sequence length.
# Outputs: prediction rows containing yes/no scores before thresholding.
def score_test_rows(
    llm: Any,
    adapter: Any,
    tokenizer: Any,
    embedding_bank: Dict[int, Any],
    rows: List[Dict[str, Any]],
    device: torch.device,
    max_length: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    scored_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            scored = predict_pair(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                embedding_bank=embedding_bank,
                ecg_id=int(row["ecg_id"]),
                question=str(row["question"]),
                device=device,
                max_length=max_length,
            )
            scored_rows.append(
                {
                    "ecg_id": int(row["ecg_id"]),
                    "target_scp_code": str(row["target_scp_code"]),
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                    "true_label": int(row["label"]),
                    "yes_score": scored["yes_score"],
                    "no_score": scored["no_score"],
                }
            )
            if progress_every > 0 and idx % progress_every == 0:
                print(f"Scored {idx}/{len(rows)} test examples")
    return scored_rows


# Function: Run proper held-out test evaluation with frozen validation-tuned thresholds.
# Inputs: command-line arguments.
# Outputs: JSON metrics and JSONL test predictions.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    test_rows = load_jsonl(args.test_path)
    embedding_bank = load_embedding_bank(args.embedding_bank)
    threshold_config = load_threshold_config(args.threshold_config)
    tokenizer, llm, adapter, checkpoint_config = load_model_components(
        llm_model=args.llm_model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        torch_dtype=args.torch_dtype,
        use_safetensors=args.use_safetensors,
        allow_model_download=args.allow_model_download,
    )

    print(f"Test rows: {len(test_rows)}")
    scored_rows = score_test_rows(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        embedding_bank=embedding_bank,
        rows=test_rows,
        device=device,
        max_length=args.max_length,
        progress_every=args.progress_every,
    )
    thresholded_rows = apply_thresholds(scored_rows, threshold_config=threshold_config)
    metrics = evaluate_thresholded_predictions(thresholded_rows)

    results = {
        "experiment": "held_out_test_threshold_evaluation",
        "test_path": str(args.test_path),
        "embedding_bank": str(args.embedding_bank),
        "llm_model": args.llm_model,
        "checkpoint_dir": str(args.checkpoint_dir),
        "threshold_config": str(args.threshold_config),
        "checkpoint_config": checkpoint_config,
        "n_test_rows": len(test_rows),
        "metrics": metrics,
    }
    write_json(args.results_path, results)
    write_jsonl(args.predictions_path, thresholded_rows)

    print("\nHeld-out test metrics")
    for method, method_metrics in metrics["overall"].items():
        print(
            f"{method:28s} "
            f"bal={method_metrics['balanced_accuracy']:.3f} "
            f"acc={method_metrics['accuracy']:.3f} "
            f"yes_rec={method_metrics['yes_recall']:.3f} "
            f"no_rec={method_metrics['no_recall']:.3f} "
            f"pred_yes={method_metrics['prediction_counts'].get(1, 0)}"
        )
    print(f"Saved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
