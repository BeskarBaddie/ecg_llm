from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_ecg_soft_prompt_adapter import (
    apply_lora_to_llm,
    build_adapter,
    build_adapter_inputs,
    load_embedding_bank,
    load_jsonl,
    parse_lora_target_modules,
    tokenize_training_batch,
    write_json,
    write_jsonl,
)


DEFAULT_VAL_PATH = Path("outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl")
DEFAULT_RESULTS_PATH = Path("outputs/ecg_question_swap_results.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/ecg_question_swap_predictions.jsonl")
DEFAULT_TARGET_CODES = "AFIB,CRBBB,PACE,LVH,STD_,INVT,NDT"


# Function: Convert a dtype name into a PyTorch dtype or auto value.
# Inputs: dtype string from the command line.
# Outputs: torch dtype object or "auto".
def parse_torch_dtype(dtype_name: str) -> Any:
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


# Function: Load adapter checkpoint configuration from disk.
# Inputs: checkpoint directory.
# Outputs: parsed adapter_config dictionary.
def load_checkpoint_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load frozen LLM, optional LoRA weights, and trained ECG adapter.
# Inputs: model path, checkpoint directory, device, dtype, safetensor flag, and download flag.
# Outputs: tokenizer, LLM, adapter, and checkpoint config.
def load_model_components(
    llm_model: str,
    checkpoint_dir: Path,
    device: torch.device,
    torch_dtype: str,
    use_safetensors: bool,
    allow_model_download: bool,
) -> tuple[Any, Any, nn.Module, Dict[str, Any]]:
    config = load_checkpoint_config(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model,
        local_files_only=not allow_model_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_model,
        local_files_only=not allow_model_download,
        use_safetensors=use_safetensors,
        torch_dtype=parse_torch_dtype(torch_dtype),
    )
    llm.to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    if config.get("lora_enabled"):
        llm = apply_lora_to_llm(
            llm,
            r=int(config["lora_r"]),
            alpha=int(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            target_modules=parse_lora_target_modules(",".join(config["lora_target_modules"])),
        )
        lora_dir = checkpoint_dir / "lora_adapter"
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_dir}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading LoRA checkpoints requires `peft`.") from exc
        llm = PeftModel.from_pretrained(llm, lora_dir, local_files_only=True)
        llm.to(device)
        llm.eval()

    adapter_dtype = torch.float32 if "float32" in str(config.get("adapter_dtype", "")) else llm.get_input_embeddings().weight.dtype
    adapter = build_adapter(
        adapter_type=str(config.get("adapter_type", "linear")),
        csfm_dim=int(config["csfm_dim"]),
        llm_hidden_size=int(config["llm_hidden_size"]),
        num_soft_tokens=int(config["num_soft_tokens"]),
        adapter_hidden_dim=int(config.get("adapter_hidden_dim") or 2048),
    ).to(device=device, dtype=adapter_dtype)

    adapter_path = checkpoint_dir / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_path}")
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()
    return tokenizer, llm, adapter, config


# Function: Score one yes/no candidate for a supplied ECG embedding and question.
# Inputs: model components, ECG embedding, question text, answer candidate, device, and max length.
# Outputs: negative language-model loss where higher means the candidate is more likely.
def score_candidate(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    ecg_embedding: torch.Tensor,
    question: str,
    answer: str,
    device: torch.device,
    max_length: int,
) -> float:
    input_ids, attention_mask, labels = tokenize_training_batch(
        tokenizer,
        [question],
        [answer],
        device=device,
        max_length=max_length,
    )
    inputs_embeds, combined_attention, combined_labels = build_adapter_inputs(
        llm=llm,
        adapter=adapter,
        ecg_embeddings=ecg_embedding.unsqueeze(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=combined_attention,
        labels=combined_labels,
    )
    return -float(outputs.loss.detach().cpu())


# Function: Predict yes/no for one ECG-question pair using forced-choice answer scoring.
# Inputs: model components, embedding bank, ECG ID, question text, device, and max length.
# Outputs: prediction dictionary with scores, predicted label, and predicted answer.
def predict_pair(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    embedding_bank: Dict[int, np.ndarray],
    ecg_id: int,
    question: str,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    if ecg_id not in embedding_bank:
        raise KeyError(f"No embedding found for ecg_id={ecg_id}")
    ecg_embedding = torch.tensor(embedding_bank[ecg_id], dtype=torch.float32, device=device)
    yes_score = score_candidate(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        ecg_embedding=ecg_embedding,
        question=question,
        answer="yes",
        device=device,
        max_length=max_length,
    )
    no_score = score_candidate(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        ecg_embedding=ecg_embedding,
        question=question,
        answer="no",
        device=device,
        max_length=max_length,
    )
    pred_label = int(yes_score >= no_score)
    return {
        "pred_label": pred_label,
        "pred_answer": "yes" if pred_label == 1 else "no",
        "yes_score": yes_score,
        "no_score": no_score,
    }


# Function: Build validation rows grouped by target SCP code and label.
# Inputs: validation rows and selected target codes.
# Outputs: nested dictionary mapping code to label to row list.
def group_rows_by_code_label(
    rows: List[Dict[str, Any]],
    target_codes: List[str],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    target_set = set(target_codes)
    grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        code = str(row.get("target_scp_code"))
        if code not in target_set:
            continue
        grouped[code][int(row["label"])].append(row)
    return grouped


# Function: Create same-question/different-ECG swap examples.
# Inputs: validation rows, target SCP codes, number of pairs per code, and random seed.
# Outputs: list of counterfactual examples where one positive and one negative ECG answer the same question.
def build_same_question_ecg_swaps(
    rows: List[Dict[str, Any]],
    target_codes: List[str],
    pairs_per_code: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped = group_rows_by_code_label(rows, target_codes)
    examples: List[Dict[str, Any]] = []
    for code in target_codes:
        positives = list(grouped.get(code, {}).get(1, []))
        negatives = list(grouped.get(code, {}).get(0, []))
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for idx, (pos_row, neg_row) in enumerate(zip(positives, negatives)):
            if idx >= pairs_per_code:
                break
            question = str(pos_row["question"])
            pair_id = f"{code}_same_question_{idx}"
            examples.append(
                {
                    "test_type": "same_question_different_ecg",
                    "pair_id": pair_id,
                    "target_scp_code": code,
                    "ecg_id": int(pos_row["ecg_id"]),
                    "question": question,
                    "true_label": 1,
                    "expected_answer": "yes",
                    "source_row_label": 1,
                }
            )
            examples.append(
                {
                    "test_type": "same_question_different_ecg",
                    "pair_id": pair_id,
                    "target_scp_code": code,
                    "ecg_id": int(neg_row["ecg_id"]),
                    "question": question,
                    "true_label": 0,
                    "expected_answer": "no",
                    "source_row_label": 0,
                }
            )
    return examples


# Function: Create same-ECG/different-question examples from validation rows.
# Inputs: validation rows, target SCP codes, max ECGs to sample, max questions per ECG, and random seed.
# Outputs: list of examples where one ECG is queried for several SCP-code attributes.
def build_same_ecg_question_swaps(
    rows: List[Dict[str, Any]],
    target_codes: List[str],
    max_ecgs: int,
    max_questions_per_ecg: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    target_set = set(target_codes)
    by_ecg: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("target_scp_code")) in target_set:
            by_ecg[int(row["ecg_id"])].append(row)

    eligible = [
        (ecg_id, ecg_rows)
        for ecg_id, ecg_rows in by_ecg.items()
        if len({str(row["target_scp_code"]) for row in ecg_rows}) >= 2
        and len({int(row["label"]) for row in ecg_rows}) >= 2
    ]
    rng.shuffle(eligible)

    examples: List[Dict[str, Any]] = []
    for ecg_idx, (ecg_id, ecg_rows) in enumerate(eligible[:max_ecgs]):
        by_code: Dict[str, Dict[str, Any]] = {}
        for row in ecg_rows:
            by_code.setdefault(str(row["target_scp_code"]), row)
        selected_rows = list(by_code.values())
        rng.shuffle(selected_rows)
        for row in selected_rows[:max_questions_per_ecg]:
            label = int(row["label"])
            examples.append(
                {
                    "test_type": "same_ecg_different_question",
                    "pair_id": f"ecg_{ecg_id}_question_set_{ecg_idx}",
                    "target_scp_code": str(row["target_scp_code"]),
                    "ecg_id": ecg_id,
                    "question": str(row["question"]),
                    "true_label": label,
                    "expected_answer": "yes" if label == 1 else "no",
                    "source_row_label": label,
                }
            )
    return examples


# Function: Compute basic binary metrics for swap examples.
# Inputs: prediction rows.
# Outputs: metrics dictionary with accuracy, recalls, and prediction counts.
def compute_binary_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    total = tn + fp + fn + tp
    yes_recall = tp / (tp + fn) if (tp + fn) else None
    no_recall = tn / (tn + fp) if (tn + fp) else None
    balanced_accuracy = (
        (yes_recall + no_recall) / 2
        if yes_recall is not None and no_recall is not None
        else None
    )
    return {
        "n": total,
        "accuracy": (tp + tn) / total if total else None,
        "balanced_accuracy": balanced_accuracy,
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "true_counts": dict(Counter(int(row["true_label"]) for row in rows)),
        "prediction_counts": dict(Counter(int(row["pred_label"]) for row in rows)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


# Function: Compute pair-level flip consistency for same-question ECG swaps.
# Inputs: prediction rows for same-question/different-ECG examples.
# Outputs: dictionary with pair-level correctness and flip rates.
def compute_pair_flip_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)

    complete_pairs = [pair_rows for pair_rows in by_pair.values() if len(pair_rows) == 2]
    both_correct = 0
    prediction_flips = 0
    expected_pattern = 0
    for pair_rows in complete_pairs:
        labels = {int(row["true_label"]): int(row["pred_label"]) for row in pair_rows}
        if all(row["correct"] for row in pair_rows):
            both_correct += 1
        if len(set(labels.values())) == 2:
            prediction_flips += 1
        if labels.get(1) == 1 and labels.get(0) == 0:
            expected_pattern += 1

    n = len(complete_pairs)
    return {
        "n_pairs": n,
        "both_correct_pairs": both_correct,
        "both_correct_rate": both_correct / n if n else None,
        "prediction_flip_pairs": prediction_flips,
        "prediction_flip_rate": prediction_flips / n if n else None,
        "expected_yes_no_pattern_pairs": expected_pattern,
        "expected_yes_no_pattern_rate": expected_pattern / n if n else None,
    }


# Function: Compute same-ECG question-set behavior metrics.
# Inputs: prediction rows for same-ECG/different-question examples.
# Outputs: dictionary describing whether predictions vary across questions for the same ECG.
def compute_same_ecg_question_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)

    complete_sets = [pair_rows for pair_rows in by_pair.values() if len(pair_rows) >= 2]
    varying_prediction_sets = 0
    varying_truth_sets = 0
    all_yes_sets = 0
    all_no_sets = 0
    for pair_rows in complete_sets:
        pred_values = {int(row["pred_label"]) for row in pair_rows}
        truth_values = {int(row["true_label"]) for row in pair_rows}
        if len(pred_values) >= 2:
            varying_prediction_sets += 1
        if len(truth_values) >= 2:
            varying_truth_sets += 1
        if pred_values == {1}:
            all_yes_sets += 1
        if pred_values == {0}:
            all_no_sets += 1

    n = len(complete_sets)
    return {
        "n_ecg_question_sets": n,
        "truth_varies_rate": varying_truth_sets / n if n else None,
        "prediction_varies_rate": varying_prediction_sets / n if n else None,
        "all_yes_set_rate": all_yes_sets / n if n else None,
        "all_no_set_rate": all_no_sets / n if n else None,
    }


# Function: Run forced-choice predictions for all swap examples.
# Inputs: model components, embedding bank, examples, device, and max length.
# Outputs: prediction rows with scores and correctness flags.
def score_examples(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    embedding_bank: Dict[int, np.ndarray],
    examples: List[Dict[str, Any]],
    device: torch.device,
    max_length: int,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, example in enumerate(examples, start=1):
            scored = predict_pair(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                embedding_bank=embedding_bank,
                ecg_id=int(example["ecg_id"]),
                question=str(example["question"]),
                device=device,
                max_length=max_length,
            )
            row = {**example, **scored}
            row["correct"] = bool(int(row["true_label"]) == int(row["pred_label"]))
            predictions.append(row)
            if idx % 100 == 0:
                print(f"Scored {idx}/{len(examples)} examples")
    return predictions


# Function: Run ECG/question swap tests against a trained adapter checkpoint.
# Inputs: command-line arguments.
# Outputs: JSON metrics and JSONL prediction rows.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--target-codes", type=str, default=DEFAULT_TARGET_CODES)
    parser.add_argument("--pairs-per-code", type=int, default=20)
    parser.add_argument("--same-ecg-max-sets", type=int, default=120)
    parser.add_argument("--same-ecg-max-questions", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    target_codes = [code.strip() for code in args.target_codes.split(",") if code.strip()]
    val_rows = load_jsonl(args.val_path)
    embedding_bank = load_embedding_bank(args.embedding_bank)
    tokenizer, llm, adapter, checkpoint_config = load_model_components(
        llm_model=args.llm_model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        torch_dtype=args.torch_dtype,
        use_safetensors=args.use_safetensors,
        allow_model_download=args.allow_model_download,
    )

    same_question_examples = build_same_question_ecg_swaps(
        rows=val_rows,
        target_codes=target_codes,
        pairs_per_code=args.pairs_per_code,
        seed=args.seed,
    )
    same_ecg_examples = build_same_ecg_question_swaps(
        rows=val_rows,
        target_codes=target_codes,
        max_ecgs=args.same_ecg_max_sets,
        max_questions_per_ecg=args.same_ecg_max_questions,
        seed=args.seed,
    )
    examples = same_question_examples + same_ecg_examples
    if not examples:
        raise RuntimeError("No swap examples were created. Check target codes and validation data.")

    print("Target codes:", ",".join(target_codes))
    print("Same-question/different-ECG examples:", len(same_question_examples))
    print("Same-ECG/different-question examples:", len(same_ecg_examples))

    predictions = score_examples(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        embedding_bank=embedding_bank,
        examples=examples,
        device=device,
        max_length=args.max_length,
    )

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_type[str(row["test_type"])].append(row)
        by_code[str(row["target_scp_code"])].append(row)

    metrics_by_type = {
        test_type: compute_binary_metrics(rows)
        for test_type, rows in sorted(by_type.items())
    }
    metrics_by_code = {
        code: compute_binary_metrics(rows)
        for code, rows in sorted(by_code.items())
    }
    same_question_rows = by_type.get("same_question_different_ecg", [])
    same_ecg_rows = by_type.get("same_ecg_different_question", [])

    results = {
        "experiment": "ecg_question_swap",
        "val_path": str(args.val_path),
        "embedding_bank": str(args.embedding_bank),
        "llm_model": args.llm_model,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_config": checkpoint_config,
        "target_codes": target_codes,
        "device": str(device),
        "n_examples": len(predictions),
        "metrics_by_type": metrics_by_type,
        "metrics_by_code": metrics_by_code,
        "same_question_pair_metrics": compute_pair_flip_metrics(same_question_rows),
        "same_ecg_question_metrics": compute_same_ecg_question_metrics(same_ecg_rows),
    }

    write_json(args.results_path, results)
    write_jsonl(args.predictions_path, predictions)

    print("\nMetrics by test type")
    for test_type, metrics in metrics_by_type.items():
        print(
            f"{test_type:30s} n={metrics['n']} "
            f"acc={metrics['accuracy']:.3f} "
            f"bal={metrics['balanced_accuracy']:.3f} "
            f"yes_rec={metrics['yes_recall']:.3f} "
            f"no_rec={metrics['no_recall']:.3f}"
        )
    print("Same-question pair metrics:", results["same_question_pair_metrics"])
    print("Same-ECG question metrics:", results["same_ecg_question_metrics"])
    print(f"Saved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
