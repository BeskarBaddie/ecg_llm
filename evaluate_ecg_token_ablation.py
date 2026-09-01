from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_ecg_soft_prompt_adapter import (
    build_adapter,
    build_prompt,
    get_module_dtype,
    load_embedding_bank,
    load_jsonl,
    normalize_answer,
    parse_lora_target_modules,
    apply_lora_to_llm,
    tokenize_training_batch,
    write_json,
    write_jsonl,
)


DEFAULT_VAL_PATH = Path("outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl")
DEFAULT_RESULTS_PATH = Path("outputs/ecg_token_ablation_results.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/ecg_token_ablation_predictions.jsonl")


class IndexedECGQADataset(Dataset):
    # Function: Store validation ECG-QA rows with stable row indices for ablation lookup.
    # Inputs: row dictionaries plus an external ecg_id-to-embedding bank.
    # Outputs: PyTorch dataset returning one validation example at a time.
    def __init__(self, rows: List[Dict[str, Any]], embedding_bank: Dict[int, np.ndarray]) -> None:
        self.rows = rows
        self.embedding_bank = embedding_bank

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        ecg_id = int(row["ecg_id"])
        if ecg_id not in self.embedding_bank:
            raise KeyError(f"No embedding found for ecg_id={ecg_id}")
        return {
            "row_index": idx,
            "embedding": torch.tensor(self.embedding_bank[ecg_id], dtype=torch.float32),
            "question": str(row["question"]),
            "answer": normalize_answer(str(row["answer"])),
            "label": int(row["label"]),
            "ecg_id": ecg_id,
            "target_scp_code": row.get("target_scp_code"),
        }


# Function: Collate indexed validation examples into tensors and metadata lists.
# Inputs: list of dataset examples.
# Outputs: batch dictionary with stacked ECG embeddings and row indices.
def collate_indexed_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "row_indices": [int(item["row_index"]) for item in batch],
        "embeddings": torch.stack([item["embedding"] for item in batch], dim=0),
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "ecg_ids": [item["ecg_id"] for item in batch],
        "target_scp_codes": [item["target_scp_code"] for item in batch],
    }


# Function: Choose the requested PyTorch dtype from a command-line string.
# Inputs: dtype name.
# Outputs: torch dtype or "auto" for Hugging Face loading.
def parse_torch_dtype(dtype_name: str) -> Any:
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


# Function: Load adapter architecture metadata from a checkpoint directory.
# Inputs: checkpoint directory path.
# Outputs: parsed adapter_config dictionary.
def load_checkpoint_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load the frozen LLM, optional LoRA weights, and trained ECG adapter.
# Inputs: model path, checkpoint path, device, dtype options, and download/safetensor flags.
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


# Function: Convert adapter soft tokens according to an ablation condition.
# Inputs: original soft tokens, ablation mode, precomputed mean tokens, shuffled token bank, batch indices, and random scale.
# Outputs: ablated soft-token tensor with the same shape as the original tokens.
def ablate_soft_tokens(
    soft_tokens: torch.Tensor,
    mode: str,
    mean_soft_token: torch.Tensor | None,
    shuffled_soft_tokens: torch.Tensor | None,
    row_indices: List[int],
    random_scale: float,
) -> torch.Tensor:
    if mode == "real":
        return soft_tokens
    if mode == "zero":
        return torch.zeros_like(soft_tokens)
    if mode == "mean":
        if mean_soft_token is None:
            raise ValueError("mean_soft_token is required for mean ablation.")
        return mean_soft_token.expand_as(soft_tokens).to(device=soft_tokens.device, dtype=soft_tokens.dtype)
    if mode == "shuffled":
        if shuffled_soft_tokens is None:
            raise ValueError("shuffled_soft_tokens is required for shuffled ablation.")
        return shuffled_soft_tokens[row_indices].to(device=soft_tokens.device, dtype=soft_tokens.dtype)
    if mode == "random":
        return torch.randn_like(soft_tokens) * random_scale
    raise ValueError(f"Unsupported ablation mode: {mode}")


# Function: Build LLM input embeddings from ablated ECG soft tokens and text tokens.
# Inputs: LLM, adapter, ECG embeddings, token IDs, attention masks, labels, ablation state, and batch row indices.
# Outputs: combined input embeddings, combined attention mask, and combined labels.
def build_ablation_inputs(
    llm: Any,
    adapter: nn.Module,
    ecg_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    mean_soft_token: torch.Tensor | None,
    shuffled_soft_tokens: torch.Tensor | None,
    row_indices: List[int],
    random_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_embeds = llm.get_input_embeddings()(input_ids)
    ecg_embeddings = ecg_embeddings.to(dtype=get_module_dtype(adapter))
    soft_tokens = adapter(ecg_embeddings)
    soft_tokens = ablate_soft_tokens(
        soft_tokens=soft_tokens,
        mode=mode,
        mean_soft_token=mean_soft_token,
        shuffled_soft_tokens=shuffled_soft_tokens,
        row_indices=row_indices,
        random_scale=random_scale,
    ).to(dtype=text_embeds.dtype)

    inputs_embeds = torch.cat([soft_tokens, text_embeds], dim=1)
    soft_attention = torch.ones(
        (attention_mask.shape[0], soft_tokens.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    combined_attention = torch.cat([soft_attention, attention_mask], dim=1)

    soft_labels = torch.full(
        (labels.shape[0], soft_tokens.shape[1]),
        fill_value=-100,
        dtype=labels.dtype,
        device=labels.device,
    )
    combined_labels = torch.cat([soft_labels, labels], dim=1)
    return inputs_embeds, combined_attention, combined_labels


# Function: Precompute adapter outputs over the validation set for mean and shuffled ablations.
# Inputs: adapter, dataloader, device, and random seed.
# Outputs: mean soft token, shuffled soft-token bank, and random token scale.
def precompute_ablation_state(
    adapter: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    adapter.eval()
    token_rows: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].to(device=device, dtype=get_module_dtype(adapter))
            token_rows.append(adapter(embeddings).detach().cpu())

    soft_tokens = torch.cat(token_rows, dim=0)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(soft_tokens.shape[0], generator=generator)
    mean_soft_token = soft_tokens.mean(dim=0, keepdim=True).to(device=device)
    shuffled_soft_tokens = soft_tokens[permutation].to(device=device)
    random_scale = float(soft_tokens.std().item())
    return mean_soft_token, shuffled_soft_tokens, random_scale


# Function: Score one yes/no candidate under a selected ECG-token ablation condition.
# Inputs: model components, one ECG embedding, question, answer candidate, device, max length, ablation state, and row index.
# Outputs: negative loss score where higher means the candidate is more likely.
def score_candidate_with_ablation(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    ecg_embedding: torch.Tensor,
    question: str,
    answer: str,
    device: torch.device,
    max_length: int,
    mode: str,
    mean_soft_token: torch.Tensor | None,
    shuffled_soft_tokens: torch.Tensor | None,
    row_index: int,
    random_scale: float,
) -> float:
    input_ids, attention_mask, labels = tokenize_training_batch(
        tokenizer,
        [question],
        [answer],
        device=device,
        max_length=max_length,
    )
    inputs_embeds, combined_attention, combined_labels = build_ablation_inputs(
        llm=llm,
        adapter=adapter,
        ecg_embeddings=ecg_embedding.unsqueeze(0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        mode=mode,
        mean_soft_token=mean_soft_token,
        shuffled_soft_tokens=shuffled_soft_tokens,
        row_indices=[row_index],
        random_scale=random_scale,
    )
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=combined_attention,
        labels=combined_labels,
    )
    return -float(outputs.loss.detach().cpu())


# Function: Compute validation metrics for predicted yes/no labels.
# Inputs: prediction rows, true labels, and predicted labels.
# Outputs: dictionary of overall classification metrics.
def evaluate_predictions(
    prediction_rows: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "confusion_matrix_labels": ["no", "yes"],
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
    }


# Function: Evaluate one ablation mode over the validation dataloader.
# Inputs: model components, dataloader, device, max length, ablation mode, and precomputed ablation state.
# Outputs: prediction rows and aggregate metrics.
def evaluate_ablation_mode(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_length: int,
    mode: str,
    mean_soft_token: torch.Tensor | None,
    shuffled_soft_tokens: torch.Tensor | None,
    random_scale: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adapter.eval()
    llm.eval()
    prediction_rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].to(device)
            for idx, question in enumerate(batch["questions"]):
                row_index = int(batch["row_indices"][idx])
                yes_score = score_candidate_with_ablation(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    ecg_embedding=embeddings[idx],
                    question=question,
                    answer="yes",
                    device=device,
                    max_length=max_length,
                    mode=mode,
                    mean_soft_token=mean_soft_token,
                    shuffled_soft_tokens=shuffled_soft_tokens,
                    row_index=row_index,
                    random_scale=random_scale,
                )
                no_score = score_candidate_with_ablation(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    ecg_embedding=embeddings[idx],
                    question=question,
                    answer="no",
                    device=device,
                    max_length=max_length,
                    mode=mode,
                    mean_soft_token=mean_soft_token,
                    shuffled_soft_tokens=shuffled_soft_tokens,
                    row_index=row_index,
                    random_scale=random_scale,
                )
                pred_label = int(yes_score >= no_score)
                true_label = int(batch["labels"][idx].item())
                y_true.append(true_label)
                y_pred.append(pred_label)
                prediction_rows.append(
                    {
                        "ablation_mode": mode,
                        "row_index": row_index,
                        "ecg_id": batch["ecg_ids"][idx],
                        "target_scp_code": batch["target_scp_codes"][idx],
                        "question": question,
                        "answer": batch["answers"][idx],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "pred_answer": "yes" if pred_label == 1 else "no",
                        "yes_score": yes_score,
                        "no_score": no_score,
                        "correct": bool(true_label == pred_label),
                    }
                )

    metrics = evaluate_predictions(
        prediction_rows=prediction_rows,
        y_true=np.array(y_true, dtype=np.int64),
        y_pred=np.array(y_pred, dtype=np.int64),
    )
    return prediction_rows, metrics


# Function: Limit validation rows for quick smoke tests while keeping deterministic sampling.
# Inputs: row dictionaries, optional row limit, and random seed.
# Outputs: sampled row dictionaries.
def limit_rows(rows: List[Dict[str, Any]], limit: int | None, seed: int) -> List[Dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:limit]


# Function: Run ECG-token ablations for a trained soft-prompt adapter checkpoint.
# Inputs: command-line arguments.
# Outputs: JSON metrics and JSONL predictions for each ablation mode.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--modes", type=str, default="real,zero,mean,shuffled,random")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--debug-val-limit", type=int, default=None)
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

    rows = limit_rows(load_jsonl(args.val_path), args.debug_val_limit, args.seed)
    embedding_bank = load_embedding_bank(args.embedding_bank)
    tokenizer, llm, adapter, checkpoint_config = load_model_components(
        llm_model=args.llm_model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        torch_dtype=args.torch_dtype,
        use_safetensors=args.use_safetensors,
        allow_model_download=args.allow_model_download,
    )

    dataloader = DataLoader(
        IndexedECGQADataset(rows, embedding_bank=embedding_bank),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_indexed_batch,
    )
    mean_soft_token, shuffled_soft_tokens, random_scale = precompute_ablation_state(
        adapter=adapter,
        dataloader=dataloader,
        device=device,
        seed=args.seed,
    )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    all_predictions: List[Dict[str, Any]] = []
    metrics_by_mode: Dict[str, Any] = {}
    for mode in modes:
        prediction_rows, metrics = evaluate_ablation_mode(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            max_length=args.max_length,
            mode=mode,
            mean_soft_token=mean_soft_token,
            shuffled_soft_tokens=shuffled_soft_tokens,
            random_scale=random_scale,
        )
        metrics_by_mode[mode] = metrics
        all_predictions.extend(prediction_rows)
        print(
            f"{mode:8s} acc={metrics['accuracy']:.3f} "
            f"bal={metrics['balanced_accuracy']:.3f} "
            f"f1={metrics['macro_f1']:.3f} "
            f"yes_recall={metrics['labels']['yes']['recall']:.3f} "
            f"no_recall={metrics['labels']['no']['recall']:.3f} "
            f"pred_counts={metrics['prediction_counts']}"
        )

    baseline = metrics_by_mode.get("real")
    deltas = {}
    if baseline:
        for mode, metrics in metrics_by_mode.items():
            deltas[mode] = {
                "delta_accuracy_vs_real": metrics["accuracy"] - baseline["accuracy"],
                "delta_balanced_accuracy_vs_real": metrics["balanced_accuracy"] - baseline["balanced_accuracy"],
                "delta_macro_f1_vs_real": metrics["macro_f1"] - baseline["macro_f1"],
                "delta_yes_recall_vs_real": metrics["labels"]["yes"]["recall"] - baseline["labels"]["yes"]["recall"],
                "delta_no_recall_vs_real": metrics["labels"]["no"]["recall"] - baseline["labels"]["no"]["recall"],
            }

    results = {
        "experiment": "ecg_token_ablation",
        "val_path": str(args.val_path),
        "embedding_bank": str(args.embedding_bank),
        "llm_model": args.llm_model,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_config": checkpoint_config,
        "device": str(device),
        "torch_dtype": args.torch_dtype,
        "modes": modes,
        "n_rows": len(rows),
        "random_scale": random_scale,
        "metrics_by_mode": metrics_by_mode,
        "deltas_vs_real": deltas,
    }
    write_json(args.results_path, results)
    write_jsonl(args.predictions_path, all_predictions)
    print(f"Saved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
