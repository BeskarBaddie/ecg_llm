from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TRAIN_PATH = Path("outputs/ecgqa_scp_binary_train.jsonl")
VAL_PATH = Path("outputs/ecgqa_scp_binary_val.jsonl")
CHECKPOINT_DIR = Path("checkpoints/ecg_soft_prompt_adapter_stage1")
RESULTS_PATH = Path("outputs/ecg_soft_prompt_adapter_stage1_results.json")
PREDICTIONS_PATH = Path("outputs/ecg_soft_prompt_adapter_stage1_predictions.jsonl")


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


# Function: Write dictionaries to a newline-delimited JSON file.
# Inputs: output path and rows to serialize.
# Outputs: None; writes the file to disk.
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# Function: Make a prompt for generative binary ECG-QA adapter training.
# Inputs: ECG-QA question string.
# Outputs: prompt text ending immediately before the answer token.
def build_prompt(question: str) -> str:
    return "\n".join(
        [
            "You are a cardiology assistant.",
            "Answer the ECG question using the ECG representation.",
            f"Question: {question}",
            "Answer:",
        ]
    )


# Function: Convert a free-form answer into the controlled yes/no target text.
# Inputs: raw answer field from a dataset row.
# Outputs: normalized answer text suitable for language-model supervision.
def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().lower()
    if answer.startswith("yes"):
        return "yes"
    if answer.startswith("no"):
        return "no"
    raise ValueError(f"Expected yes/no answer, got: {answer}")


# Function: Parse generated model text into a yes/no prediction.
# Inputs: generated text after the prompt.
# Outputs: numeric label and normalized answer string.
def parse_yes_no_response(text: str) -> tuple[int, str]:
    cleaned = text.strip().lower()
    if cleaned.startswith("yes"):
        return 1, "yes"
    if cleaned.startswith("no"):
        return 0, "no"

    tokens = cleaned.replace(".", " ").replace(",", " ").split()
    if tokens:
        if tokens[0] == "yes":
            return 1, "yes"
        if tokens[0] == "no":
            return 0, "no"

    return -1, "unknown"


# Function: Seed Python, NumPy, and PyTorch for reproducible smoke tests.
# Inputs: integer random seed.
# Outputs: None; mutates global random state.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function: Select a small stratified debug subset without changing full-run behavior.
# Inputs: dataset rows, optional row limit, and random seed.
# Outputs: sampled rows with both labels represented when possible.
def limit_debug_rows(
    rows: List[Dict[str, Any]],
    limit: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("Debug row limits must be positive when supplied.")

    rng = random.Random(seed)
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    if len(by_label) < 2:
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled[:limit]

    labels = sorted(by_label)
    per_label = max(1, limit // len(labels))
    sampled_rows: List[Dict[str, Any]] = []
    for label in labels:
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        sampled_rows.extend(label_rows[:per_label])

    remaining_slots = limit - len(sampled_rows)
    if remaining_slots > 0:
        selected_ids = {id(row) for row in sampled_rows}
        remaining = [row for row in rows if id(row) not in selected_ids]
        rng.shuffle(remaining)
        sampled_rows.extend(remaining[:remaining_slots])

    rng.shuffle(sampled_rows)
    return sampled_rows[:limit]


class ECGQASoftPromptDataset(Dataset):
    # Function: Store ECG-QA rows for soft-prompt adapter training.
    # Inputs: row dictionaries with embedding, question, answer, and label fields.
    # Outputs: PyTorch dataset returning one raw row at a time.
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "embedding": torch.tensor(row["embedding"], dtype=torch.float32),
            "question": str(row["question"]),
            "answer": normalize_answer(str(row["answer"])),
            "label": int(row["label"]),
            "ecg_id": row.get("ecg_id"),
            "target_scp_code": row.get("target_scp_code"),
        }


# Function: Collate variable-length text examples while stacking fixed ECG embeddings.
# Inputs: batch of dataset examples.
# Outputs: dictionary with stacked embeddings and row metadata lists.
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "embeddings": torch.stack([item["embedding"] for item in batch], dim=0),
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "ecg_ids": [item["ecg_id"] for item in batch],
        "target_scp_codes": [item["target_scp_code"] for item in batch],
    }


class LinearSoftPromptAdapter(nn.Module):
    # Function: Project one pooled CSFM embedding into multiple LLM soft tokens.
    # Inputs: CSFM embedding dimension, LLM hidden size, and number of soft tokens.
    # Outputs: trainable PyTorch module returning [batch, num_soft_tokens, llm_hidden_size].
    def __init__(self, csfm_dim: int, llm_hidden_size: int, num_soft_tokens: int) -> None:
        super().__init__()
        self.csfm_dim = csfm_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_soft_tokens = num_soft_tokens
        self.proj = nn.Linear(csfm_dim, num_soft_tokens * llm_hidden_size)

    def forward(self, ecg_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = ecg_embeddings.shape[0]
        soft_tokens = self.proj(ecg_embeddings)
        return soft_tokens.view(batch_size, self.num_soft_tokens, self.llm_hidden_size)


# Function: Tokenize prompt+answer examples and build labels masked to answer tokens only.
# Inputs: tokenizer, questions, answers, device, and max sequence length.
# Outputs: token IDs, attention mask, and label tensor with prompt tokens set to -100.
def tokenize_training_batch(
    tokenizer: Any,
    questions: List[str],
    answers: List[str],
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_texts = [build_prompt(question) for question in questions]
    full_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompt_texts, answers)]

    encoded_full = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded_prompt = tokenizer(
        prompt_texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    input_ids = encoded_full["input_ids"].to(device)
    attention_mask = encoded_full["attention_mask"].to(device)
    labels = input_ids.clone()

    for row_idx, prompt_ids in enumerate(encoded_prompt["input_ids"]):
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[row_idx, :prompt_len] = -100

    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


# Function: Prepend ECG soft-token embeddings to token embeddings and adjust masks/labels.
# Inputs: frozen LLM, adapter, ECG embeddings, token IDs, attention masks, and labels.
# Outputs: combined input embeddings, combined attention mask, and combined labels.
def build_adapter_inputs(
    llm: Any,
    adapter: nn.Module,
    ecg_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_tokens = adapter(ecg_embeddings)

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


# Function: Run one adapter training epoch.
# Inputs: model components, dataloader, optimizer, device, max length, and gradient clipping value.
# Outputs: mean training loss for the epoch.
def train_one_epoch(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_length: int,
    grad_clip: float,
) -> float:
    adapter.train()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        ecg_embeddings = batch["embeddings"].to(device)
        input_ids, attention_mask, labels = tokenize_training_batch(
            tokenizer,
            batch["questions"],
            batch["answers"],
            device=device,
            max_length=max_length,
        )
        inputs_embeds, combined_attention, combined_labels = build_adapter_inputs(
            llm,
            adapter,
            ecg_embeddings,
            input_ids,
            attention_mask,
            labels,
        )

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
        )
        loss = outputs.loss
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return total_loss / max(total_batches, 1)


# Function: Generate yes/no predictions from the adapter-conditioned LLM.
# Inputs: model components, validation dataloader, device, max length, and generation length.
# Outputs: prediction rows and numeric true/predicted label arrays.
def evaluate_adapter(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    adapter.eval()
    predictions: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            ecg_embeddings = batch["embeddings"].to(device)
            prompts = [build_prompt(question) for question in batch["questions"]]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            dummy_labels = torch.full_like(input_ids, fill_value=-100)
            inputs_embeds, combined_attention, _ = build_adapter_inputs(
                llm,
                adapter,
                ecg_embeddings,
                input_ids,
                attention_mask,
                dummy_labels,
            )

            generated = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for idx, raw_response in enumerate(decoded):
                pred_label, pred_answer = parse_yes_no_response(raw_response)
                true_label = int(batch["labels"][idx].item())
                y_true.append(true_label)
                y_pred.append(pred_label)
                predictions.append(
                    {
                        "ecg_id": batch["ecg_ids"][idx],
                        "target_scp_code": batch["target_scp_codes"][idx],
                        "question": batch["questions"][idx],
                        "answer": batch["answers"][idx],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "pred_answer": pred_answer,
                        "raw_response": raw_response,
                        "correct": bool(true_label == pred_label),
                    }
                )

    return predictions, np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


# Function: Score one candidate answer by its language-model loss after the prompt.
# Inputs: model components, ECG embedding, question string, answer candidate, device, and max length.
# Outputs: negative loss score where higher means the answer is more likely.
def score_answer_candidate(
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
        llm,
        adapter,
        ecg_embedding.unsqueeze(0),
        input_ids,
        attention_mask,
        labels,
    )
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=combined_attention,
        labels=combined_labels,
    )
    return -float(outputs.loss.detach().cpu())


# Function: Evaluate adapter predictions by comparing yes/no answer likelihoods.
# Inputs: model components, validation dataloader, device, and max sequence length.
# Outputs: prediction rows and numeric true/predicted label arrays.
def evaluate_adapter_forced_choice(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_length: int,
) -> tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    adapter.eval()
    predictions: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].to(device)
            for idx, question in enumerate(batch["questions"]):
                yes_score = score_answer_candidate(
                    llm,
                    adapter,
                    tokenizer,
                    embeddings[idx],
                    question,
                    "yes",
                    device,
                    max_length,
                )
                no_score = score_answer_candidate(
                    llm,
                    adapter,
                    tokenizer,
                    embeddings[idx],
                    question,
                    "no",
                    device,
                    max_length,
                )
                pred_label = int(yes_score >= no_score)
                pred_answer = "yes" if pred_label == 1 else "no"
                true_label = int(batch["labels"][idx].item())

                y_true.append(true_label)
                y_pred.append(pred_label)
                predictions.append(
                    {
                        "ecg_id": batch["ecg_ids"][idx],
                        "target_scp_code": batch["target_scp_codes"][idx],
                        "question": question,
                        "answer": batch["answers"][idx],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "pred_answer": pred_answer,
                        "yes_score": yes_score,
                        "no_score": no_score,
                        "raw_response": "",
                        "correct": bool(true_label == pred_label),
                    }
                )

    return predictions, np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


# Function: Compute validation metrics separately for each target SCP code.
# Inputs: prediction rows, true labels, and predicted labels.
# Outputs: nested dictionary of per-code metric values.
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
            "prediction_counts": dict(Counter(int(x) for x in code_pred)),
        }

    return metrics


# Function: Compute overall validation metrics for adapter predictions.
# Inputs: prediction rows, true labels, and predicted labels.
# Outputs: JSON-serializable metrics dictionary.
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
        "invalid_predictions": int(np.sum(y_pred == -1)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "confusion_matrix_labels": ["no", "yes", "unknown"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, -1]).tolist(),
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
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        ),
        "per_code": per_code_metrics(prediction_rows, y_true, y_pred),
    }


# Function: Save adapter weights and reproducibility metadata.
# Inputs: checkpoint directory, adapter, tokenizer/model metadata, config, and metrics.
# Outputs: None; writes adapter.pt, adapter_config.json, and metrics.json.
def save_checkpoint(
    checkpoint_dir: Path,
    adapter: nn.Module,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), checkpoint_dir / "adapter.pt")

    with (checkpoint_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with (checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# Function: Run the stage-1 linear soft-prompt adapter experiment.
# Inputs: command-line arguments.
# Outputs: checkpoint, metrics JSON, and prediction JSONL files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--llm-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Load safetensors model weights when the selected Hugging Face model requires them.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for loading the frozen LLM. Use float32 for CPU smoke tests.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--num-soft-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="forced_choice",
        choices=["forced_choice", "generate"],
        help="Use yes/no likelihood scoring for smoke tests or free generation for LLM-style evaluation.",
    )
    parser.add_argument("--debug-train-limit", type=int, default=32)
    parser.add_argument("--debug-val-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    train_rows = limit_debug_rows(train_rows, args.debug_train_limit, seed=args.seed)
    val_rows = limit_debug_rows(val_rows, args.debug_val_limit, seed=args.seed + 1)

    if not train_rows or not val_rows:
        raise RuntimeError("Train and validation rows must both be non-empty.")

    csfm_dim = int(train_rows[0].get("embedding_dim", len(train_rows[0]["embedding"])))

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        local_files_only=not args.allow_model_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        local_files_only=not args.allow_model_download,
        use_safetensors=args.use_safetensors,
        torch_dtype=dtype_map[args.torch_dtype],
    )
    llm.to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    llm_hidden_size = int(llm.config.hidden_size)
    adapter = LinearSoftPromptAdapter(
        csfm_dim=csfm_dim,
        llm_hidden_size=llm_hidden_size,
        num_soft_tokens=args.num_soft_tokens,
    ).to(device)

    train_loader = DataLoader(
        ECGQASoftPromptDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ECGQASoftPromptDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("LLM model:", args.llm_model)
    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("CSFM dim:", csfm_dim)
    print("LLM hidden size:", llm_hidden_size)
    print("Soft tokens:", args.num_soft_tokens)
    print("Adapter parameters:", sum(p.numel() for p in adapter.parameters()))

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        mean_loss = train_one_epoch(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_length=args.max_length,
            grad_clip=args.grad_clip,
        )
        train_losses.append(mean_loss)
        print(f"Epoch {epoch}/{args.epochs} train_loss={mean_loss:.4f}")

    if args.eval_mode == "forced_choice":
        prediction_rows, y_true, y_pred = evaluate_adapter_forced_choice(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            max_length=args.max_length,
        )
    else:
        prediction_rows, y_true, y_pred = evaluate_adapter(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    metrics = evaluate_predictions(prediction_rows, y_true, y_pred)

    results = {
        "stage": "stage1_linear_soft_prompt_smoke_test",
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "llm_model": args.llm_model,
        "csfm_dim": csfm_dim,
        "llm_hidden_size": llm_hidden_size,
        "num_soft_tokens": args.num_soft_tokens,
        "adapter_type": "linear",
        "use_safetensors": args.use_safetensors,
        "torch_dtype": args.torch_dtype,
        "adapter_parameters": int(sum(p.numel() for p in adapter.parameters())),
        "trainable": "adapter_only",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "eval_mode": args.eval_mode,
        "debug_train_limit": args.debug_train_limit,
        "debug_val_limit": args.debug_val_limit,
        "train_losses": train_losses,
        "metrics": metrics,
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_jsonl(args.predictions_path, prediction_rows)

    checkpoint_config = {
        "llm_model": args.llm_model,
        "csfm_dim": csfm_dim,
        "llm_hidden_size": llm_hidden_size,
        "num_soft_tokens": args.num_soft_tokens,
        "adapter_type": "linear",
        "adapter_parameters": int(sum(p.numel() for p in adapter.parameters())),
        "prompt_template": build_prompt("{question}"),
        "trainable": "adapter_only",
    }
    save_checkpoint(args.checkpoint_dir, adapter, checkpoint_config, results)

    print("\n=== Validation Results ===")
    print(
        f"acc={metrics['accuracy']:.3f} "
        f"bal_acc={metrics['balanced_accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"invalid={metrics['invalid_predictions']}"
    )
    print(f"Saved checkpoint: {args.checkpoint_dir}")
    print(f"Saved results: {args.results_path}")
    print(f"Saved predictions: {args.predictions_path}")


if __name__ == "__main__":
    main()
