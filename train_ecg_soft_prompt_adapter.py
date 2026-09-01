from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
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


# Function: Convert a free-text run note into a filename-safe slug.
# Inputs: run note string and maximum slug length.
# Outputs: lowercase slug safe to use in result filenames.
def slugify_run_note(note: str, max_length: int = 80) -> str:
    slug_chars: List[str] = []
    previous_was_separator = False
    for char in note.lower().strip():
        if char.isalnum():
            slug_chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            slug_chars.append("_")
            previous_was_separator = True

    slug = "".join(slug_chars).strip("_")
    if not slug:
        slug = "run"
    return slug[:max_length].strip("_") or "run"


# Function: Prefix an output file path with a run identifier.
# Inputs: original output file path and run identifier.
# Outputs: output path in the same directory with the run identifier prepended.
def prefix_output_file(path: Path, run_id: str) -> Path:
    return path.with_name(f"{run_id}_{path.name}")


# Function: Prefix a checkpoint directory name with a run identifier.
# Inputs: original checkpoint directory path and run identifier.
# Outputs: checkpoint directory path under the same parent with run identifier prepended.
def prefix_checkpoint_dir(path: Path, run_id: str) -> Path:
    return path.with_name(f"{run_id}_{path.name}")


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


# Function: Write a dictionary to a JSON file.
# Inputs: output path and JSON-serializable payload.
# Outputs: None; writes the file to disk.
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Function: Load a CSFM embedding bank keyed by ECG ID.
# Inputs: Optional JSONL path containing ecg_id and embedding fields.
# Outputs: Dictionary mapping ecg_id to float32 embedding arrays, or empty dictionary when no path is given.
def load_embedding_bank(path: Path | None) -> Dict[int, np.ndarray]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Embedding bank not found: {path}")

    embeddings: Dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ecg_id = int(row["ecg_id"])
            if ecg_id in embeddings:
                raise ValueError(f"Duplicate ecg_id={ecg_id} in embedding bank line {line_no}")
            embeddings[ecg_id] = np.array(row["embedding"], dtype=np.float32)

    return embeddings


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


# Function: Optionally rebalance training rows by sampling labels before DataLoader creation.
# Inputs: training rows, sampling mode, and random seed.
# Outputs: sampled training rows; validation/test rows should never be passed through this function.
def sample_training_rows(
    rows: List[Dict[str, Any]],
    mode: str,
    seed: int,
    target_yes_fraction: float,
) -> List[Dict[str, Any]]:
    if mode == "none":
        return rows

    rng = random.Random(seed)
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    no_rows = list(by_label.get(0, []))
    yes_rows = list(by_label.get(1, []))
    if not no_rows or not yes_rows:
        raise ValueError("Training sampling requires both yes and no labels.")

    if mode == "upsample_yes_to_no":
        sampled_yes = [rng.choice(yes_rows) for _ in range(len(no_rows))]
        sampled = list(no_rows) + sampled_yes
    elif mode == "upsample_yes_to_fraction":
        if not 0.0 < target_yes_fraction < 1.0:
            raise ValueError("--target-yes-fraction must be between 0 and 1.")
        target_yes_count = round(
            (target_yes_fraction * len(no_rows)) / (1.0 - target_yes_fraction)
        )
        if target_yes_count < len(yes_rows):
            raise ValueError(
                "Requested target yes fraction is lower than the current yes fraction. "
                "Use downsample_no_to_yes or a larger --target-yes-fraction."
            )
        sampled_yes = list(yes_rows) + [
            rng.choice(yes_rows) for _ in range(target_yes_count - len(yes_rows))
        ]
        sampled = list(no_rows) + sampled_yes
    elif mode == "downsample_no_to_yes":
        sampled_no = rng.sample(no_rows, k=len(yes_rows))
        sampled = sampled_no + list(yes_rows)
    else:
        raise ValueError(f"Unsupported train sampling mode: {mode}")

    rng.shuffle(sampled)
    return sampled


class ECGQASoftPromptDataset(Dataset):
    # Function: Store ECG-QA rows for soft-prompt adapter training.
    # Inputs: row dictionaries plus optional external ecg_id-to-embedding bank.
    # Outputs: PyTorch dataset returning one raw row at a time.
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        embedding_bank: Dict[int, np.ndarray] | None = None,
    ) -> None:
        self.rows = rows
        self.embedding_bank = embedding_bank or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        ecg_id = int(row["ecg_id"])
        if "embedding" in row:
            embedding = np.array(row["embedding"], dtype=np.float32)
        elif ecg_id in self.embedding_bank:
            embedding = self.embedding_bank[ecg_id]
        else:
            raise KeyError(f"No embedding found for ecg_id={ecg_id}")

        return {
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "question": str(row["question"]),
            "answer": normalize_answer(str(row["answer"])),
            "label": int(row["label"]),
            "ecg_id": ecg_id,
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


class MLPSoftPromptAdapter(nn.Module):
    # Function: Map one pooled CSFM embedding into LLM soft tokens using a nonlinear MLP.
    # Inputs: CSFM dimension, LLM hidden size, number of soft tokens, and hidden layer size.
    # Outputs: trainable PyTorch module returning [batch, num_soft_tokens, llm_hidden_size].
    def __init__(
        self,
        csfm_dim: int,
        llm_hidden_size: int,
        num_soft_tokens: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.csfm_dim = csfm_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_soft_tokens = num_soft_tokens
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(csfm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_soft_tokens * llm_hidden_size),
        )

    def forward(self, ecg_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = ecg_embeddings.shape[0]
        soft_tokens = self.net(ecg_embeddings)
        return soft_tokens.view(batch_size, self.num_soft_tokens, self.llm_hidden_size)


# Function: Build the requested ECG-to-soft-token adapter module.
# Inputs: adapter type, CSFM dimension, LLM hidden size, number of soft tokens, and MLP hidden size.
# Outputs: initialized adapter module.
def build_adapter(
    adapter_type: str,
    csfm_dim: int,
    llm_hidden_size: int,
    num_soft_tokens: int,
    adapter_hidden_dim: int,
) -> nn.Module:
    if adapter_type == "linear":
        return LinearSoftPromptAdapter(
            csfm_dim=csfm_dim,
            llm_hidden_size=llm_hidden_size,
            num_soft_tokens=num_soft_tokens,
        )
    if adapter_type == "mlp":
        return MLPSoftPromptAdapter(
            csfm_dim=csfm_dim,
            llm_hidden_size=llm_hidden_size,
            num_soft_tokens=num_soft_tokens,
            hidden_dim=adapter_hidden_dim,
        )
    raise ValueError(f"Unsupported adapter type: {adapter_type}")


# Function: Read the dtype used by a trainable PyTorch module.
# Inputs: PyTorch module with at least one parameter.
# Outputs: torch dtype of the first parameter.
def get_module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


# Function: Parse a comma-separated list of LoRA target module names.
# Inputs: comma-separated target module string.
# Outputs: cleaned list of target module names.
def parse_lora_target_modules(target_modules: str) -> List[str]:
    modules = [module.strip() for module in target_modules.split(",") if module.strip()]
    if not modules:
        raise ValueError("At least one LoRA target module must be supplied.")
    return modules


# Function: Add trainable LoRA adapters to a frozen causal language model.
# Inputs: frozen LLM and LoRA hyperparameters.
# Outputs: PEFT-wrapped LLM with trainable LoRA parameters.
def apply_lora_to_llm(
    llm: Any,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the `peft` package. Install it in the training environment with: "
            "pip install peft"
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(llm, lora_config)


# Function: Load trainable LoRA weights into a frozen base language model.
# Inputs: frozen LLM and checkpoint directory containing a lora_adapter subdirectory.
# Outputs: PEFT-wrapped LLM with loaded LoRA weights marked trainable.
def load_lora_from_checkpoint(llm: Any, checkpoint_dir: Path) -> Any:
    lora_dir = checkpoint_dir / "lora_adapter"
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {lora_dir}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Resuming LoRA requires the `peft` package. Install it with: pip install peft"
        ) from exc

    return PeftModel.from_pretrained(llm, lora_dir, is_trainable=True)


# Function: Load saved adapter weights into the ECG soft-prompt adapter.
# Inputs: adapter module, checkpoint directory, and target device.
# Outputs: None; mutates adapter weights in place.
def load_adapter_from_checkpoint(
    adapter: nn.Module,
    checkpoint_dir: Path,
    device: torch.device,
) -> None:
    adapter_path = checkpoint_dir / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")
    state_dict = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)


# Function: Collect all parameters that should be optimized in this experiment.
# Inputs: adapter module and LLM, which may or may not contain trainable LoRA parameters.
# Outputs: list of trainable PyTorch parameters.
def collect_trainable_parameters(adapter: nn.Module, llm: Any) -> List[torch.nn.Parameter]:
    return [param for param in list(adapter.parameters()) + list(llm.parameters()) if param.requires_grad]


# Function: Count trainable parameters for reporting.
# Inputs: PyTorch module.
# Outputs: integer count of parameters with requires_grad=True.
def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


# Function: Compute no/yes class weights from training rows.
# Inputs: training row dictionaries, weighting mode, and optional explicit no/yes weights.
# Outputs: dictionary mapping class label 0/1 to loss weight.
def compute_class_weights(
    train_rows: List[Dict[str, Any]],
    class_weight_mode: str,
    no_loss_weight: float | None,
    yes_loss_weight: float | None,
) -> Dict[int, float]:
    if no_loss_weight is not None or yes_loss_weight is not None:
        return {
            0: float(no_loss_weight if no_loss_weight is not None else 1.0),
            1: float(yes_loss_weight if yes_loss_weight is not None else 1.0),
        }

    counts = Counter(int(row["label"]) for row in train_rows)
    n_no = max(int(counts[0]), 1)
    n_yes = max(int(counts[1]), 1)
    n_total = n_no + n_yes

    if class_weight_mode == "none":
        return {0: 1.0, 1: 1.0}
    if class_weight_mode == "balanced":
        return {
            0: n_total / (2.0 * n_no),
            1: n_total / (2.0 * n_yes),
        }
    if class_weight_mode == "sqrt_balanced":
        return {
            0: float(np.sqrt(n_total / (2.0 * n_no))),
            1: float(np.sqrt(n_total / (2.0 * n_yes))),
        }
    raise ValueError(f"Unsupported class weight mode: {class_weight_mode}")


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
    ecg_embeddings = ecg_embeddings.to(dtype=get_module_dtype(adapter))
    soft_tokens = adapter(ecg_embeddings)
    soft_tokens = soft_tokens.to(dtype=text_embeds.dtype)

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


# Function: Compute language-model loss with optional per-example class weights.
# Inputs: model logits, combined labels, batch class labels, and no/yes class weights.
# Outputs: scalar weighted cross-entropy loss over answer tokens only.
def weighted_answer_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_labels: torch.Tensor,
    class_weights: Dict[int, float],
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    answer_mask = shift_labels.ne(-100).to(dtype=token_loss.dtype)
    example_weights = torch.ones_like(class_labels, dtype=token_loss.dtype)
    example_weights = example_weights.to(device=token_loss.device)
    example_weights[class_labels.to(device=token_loss.device) == 0] = float(class_weights[0])
    example_weights[class_labels.to(device=token_loss.device) == 1] = float(class_weights[1])
    weighted_mask = answer_mask * example_weights.unsqueeze(1)
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (token_loss * weighted_mask).sum() / denominator


# Function: Run one adapter training epoch.
# Inputs: model components, trainable parameters, dataloader, optimizer, device, max length, clipping, accumulation settings, and class weights.
# Outputs: mean training loss for the epoch.
def train_one_epoch(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: List[torch.nn.Parameter],
    device: torch.device,
    max_length: int,
    grad_clip: float,
    gradient_accumulation_steps: int,
    class_weights: Dict[int, float],
) -> float:
    adapter.train()
    if any(param.requires_grad for param in llm.parameters()):
        llm.train()
    else:
        llm.eval()
    total_loss = 0.0
    total_batches = 0
    accumulation_steps = max(1, gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader, start=1):
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
        )
        batch_loss = weighted_answer_token_loss(
            logits=outputs.logits,
            labels=combined_labels,
            class_labels=batch["labels"].to(device),
            class_weights=class_weights,
        )
        if not torch.isfinite(batch_loss):
            raise RuntimeError(
                "Non-finite training loss detected. The run is numerically unstable; "
                "try a lower learning rate, stronger gradient clipping, or float32 adapter weights."
            )
        loss = batch_loss / accumulation_steps
        loss.backward()

        if batch_idx % accumulation_steps == 0 or batch_idx == len(dataloader):
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_parameters, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * accumulation_steps
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


# Function: Evaluate the adapter using the configured validation mode.
# Inputs: model components, validation loader, device, sequence lengths, and evaluation mode.
# Outputs: prediction rows plus aggregate validation metrics.
def run_validation(
    llm: Any,
    adapter: nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    eval_mode: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if eval_mode == "forced_choice":
        prediction_rows, y_true, y_pred = evaluate_adapter_forced_choice(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            max_length=max_length,
        )
    else:
        prediction_rows, y_true, y_pred = evaluate_adapter(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )

    return prediction_rows, evaluate_predictions(prediction_rows, y_true, y_pred)


# Function: Save adapter weights, optional LoRA weights, and reproducibility metadata.
# Inputs: checkpoint directory, adapter, optional PEFT-wrapped LLM, config, and metrics.
# Outputs: None; writes adapter.pt, optional lora_adapter directory, adapter_config.json, and metrics.json.
def save_checkpoint(
    checkpoint_dir: Path,
    adapter: nn.Module,
    llm: Any,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), checkpoint_dir / "adapter.pt")
    if config.get("lora_enabled") and hasattr(llm, "save_pretrained"):
        llm.save_pretrained(checkpoint_dir / "lora_adapter")

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
    parser.add_argument(
        "--embedding-bank",
        type=Path,
        default=None,
        help=(
            "Optional CSFM embedding JSONL bank keyed by ecg_id. Required for lightweight "
            "datasets that do not store an embedding vector in every row."
        ),
    )
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
    parser.add_argument(
        "--resume-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint directory containing adapter.pt and, for LoRA runs, "
            "lora_adapter/. This resumes model weights but not optimizer state."
        ),
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument(
        "--run-note",
        type=str,
        default="run",
        help="Short sentence describing what changed in this run. Used in timestamped output names.",
    )
    parser.add_argument(
        "--no-run-prefix",
        action="store_true",
        help="Disable automatic timestamp/run-note prefixes for outputs.",
    )
    parser.add_argument("--num-soft-tokens", type=int, default=8)
    parser.add_argument(
        "--adapter-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Adapter architecture mapping CSFM embeddings into LLM soft tokens.",
    )
    parser.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension for --adapter-type mlp.",
    )
    parser.add_argument(
        "--adapter-dtype",
        type=str,
        default="float32",
        choices=["float32", "llm"],
        help=(
            "Dtype for the trainable adapter. Use float32 with float16 LLMs to reduce NaN risk; "
            "use llm to reproduce older behavior."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate adapter gradients over this many batches before each optimizer step.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "balanced", "sqrt_balanced"],
        help=(
            "Apply class weights to the answer-token loss. balanced uses inverse-frequency "
            "weights from the training labels; sqrt_balanced uses the square root of those weights."
        ),
    )
    parser.add_argument(
        "--yes-loss-weight",
        type=float,
        default=None,
        help="Optional explicit loss weight for yes-labelled examples. Overrides --class-weight-mode.",
    )
    parser.add_argument(
        "--no-loss-weight",
        type=float,
        default=None,
        help="Optional explicit loss weight for no-labelled examples. Overrides --class-weight-mode.",
    )
    parser.add_argument(
        "--train-sampling-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "upsample_yes_to_no",
            "upsample_yes_to_fraction",
            "downsample_no_to_yes",
        ],
        help=(
            "Optional train-only label balancing before DataLoader creation. "
            "Validation is never sampled or modified."
        ),
    )
    parser.add_argument(
        "--target-yes-fraction",
        type=float,
        default=0.4,
        help=(
            "Target yes fraction for --train-sampling-mode upsample_yes_to_fraction. "
            "For example, 0.4 creates an approximate 40:60 yes/no train mix."
        ),
    )
    parser.add_argument(
        "--train-sampling-seed",
        type=int,
        default=None,
        help="Optional seed for train-only sampling. Defaults to --seed when omitted.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable gradient checkpointing inside the frozen LLM. This is slower but can reduce "
            "activation memory for larger LLMs because gradients still need to flow back to the "
            "adapter-generated input embeddings."
        ),
    )
    parser.add_argument(
        "--lora-enable",
        action="store_true",
        help="Train LoRA adapters inside the LLM in addition to the ECG soft-prompt adapter.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LLM module names to receive LoRA adapters.",
    )
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="forced_choice",
        choices=["forced_choice", "generate"],
        help="Use yes/no likelihood scoring for smoke tests or free generation for LLM-style evaluation.",
    )
    parser.add_argument(
        "--debug-train-limit",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Omit this for full training.",
    )
    parser.add_argument(
        "--debug-val-limit",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. Omit this for full validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_seed(args.seed)

    run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_timestamp}_{slugify_run_note(args.run_note)}"
    original_results_path = args.results_path
    original_predictions_path = args.predictions_path
    original_checkpoint_dir = args.checkpoint_dir
    if not args.no_run_prefix:
        args.results_path = prefix_output_file(args.results_path, run_id)
        args.predictions_path = prefix_output_file(args.predictions_path, run_id)
        args.checkpoint_dir = prefix_checkpoint_dir(args.checkpoint_dir, run_id)

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
    original_train_row_count = len(train_rows)
    original_train_label_counts = dict(Counter(int(row["label"]) for row in train_rows))
    train_sampling_seed = (
        args.train_sampling_seed if args.train_sampling_seed is not None else args.seed
    )
    train_rows = sample_training_rows(
        rows=train_rows,
        mode=args.train_sampling_mode,
        seed=train_sampling_seed,
        target_yes_fraction=args.target_yes_fraction,
    )
    sampled_train_label_counts = dict(Counter(int(row["label"]) for row in train_rows))

    if not train_rows or not val_rows:
        raise RuntimeError("Train and validation rows must both be non-empty.")

    class_weights = compute_class_weights(
        train_rows=train_rows,
        class_weight_mode=args.class_weight_mode,
        no_loss_weight=args.no_loss_weight,
        yes_loss_weight=args.yes_loss_weight,
    )

    embedding_bank = load_embedding_bank(args.embedding_bank)
    if embedding_bank:
        print("Loaded embedding bank ECGs:", len(embedding_bank))

    first_row = train_rows[0]
    if "embedding" in first_row:
        csfm_dim = int(first_row.get("embedding_dim", len(first_row["embedding"])))
    elif embedding_bank:
        first_ecg_id = int(first_row["ecg_id"])
        csfm_dim = int(embedding_bank[first_ecg_id].shape[0])
    else:
        raise RuntimeError(
            "Training rows do not contain embedding vectors. Provide --embedding-bank."
        )

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
    if args.gradient_checkpointing:
        if hasattr(llm.config, "use_cache"):
            llm.config.use_cache = False
        llm.gradient_checkpointing_enable()
    for param in llm.parameters():
        param.requires_grad = False

    lora_target_modules = parse_lora_target_modules(args.lora_target_modules)
    if args.lora_enable:
        if args.resume_checkpoint_dir is not None:
            llm = load_lora_from_checkpoint(llm, args.resume_checkpoint_dir)
            llm.to(device)
        else:
            llm = apply_lora_to_llm(
                llm,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=lora_target_modules,
            )
        llm.print_trainable_parameters()

    llm_dtype = llm.get_input_embeddings().weight.dtype
    llm_hidden_size = int(llm.config.hidden_size)
    adapter_dtype = torch.float32 if args.adapter_dtype == "float32" else llm_dtype
    adapter = build_adapter(
        adapter_type=args.adapter_type,
        csfm_dim=csfm_dim,
        llm_hidden_size=llm_hidden_size,
        num_soft_tokens=args.num_soft_tokens,
        adapter_hidden_dim=args.adapter_hidden_dim,
    ).to(device=device, dtype=adapter_dtype)
    if args.resume_checkpoint_dir is not None:
        load_adapter_from_checkpoint(adapter, args.resume_checkpoint_dir, device=device)

    train_loader = DataLoader(
        ECGQASoftPromptDataset(train_rows, embedding_bank=embedding_bank),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ECGQASoftPromptDataset(val_rows, embedding_bank=embedding_bank),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    trainable_parameters = collect_trainable_parameters(adapter=adapter, llm=llm)
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("LLM model:", args.llm_model)
    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("Original train rows:", original_train_row_count)
    print("Original train label counts:", original_train_label_counts)
    print("Train sampling mode:", args.train_sampling_mode)
    print("Target yes fraction:", args.target_yes_fraction)
    print("Train sampling seed:", train_sampling_seed)
    print("Sampled train label counts:", sampled_train_label_counts)
    print("CSFM dim:", csfm_dim)
    print("LLM hidden size:", llm_hidden_size)
    print("LLM dtype:", llm_dtype)
    print("Soft tokens:", args.num_soft_tokens)
    print("Adapter type:", args.adapter_type)
    print("Adapter hidden dim:", args.adapter_hidden_dim if args.adapter_type == "mlp" else "n/a")
    print("Adapter dtype:", get_module_dtype(adapter))
    print("Adapter parameters:", sum(p.numel() for p in adapter.parameters()))
    print("LoRA enabled:", args.lora_enable)
    print("LoRA target modules:", ",".join(lora_target_modules))
    print("LoRA r:", args.lora_r)
    print("LoRA alpha:", args.lora_alpha)
    print("LoRA dropout:", args.lora_dropout)
    print("LLM trainable parameters:", count_trainable_parameters(llm))
    print("Total trainable parameters:", sum(p.numel() for p in trainable_parameters))
    print("Resume checkpoint dir:", args.resume_checkpoint_dir)
    print("Gradient checkpointing:", args.gradient_checkpointing)
    print("Gradient accumulation steps:", args.gradient_accumulation_steps)
    print("Class weight mode:", args.class_weight_mode)
    print("Class weights:", class_weights)
    print("Run ID:", run_id)
    print("Run note:", args.run_note)
    print("Results path:", args.results_path)
    print("Predictions path:", args.predictions_path)
    print("Checkpoint dir:", args.checkpoint_dir)

    checkpoint_config = {
        "llm_model": args.llm_model,
        "embedding_bank": str(args.embedding_bank) if args.embedding_bank else None,
        "resume_checkpoint_dir": str(args.resume_checkpoint_dir)
        if args.resume_checkpoint_dir
        else None,
        "resume_mode": "weights_only_no_optimizer_state"
        if args.resume_checkpoint_dir
        else None,
        "csfm_dim": csfm_dim,
        "llm_hidden_size": llm_hidden_size,
        "num_soft_tokens": args.num_soft_tokens,
        "adapter_type": args.adapter_type,
        "adapter_hidden_dim": args.adapter_hidden_dim if args.adapter_type == "mlp" else None,
        "adapter_dtype": str(get_module_dtype(adapter)),
        "adapter_parameters": int(sum(p.numel() for p in adapter.parameters())),
        "lora_enabled": args.lora_enable,
        "lora_r": args.lora_r if args.lora_enable else None,
        "lora_alpha": args.lora_alpha if args.lora_enable else None,
        "lora_dropout": args.lora_dropout if args.lora_enable else None,
        "lora_target_modules": lora_target_modules if args.lora_enable else [],
        "llm_trainable_parameters": count_trainable_parameters(llm),
        "total_trainable_parameters": int(sum(p.numel() for p in trainable_parameters)),
        "prompt_template": build_prompt("{question}"),
        "trainable": "adapter_plus_lora" if args.lora_enable else "adapter_only",
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights,
        "yes_loss_weight": args.yes_loss_weight,
        "no_loss_weight": args.no_loss_weight,
        "train_sampling_mode": args.train_sampling_mode,
        "target_yes_fraction": args.target_yes_fraction,
        "train_sampling_seed": train_sampling_seed,
        "original_train_row_count": original_train_row_count,
        "original_train_label_counts": original_train_label_counts,
        "sampled_train_row_count": len(train_rows),
        "sampled_train_label_counts": sampled_train_label_counts,
    }

    results = {
        "stage": "stage1_linear_soft_prompt_smoke_test",
        "run_id": run_id,
        "run_note": args.run_note,
        "run_started_at": run_started_at,
        "run_output_prefix_enabled": not args.no_run_prefix,
        "requested_results_path": str(original_results_path),
        "requested_predictions_path": str(original_predictions_path),
        "requested_checkpoint_dir": str(original_checkpoint_dir),
        "resolved_results_path": str(args.results_path),
        "resolved_predictions_path": str(args.predictions_path),
        "resolved_checkpoint_dir": str(args.checkpoint_dir),
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "embedding_bank": str(args.embedding_bank) if args.embedding_bank else None,
        "resume_checkpoint_dir": str(args.resume_checkpoint_dir)
        if args.resume_checkpoint_dir
        else None,
        "resume_mode": "weights_only_no_optimizer_state"
        if args.resume_checkpoint_dir
        else None,
        "llm_model": args.llm_model,
        "csfm_dim": csfm_dim,
        "llm_hidden_size": llm_hidden_size,
        "num_soft_tokens": args.num_soft_tokens,
        "adapter_type": args.adapter_type,
        "adapter_hidden_dim": args.adapter_hidden_dim if args.adapter_type == "mlp" else None,
        "use_safetensors": args.use_safetensors,
        "torch_dtype": args.torch_dtype,
        "llm_dtype": str(llm_dtype),
        "adapter_dtype": str(get_module_dtype(adapter)),
        "adapter_parameters": int(sum(p.numel() for p in adapter.parameters())),
        "lora_enabled": args.lora_enable,
        "lora_r": args.lora_r if args.lora_enable else None,
        "lora_alpha": args.lora_alpha if args.lora_enable else None,
        "lora_dropout": args.lora_dropout if args.lora_enable else None,
        "lora_target_modules": lora_target_modules if args.lora_enable else [],
        "llm_trainable_parameters": count_trainable_parameters(llm),
        "total_trainable_parameters": int(sum(p.numel() for p in trainable_parameters)),
        "trainable": "adapter_plus_lora" if args.lora_enable else "adapter_only",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights,
        "yes_loss_weight": args.yes_loss_weight,
        "no_loss_weight": args.no_loss_weight,
        "train_sampling_mode": args.train_sampling_mode,
        "target_yes_fraction": args.target_yes_fraction,
        "train_sampling_seed": train_sampling_seed,
        "original_train_row_count": original_train_row_count,
        "original_train_label_counts": original_train_label_counts,
        "sampled_train_row_count": len(train_rows),
        "sampled_train_label_counts": sampled_train_label_counts,
        "grad_clip": args.grad_clip,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "eval_mode": args.eval_mode,
        "debug_train_limit": args.debug_train_limit,
        "debug_val_limit": args.debug_val_limit,
        "train_losses": [],
        "epoch_history": [],
        "best_epoch": None,
        "best_metric_name": "balanced_accuracy",
        "best_metric_value": None,
        "metrics": None,
        "status": "running",
    }

    best_score = -float("inf")
    best_epoch = 0
    best_metrics: Dict[str, Any] | None = None
    best_prediction_rows: List[Dict[str, Any]] = []

    write_json(args.results_path, results)
    for epoch in range(1, args.epochs + 1):
        mean_loss = train_one_epoch(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            device=device,
            max_length=args.max_length,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            class_weights=class_weights,
        )

        prediction_rows, epoch_metrics = run_validation(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            eval_mode=args.eval_mode,
        )

        score = float(epoch_metrics["balanced_accuracy"])
        results["train_losses"].append(mean_loss)
        results["epoch_history"].append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "metrics": epoch_metrics,
            }
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = epoch_metrics
            best_prediction_rows = prediction_rows
            results["best_epoch"] = best_epoch
            results["best_metric_value"] = best_score
            results["metrics"] = best_metrics
            save_checkpoint(args.checkpoint_dir, adapter, llm, checkpoint_config, results)
            write_jsonl(args.predictions_path, best_prediction_rows)

        write_json(args.results_path, results)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={mean_loss:.4f} "
            f"val_acc={epoch_metrics['accuracy']:.3f} "
            f"val_bal_acc={epoch_metrics['balanced_accuracy']:.3f} "
            f"val_macro_f1={epoch_metrics['macro_f1']:.3f} "
            f"best_epoch={best_epoch} "
            f"best_bal_acc={best_score:.3f}"
        )

    if best_metrics is None:
        raise RuntimeError("No validation metrics were produced.")

    metrics = best_metrics
    prediction_rows = best_prediction_rows
    results["status"] = "completed"
    results["metrics"] = metrics
    results["best_epoch"] = best_epoch
    results["best_metric_value"] = best_score
    write_json(args.results_path, results)
    write_jsonl(args.predictions_path, prediction_rows)

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
