from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from train_ecg_soft_prompt_adapter import (
    apply_lora_to_llm,
    build_adapter,
    get_module_dtype,
    load_embedding_bank,
    load_jsonl,
    parse_lora_target_modules,
    write_json,
)


DEFAULT_TRAIN_PATH = Path("outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_train.jsonl")
DEFAULT_VAL_PATH = Path("outputs/ecgqa_scp_binary_all_codes/ecgqa_scp_binary_val.jsonl")
DEFAULT_RESULTS_PATH = Path("outputs/adapter_token_probe_results.json")


class UniqueECGEmbeddingDataset(Dataset):
    # Function: Store unique ECG embeddings for adapter-token feature extraction.
    # Inputs: sorted ECG IDs and an ecg_id-to-CSFM-embedding bank.
    # Outputs: PyTorch dataset returning ECG IDs and pooled CSFM embeddings.
    def __init__(self, ecg_ids: List[int], embedding_bank: Dict[int, np.ndarray]) -> None:
        self.ecg_ids = ecg_ids
        self.embedding_bank = embedding_bank

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ecg_id = self.ecg_ids[idx]
        if ecg_id not in self.embedding_bank:
            raise KeyError(f"No embedding found for ecg_id={ecg_id}")
        return {
            "ecg_id": ecg_id,
            "embedding": torch.tensor(self.embedding_bank[ecg_id], dtype=torch.float32),
        }


# Function: Collate ECG embeddings for batched adapter-token extraction.
# Inputs: list of ECG dataset examples.
# Outputs: batch dictionary with ECG IDs and stacked embeddings.
def collate_ecg_embeddings(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ecg_ids": [int(item["ecg_id"]) for item in batch],
        "embeddings": torch.stack([item["embedding"] for item in batch], dim=0),
    }


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


# Function: Load the adapter checkpoint configuration.
# Inputs: checkpoint directory path.
# Outputs: parsed adapter_config dictionary.
def load_checkpoint_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load the trained soft-prompt adapter and a minimal LLM config context.
# Inputs: LLM path, checkpoint path, device, dtype, safetensor flag, and download flag.
# Outputs: frozen LLM, trained adapter, and checkpoint config.
def load_adapter_components(
    llm_model: str,
    checkpoint_dir: Path,
    device: torch.device,
    torch_dtype: str,
    use_safetensors: bool,
    allow_model_download: bool,
) -> tuple[Any, nn.Module, Dict[str, Any]]:
    config = load_checkpoint_config(checkpoint_dir)
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
    return llm, adapter, config


# Function: Build one binary label per unique ECG for each target SCP code.
# Inputs: ECG-QA rows with ecg_id, target_scp_code, and label fields.
# Outputs: nested dictionary mapping SCP code to ecg_id-to-label mapping.
def build_unique_labels_by_code(rows: List[Dict[str, Any]]) -> Dict[str, Dict[int, int]]:
    labels_by_code: Dict[str, Dict[int, int]] = defaultdict(dict)
    conflicts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        code = str(row["target_scp_code"])
        ecg_id = int(row["ecg_id"])
        label = int(row["label"])
        existing = labels_by_code[code].get(ecg_id)
        if existing is not None and existing != label:
            conflicts[(code, ecg_id)] += 1
            continue
        labels_by_code[code][ecg_id] = label
    if conflicts:
        print(f"Warning: ignored {sum(conflicts.values())} conflicting duplicate ECG/code labels.")
    return dict(labels_by_code)


# Function: Extract CSFM, adapter-mean, and adapter-flattened features for ECG IDs.
# Inputs: ECG IDs, embedding bank, trained adapter, device, and batch size.
# Outputs: dictionary mapping feature-set name to ecg_id-to-feature mapping.
def extract_feature_maps(
    ecg_ids: List[int],
    embedding_bank: Dict[int, np.ndarray],
    adapter: nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    dataloader = DataLoader(
        UniqueECGEmbeddingDataset(ecg_ids, embedding_bank),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ecg_embeddings,
    )
    feature_maps: Dict[str, Dict[int, np.ndarray]] = {
        "csfm": {},
        "adapter_mean": {},
        "adapter_flat": {},
    }
    adapter.eval()
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].to(device=device)
            soft_tokens = adapter(embeddings.to(dtype=get_module_dtype(adapter))).detach().cpu().numpy()
            pooled_embeddings = embeddings.detach().cpu().numpy()
            for idx, ecg_id in enumerate(batch["ecg_ids"]):
                feature_maps["csfm"][ecg_id] = pooled_embeddings[idx].astype(np.float32)
                feature_maps["adapter_mean"][ecg_id] = soft_tokens[idx].mean(axis=0).astype(np.float32)
                feature_maps["adapter_flat"][ecg_id] = soft_tokens[idx].reshape(-1).astype(np.float32)
    return feature_maps


# Function: Build classifier matrices for one SCP code and one feature representation.
# Inputs: train/val labels by ECG ID and feature mapping.
# Outputs: X/y arrays plus included ECG ID lists.
def build_xy(
    train_labels: Dict[int, int],
    val_labels: Dict[int, int],
    features: Dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    train_ids = sorted(ecg_id for ecg_id in train_labels if ecg_id in features)
    val_ids = sorted(ecg_id for ecg_id in val_labels if ecg_id in features)
    x_train = np.stack([features[ecg_id] for ecg_id in train_ids]).astype(np.float32)
    y_train = np.array([train_labels[ecg_id] for ecg_id in train_ids], dtype=np.int64)
    x_val = np.stack([features[ecg_id] for ecg_id in val_ids]).astype(np.float32)
    y_val = np.array([val_labels[ecg_id] for ecg_id in val_ids], dtype=np.int64)
    return x_train, y_train, x_val, y_val, train_ids, val_ids


# Function: Train and evaluate one balanced logistic-regression probe.
# Inputs: train/validation features and labels plus classifier hyperparameters.
# Outputs: metrics dictionary for the probe.
def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
    c_value: float,
) -> Dict[str, Any]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            C=c_value,
            solver="liblinear",
            random_state=42,
        ),
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    y_prob = clf.predict_proba(x_val)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "prediction_counts": dict(Counter(int(x) for x in y_pred)),
        "label_counts": dict(Counter(int(x) for x in y_val)),
    }
    if len(np.unique(y_val)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_val, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


# Function: Run all per-SCP-code probes across selected feature representations.
# Inputs: train/val rows, feature maps, selected feature sets, and fitting options.
# Outputs: list of per-code/per-feature probe result rows.
def run_probes(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    feature_maps: Dict[str, Dict[int, np.ndarray]],
    feature_sets: List[str],
    min_train_pos: int,
    min_val_pos: int,
    max_iter: int,
    c_value: float,
) -> List[Dict[str, Any]]:
    train_by_code = build_unique_labels_by_code(train_rows)
    val_by_code = build_unique_labels_by_code(val_rows)
    rows: List[Dict[str, Any]] = []
    for code in sorted(set(train_by_code) & set(val_by_code)):
        train_labels = train_by_code[code]
        val_labels = val_by_code[code]
        train_counts = Counter(train_labels.values())
        val_counts = Counter(val_labels.values())
        if train_counts[1] < min_train_pos or val_counts[1] < min_val_pos:
            continue
        if train_counts[0] == 0 or val_counts[0] == 0:
            continue

        for feature_set in feature_sets:
            x_train, y_train, x_val, y_val, train_ids, val_ids = build_xy(
                train_labels=train_labels,
                val_labels=val_labels,
                features=feature_maps[feature_set],
            )
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            metrics = fit_probe(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                max_iter=max_iter,
                c_value=c_value,
            )
            rows.append(
                {
                    "scp_code": code,
                    "feature_set": feature_set,
                    "train_n": int(len(y_train)),
                    "train_pos": int(np.sum(y_train == 1)),
                    "train_neg": int(np.sum(y_train == 0)),
                    "val_n": int(len(y_val)),
                    "val_pos": int(np.sum(y_val == 1)),
                    "val_neg": int(np.sum(y_val == 0)),
                    "feature_dim": int(x_train.shape[1]),
                    **metrics,
                }
            )
    return rows


# Function: Convert probe rows into a wide comparison table with deltas against CSFM.
# Inputs: long-form probe result rows.
# Outputs: list of wide per-SCP-code comparison rows.
def make_comparison_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_code: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_code[str(row["scp_code"])][str(row["feature_set"])] = row

    comparisons: List[Dict[str, Any]] = []
    for code, by_feature in sorted(by_code.items()):
        if "csfm" not in by_feature:
            continue
        csfm = by_feature["csfm"]
        comparison: Dict[str, Any] = {
            "scp_code": code,
            "train_n": csfm["train_n"],
            "train_pos": csfm["train_pos"],
            "val_n": csfm["val_n"],
            "val_pos": csfm["val_pos"],
            "csfm_roc_auc": csfm["roc_auc"],
            "csfm_balanced_accuracy": csfm["balanced_accuracy"],
        }
        for feature_set in ["adapter_mean", "adapter_flat"]:
            if feature_set not in by_feature:
                continue
            row = by_feature[feature_set]
            comparison[f"{feature_set}_roc_auc"] = row["roc_auc"]
            comparison[f"{feature_set}_balanced_accuracy"] = row["balanced_accuracy"]
            comparison[f"{feature_set}_delta_auc_vs_csfm"] = (
                row["roc_auc"] - csfm["roc_auc"]
                if row["roc_auc"] is not None and csfm["roc_auc"] is not None
                else None
            )
            comparison[f"{feature_set}_delta_bal_acc_vs_csfm"] = (
                row["balanced_accuracy"] - csfm["balanced_accuracy"]
            )
        comparisons.append(comparison)
    return comparisons


# Function: Write a list of dictionaries to CSV.
# Inputs: CSV path and result rows.
# Outputs: None; writes CSV.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Function: Print a compact summary of adapter-token probe results.
# Inputs: long-form probe rows and wide comparison rows.
# Outputs: None; prints high-signal metrics.
def print_summary(rows: List[Dict[str, Any]], comparisons: List[Dict[str, Any]]) -> None:
    print("Probe rows:", len(rows))
    by_feature: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[str(row["feature_set"])].append(row)
    for feature_set, feature_rows in sorted(by_feature.items()):
        aucs = [row["roc_auc"] for row in feature_rows if row["roc_auc"] is not None]
        bals = [row["balanced_accuracy"] for row in feature_rows]
        print(
            f"{feature_set:13s} codes={len(feature_rows):>3} "
            f"mean_auc={np.mean(aucs):.3f} "
            f"median_auc={np.median(aucs):.3f} "
            f"mean_bal={np.mean(bals):.3f}"
        )

    delta_rows = [
        row for row in comparisons
        if row.get("adapter_flat_delta_auc_vs_csfm") is not None
    ]
    print("\nLargest adapter_flat AUC drops vs CSFM")
    for row in sorted(delta_rows, key=lambda item: item["adapter_flat_delta_auc_vs_csfm"])[:10]:
        print(
            f"{row['scp_code']:8s} csfm_auc={row['csfm_roc_auc']:.3f} "
            f"flat_auc={row['adapter_flat_roc_auc']:.3f} "
            f"delta={row['adapter_flat_delta_auc_vs_csfm']:+.3f}"
        )
    print("\nLargest adapter_flat AUC gains vs CSFM")
    for row in sorted(delta_rows, key=lambda item: item["adapter_flat_delta_auc_vs_csfm"], reverse=True)[:10]:
        print(
            f"{row['scp_code']:8s} csfm_auc={row['csfm_roc_auc']:.3f} "
            f"flat_auc={row['adapter_flat_roc_auc']:.3f} "
            f"delta={row['adapter_flat_delta_auc_vs_csfm']:+.3f}"
        )


# Function: Run adapter-token information probes from command-line arguments.
# Inputs: command-line arguments.
# Outputs: JSON and CSV probe result files.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--embedding-bank", type=Path, required=True)
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--feature-sets", type=str, default="csfm,adapter_mean,adapter_flat")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-train-pos", type=int, default=20)
    parser.add_argument("--min-val-pos", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--use-safetensors", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
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

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)
    embedding_bank = load_embedding_bank(args.embedding_bank)
    _, adapter, checkpoint_config = load_adapter_components(
        llm_model=args.llm_model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        torch_dtype=args.torch_dtype,
        use_safetensors=args.use_safetensors,
        allow_model_download=args.allow_model_download,
    )

    train_ecg_ids = {int(row["ecg_id"]) for row in train_rows}
    val_ecg_ids = {int(row["ecg_id"]) for row in val_rows}
    all_ecg_ids = sorted((train_ecg_ids | val_ecg_ids) & set(embedding_bank))
    print(f"Unique train ECGs: {len(train_ecg_ids)}")
    print(f"Unique val ECGs: {len(val_ecg_ids)}")
    print(f"Feature ECGs available: {len(all_ecg_ids)}")

    feature_maps = extract_feature_maps(
        ecg_ids=all_ecg_ids,
        embedding_bank=embedding_bank,
        adapter=adapter,
        device=device,
        batch_size=args.batch_size,
    )
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    rows = run_probes(
        train_rows=train_rows,
        val_rows=val_rows,
        feature_maps=feature_maps,
        feature_sets=feature_sets,
        min_train_pos=args.min_train_pos,
        min_val_pos=args.min_val_pos,
        max_iter=args.max_iter,
        c_value=args.c_value,
    )
    comparisons = make_comparison_rows(rows)
    output_dir = args.results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.results_path,
        {
            "experiment": "adapter_token_information_probe",
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "embedding_bank": str(args.embedding_bank),
            "llm_model": args.llm_model,
            "checkpoint_dir": str(args.checkpoint_dir),
            "checkpoint_config": checkpoint_config,
            "feature_sets": feature_sets,
            "min_train_pos": args.min_train_pos,
            "min_val_pos": args.min_val_pos,
            "max_iter": args.max_iter,
            "c_value": args.c_value,
            "rows": rows,
            "comparisons": comparisons,
        },
    )
    write_csv(args.results_path.with_name(args.results_path.stem + "_long.csv"), rows)
    write_csv(args.results_path.with_name(args.results_path.stem + "_comparison.csv"), comparisons)
    print_summary(rows, comparisons)
    print(f"\nSaved results: {args.results_path}")
    print(f"Saved long CSV: {args.results_path.with_name(args.results_path.stem + '_long.csv')}")
    print(f"Saved comparison CSV: {args.results_path.with_name(args.results_path.stem + '_comparison.csv')}")


if __name__ == "__main__":
    main()
