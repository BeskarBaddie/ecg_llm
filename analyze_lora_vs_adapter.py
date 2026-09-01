from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_scp_code_performance import (
    clinical_group,
    compute_binary_metrics,
    load_jsonl,
    load_scp_metadata,
    write_csv,
    write_json,
)


DEFAULT_ADAPTER_RESULTS = Path(
    "outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_"
    "lr5e_6_batch1_accum4_adapter_full_all_codes_results.json"
)
DEFAULT_ADAPTER_PREDICTIONS = Path(
    "outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_"
    "lr5e_6_batch1_accum4_adapter_full_all_codes_predictions.jsonl"
)
DEFAULT_LORA_RESULTS = Path(
    "outputs/20260715_153929_20_epochs_llama_3_2_3b_csfm_tiny_linear_adapter_plus_lora_"
    "r8_alpha16_float16_llm_adapter_full_all_codes_results.json"
)
DEFAULT_LORA_PREDICTIONS = Path(
    "outputs/20260715_153929_20_epochs_llama_3_2_3b_csfm_tiny_linear_adapter_plus_lora_"
    "r8_alpha16_float16_llm_adapter_full_all_codes_predictions.jsonl"
)
DEFAULT_SCP_METADATA = Path("outputs/scp_code_comparison_smollm_vs_llama_with_descriptions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/lora_vs_adapter")
DEFAULT_FIGURE_DIR = Path("figures/lora_vs_adapter")


# Function: Load a JSON dictionary from disk.
# Inputs: path to a JSON file.
# Outputs: parsed JSON dictionary.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Count binary confusion matrix entries from prediction rows.
# Inputs: prediction rows with true_label and pred_label values.
# Outputs: tuple of true negatives, false positives, false negatives, and true positives.
def compute_counts(rows: List[Dict[str, Any]]) -> tuple[int, int, int, int]:
    counts = Counter((int(row["true_label"]), int(row["pred_label"])) for row in rows)
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    tp = counts[(1, 1)]
    return tn, fp, fn, tp


# Function: Compute binary classification metrics from prediction rows.
# Inputs: prediction rows with true_label and pred_label values.
# Outputs: dictionary of accuracy, balanced accuracy, precision, recall, F1, and counts.
def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tn, fp, fn, tp = compute_counts(rows)
    return compute_binary_metrics(tn=tn, fp=fp, fn=fn, tp=tp)


# Function: Calculate prediction-rate and error-rate diagnostics from prediction rows.
# Inputs: prediction rows with true_label and pred_label.
# Outputs: dictionary of class-bias and error-rate diagnostics.
def prediction_bias_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    true_counts = Counter(int(row["true_label"]) for row in rows)
    pred_counts = Counter(int(row["pred_label"]) for row in rows)
    tn, fp, fn, tp = compute_counts(rows)
    return {
        "n": total,
        "true_no": int(true_counts[0]),
        "true_yes": int(true_counts[1]),
        "pred_no": int(pred_counts[0]),
        "pred_yes": int(pred_counts[1]),
        "true_yes_rate": true_counts[1] / total if total else None,
        "pred_yes_rate": pred_counts[1] / total if total else None,
        "false_positive_rate_overall": fp / total if total else None,
        "false_negative_rate_overall": fn / total if total else None,
        "false_positive_rate_among_true_no": fp / (tn + fp) if (tn + fp) else None,
        "false_negative_rate_among_true_yes": fn / (tp + fn) if (tp + fn) else None,
    }


# Function: Summarize one set of predictions by an arbitrary row grouping.
# Inputs: prediction rows and a field/group lookup function.
# Outputs: dictionary mapping group name to metric dictionary.
def summarize_by_group(
    rows: List[Dict[str, Any]],
    group_fn: Any,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_fn(row)].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        metrics = compute_metrics(group_rows)
        bias = prediction_bias_metrics(group_rows)
        summary[group] = {**metrics, **bias}
    return summary


# Function: Create a row-wise delta table comparing adapter-only and LoRA summaries.
# Inputs: adapter summary, LoRA summary, key field name, optional metadata.
# Outputs: list of comparison rows.
def compare_summaries(
    adapter_summary: Dict[str, Dict[str, Any]],
    lora_summary: Dict[str, Dict[str, Any]],
    key_name: str,
    metadata: Dict[str, Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(adapter_summary) & set(lora_summary)):
        adapter = adapter_summary[key]
        lora = lora_summary[key]
        row: Dict[str, Any] = {
            key_name: key,
            "n": int(lora["n"]),
            "adapter_balanced_accuracy": adapter["balanced_accuracy"],
            "lora_balanced_accuracy": lora["balanced_accuracy"],
            "delta_balanced_accuracy": lora["balanced_accuracy"] - adapter["balanced_accuracy"],
            "adapter_accuracy": adapter["accuracy"],
            "lora_accuracy": lora["accuracy"],
            "delta_accuracy": lora["accuracy"] - adapter["accuracy"],
            "adapter_macro_f1": adapter["macro_f1"],
            "lora_macro_f1": lora["macro_f1"],
            "delta_macro_f1": lora["macro_f1"] - adapter["macro_f1"],
            "adapter_yes_recall": adapter["yes_recall"],
            "lora_yes_recall": lora["yes_recall"],
            "delta_yes_recall": lora["yes_recall"] - adapter["yes_recall"],
            "adapter_no_recall": adapter["no_recall"],
            "lora_no_recall": lora["no_recall"],
            "delta_no_recall": lora["no_recall"] - adapter["no_recall"],
            "adapter_pred_yes_rate": adapter["pred_yes_rate"],
            "lora_pred_yes_rate": lora["pred_yes_rate"],
            "delta_pred_yes_rate": lora["pred_yes_rate"] - adapter["pred_yes_rate"],
        }
        if metadata and key in metadata:
            row["description"] = metadata[key].get("description", key)
            row["clinical_group"] = clinical_group(metadata[key])
        rows.append(row)
    return rows


# Function: Load a readable font with fallback to PIL default.
# Inputs: point size.
# Outputs: PIL font object.
def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# Function: Draw a horizontal delta bar chart.
# Inputs: labels, delta values, output path, title, and x-axis label.
# Outputs: PNG figure written to disk.
def draw_delta_bar_chart(
    labels: Sequence[str],
    deltas: Sequence[float],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    width = 1500
    row_h = 56
    height = 160 + len(labels) * row_h
    margin_left, margin_right = 430, 90
    margin_top, margin_bottom = 100, 70
    plot_w = width - margin_left - margin_right
    max_abs = max(max(abs(x) for x in deltas), 0.05)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(24)
    axis_font = font(15)
    label_font = font(16)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 28), title, fill="#111827", font=title_font)

    zero_x = margin_left + plot_w / 2
    draw.line((zero_x, margin_top - 10, zero_x, height - margin_bottom), fill="#111827", width=2)
    for tick in np.linspace(-max_abs, max_abs, 7):
        x = zero_x + (tick / max_abs) * (plot_w / 2)
        draw.line((x, margin_top - 10, x, height - margin_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 28, height - margin_bottom + 14), f"{tick:+.2f}", fill="#374151", font=axis_font)

    for idx, (label, delta) in enumerate(zip(labels, deltas)):
        y = margin_top + idx * row_h
        draw.text((26, y + 9), label, fill="#111827", font=label_font)
        x = zero_x + (delta / max_abs) * (plot_w / 2)
        color = "#16a34a" if delta >= 0 else "#dc2626"
        draw.rectangle((min(zero_x, x), y + 10, max(zero_x, x), y + 39), fill=color)
        draw.text((x + (8 if delta >= 0 else -64), y + 10), f"{delta:+.3f}", fill="#111827", font=label_font)

    draw.text((margin_left + plot_w / 2 - 120, height - 32), x_label, fill="#111827", font=axis_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# Function: Draw LoRA and adapter-only validation training curves.
# Inputs: result JSONs and output path.
# Outputs: PNG figure written to disk.
def draw_epoch_curve(adapter_results: Dict[str, Any], lora_results: Dict[str, Any], output_path: Path) -> None:
    width, height = 1500, 820
    margin_left, margin_right = 110, 70
    margin_top, margin_bottom = 110, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(24)
    axis_font = font(15)

    title = "Validation Balanced Accuracy by Epoch: Adapter-only vs LoRA"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 28), title, fill="#111827", font=title_font)

    for tick in np.linspace(0.45, 0.80, 8):
        y = margin_top + plot_h - ((tick - 0.45) / 0.35) * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw.text((42, y - 9), f"{tick:.2f}", fill="#374151", font=axis_font)

    def pts(result: Dict[str, Any]) -> List[tuple[float, float]]:
        rows = result.get("epoch_history", [])
        max_epoch = max(20, max(int(row["epoch"]) for row in rows))
        points = []
        for row in rows:
            epoch = int(row["epoch"])
            bal = float(row["metrics"]["balanced_accuracy"])
            x = margin_left + ((epoch - 1) / max_epoch) * plot_w
            y = margin_top + plot_h - ((bal - 0.45) / 0.35) * plot_h
            points.append((x, y))
        return points

    for result, color, name, y_legend in [
        (adapter_results, "#2563eb", "Adapter-only", height - 52),
        (lora_results, "#dc2626", "Adapter + LoRA", height - 30),
    ]:
        points = pts(result)
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.line((margin_left, y_legend + 7, margin_left + 45, y_legend + 7), fill=color, width=4)
        draw.text((margin_left + 55, y_legend), name, fill="#111827", font=axis_font)

    for epoch in range(1, 21, 2):
        x = margin_left + ((epoch - 1) / 20) * plot_w
        draw.text((x - 8, height - 78), str(epoch), fill="#374151", font=axis_font)
    draw.text((width - 140, height - 78), "Epoch", fill="#111827", font=axis_font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#111827", width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill="#111827", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# Function: Print a compact console summary for the LoRA comparison.
# Inputs: summary dictionary and comparison rows.
# Outputs: None; prints findings.
def print_summary(summary: Dict[str, Any], clinical_rows: List[Dict[str, Any]], code_rows: List[Dict[str, Any]]) -> None:
    print("Overall comparison")
    for model_name, block in summary["overall"].items():
        print(
            f"{model_name:14s} bal={block['balanced_accuracy']:.3f} "
            f"acc={block['accuracy']:.3f} f1={block['macro_f1']:.3f} "
            f"yes_rec={block['yes_recall']:.3f} no_rec={block['no_recall']:.3f} "
            f"pred_yes_rate={block['pred_yes_rate']:.3f}"
        )

    print("\nClinical group deltas: LoRA minus adapter-only")
    for row in sorted(clinical_rows, key=lambda item: item["delta_balanced_accuracy"], reverse=True):
        print(
            f"{row['clinical_group']:30s} n={row['n']:>4} "
            f"delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"delta_yes={row['delta_yes_recall']:+.3f} "
            f"delta_no={row['delta_no_recall']:+.3f}"
        )

    eligible_codes = [row for row in code_rows if int(row["n"]) >= 50]
    print("\nTop per-code LoRA gains")
    for row in sorted(eligible_codes, key=lambda item: item["delta_balanced_accuracy"], reverse=True)[:10]:
        print(
            f"{row['scp_code']:8s} n={row['n']:>4} delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"{row.get('description', '')}"
        )
    print("\nTop per-code LoRA drops")
    for row in sorted(eligible_codes, key=lambda item: item["delta_balanced_accuracy"])[:10]:
        print(
            f"{row['scp_code']:8s} n={row['n']:>4} delta_bal={row['delta_balanced_accuracy']:+.3f} "
            f"{row.get('description', '')}"
        )


# Function: Run LoRA-vs-adapter comparison analyses and figure generation.
# Inputs: command-line arguments.
# Outputs: CSV, JSON, and PNG artifacts.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-results", type=Path, default=DEFAULT_ADAPTER_RESULTS)
    parser.add_argument("--adapter-predictions", type=Path, default=DEFAULT_ADAPTER_PREDICTIONS)
    parser.add_argument("--lora-results", type=Path, default=DEFAULT_LORA_RESULTS)
    parser.add_argument("--lora-predictions", type=Path, default=DEFAULT_LORA_PREDICTIONS)
    parser.add_argument("--scp-metadata-csv", type=Path, default=DEFAULT_SCP_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    adapter_results = load_json(args.adapter_results)
    lora_results = load_json(args.lora_results)
    adapter_predictions = load_jsonl(args.adapter_predictions)
    lora_predictions = load_jsonl(args.lora_predictions)
    metadata = load_scp_metadata(args.scp_metadata_csv)

    adapter_overall = {**compute_metrics(adapter_predictions), **prediction_bias_metrics(adapter_predictions)}
    lora_overall = {**compute_metrics(lora_predictions), **prediction_bias_metrics(lora_predictions)}

    adapter_by_code = summarize_by_group(adapter_predictions, lambda row: str(row["target_scp_code"]))
    lora_by_code = summarize_by_group(lora_predictions, lambda row: str(row["target_scp_code"]))
    code_rows = compare_summaries(adapter_by_code, lora_by_code, "scp_code", metadata=metadata)

    adapter_by_clinical = summarize_by_group(
        adapter_predictions,
        lambda row: clinical_group(metadata.get(str(row["target_scp_code"]), {"description": str(row["target_scp_code"])})),
    )
    lora_by_clinical = summarize_by_group(
        lora_predictions,
        lambda row: clinical_group(metadata.get(str(row["target_scp_code"]), {"description": str(row["target_scp_code"])})),
    )
    clinical_rows = compare_summaries(adapter_by_clinical, lora_by_clinical, "clinical_group")

    summary = {
        "adapter_results_path": str(args.adapter_results),
        "lora_results_path": str(args.lora_results),
        "adapter_status": adapter_results.get("status"),
        "lora_status": lora_results.get("status"),
        "adapter_epochs_completed": len(adapter_results.get("epoch_history", [])),
        "lora_epochs_completed": len(lora_results.get("epoch_history", [])),
        "adapter_best_epoch": adapter_results.get("best_epoch"),
        "lora_best_epoch": lora_results.get("best_epoch"),
        "adapter_trainable_parameters": adapter_results.get("total_trainable_parameters")
        or adapter_results.get("adapter_parameters"),
        "lora_trainable_parameters": lora_results.get("total_trainable_parameters"),
        "overall": {
            "adapter_only": adapter_overall,
            "adapter_plus_lora": lora_overall,
            "delta": {
                key: lora_overall[key] - adapter_overall[key]
                for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "yes_recall",
                    "no_recall",
                    "pred_yes_rate",
                    "false_positive_rate_among_true_no",
                    "false_negative_rate_among_true_yes",
                ]
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "lora_vs_adapter_summary.json", summary)
    write_csv(args.output_dir / "lora_vs_adapter_by_clinical_group.csv", clinical_rows)
    write_csv(args.output_dir / "lora_vs_adapter_by_scp_code.csv", code_rows)

    draw_delta_bar_chart(
        labels=[row["clinical_group"] for row in sorted(clinical_rows, key=lambda item: item["delta_balanced_accuracy"])],
        deltas=[row["delta_balanced_accuracy"] for row in sorted(clinical_rows, key=lambda item: item["delta_balanced_accuracy"])],
        output_path=args.figure_dir / "lora_delta_balanced_accuracy_by_clinical_group.png",
        title="LoRA Effect by Clinical Group",
        x_label="Delta balanced accuracy (LoRA - adapter-only)",
    )
    draw_delta_bar_chart(
        labels=[row["clinical_group"] for row in sorted(clinical_rows, key=lambda item: item["delta_yes_recall"])],
        deltas=[row["delta_yes_recall"] for row in sorted(clinical_rows, key=lambda item: item["delta_yes_recall"])],
        output_path=args.figure_dir / "lora_delta_yes_recall_by_clinical_group.png",
        title="LoRA Effect on Positive Recall by Clinical Group",
        x_label="Delta yes recall (LoRA - adapter-only)",
    )
    draw_epoch_curve(
        adapter_results=adapter_results,
        lora_results=lora_results,
        output_path=args.figure_dir / "lora_vs_adapter_epoch_curve.png",
    )

    print_summary(summary, clinical_rows, code_rows)
    print(f"\nSaved summary JSON: {args.output_dir / 'lora_vs_adapter_summary.json'}")
    print(f"Saved clinical-group CSV: {args.output_dir / 'lora_vs_adapter_by_clinical_group.csv'}")
    print(f"Saved SCP-code CSV: {args.output_dir / 'lora_vs_adapter_by_scp_code.csv'}")
    print(f"Saved figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
