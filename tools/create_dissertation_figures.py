from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

ADAPTER_1B = OUTPUTS / (
    "20260704_183337_20_epochs_llama_3_2_1b_instruct_float32_lr1e_5_batch2_"
    "gradclip0_5_real_train_rea_adapter_full_all_codes_results.json"
)
ADAPTER_3B = OUTPUTS / (
    "20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_"
    "lr5e_6_batch1_accum4_adapter_full_all_codes_results.json"
)
LORA_3B = OUTPUTS / (
    "20260715_153929_20_epochs_llama_3_2_3b_csfm_tiny_linear_adapter_plus_lora_"
    "r8_alpha16_float16_llm_adapter_full_all_codes_results.json"
)
CLINICAL_GROUP_CSV = OUTPUTS / "llama3b_by_clinical_group_performance.csv"
LOCALIZATION_CSV = OUTPUTS / "llama3b_by_question_localization_performance.csv"


# Function: Load a JSON dictionary from disk.
# Inputs: path to a JSON file.
# Outputs: parsed JSON dictionary.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load a CSV file into dictionaries.
# Inputs: path to a CSV file.
# Outputs: list of row dictionaries.
def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Function: Load a readable font with fallback to PIL default.
# Inputs: point size.
# Outputs: PIL font object.
def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# Function: Draw multiline chart title text.
# Inputs: draw context, title, canvas width, and y coordinate.
# Outputs: y coordinate after the title.
def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int, y: int) -> int:
    title_font = font(26)
    subtitle_font = font(15)
    lines = title.split("\n")
    for idx, line in enumerate(lines):
        fnt = title_font if idx == 0 else subtitle_font
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), line, fill="#111827", font=fnt)
        y += bbox[3] - bbox[1] + (8 if idx == 0 else 4)
    return y


# Function: Format a metric value for chart labels.
# Inputs: floating point metric.
# Outputs: string rounded to three decimals.
def fmt(value: float) -> str:
    return f"{value:.3f}"


# Function: Draw a grouped bar chart for model-level metrics.
# Inputs: model labels, metric names, metric values, output path, and title.
# Outputs: PNG file written to disk.
def draw_grouped_bar_chart(
    labels: Sequence[str],
    metrics: Sequence[str],
    values: Sequence[Sequence[float]],
    output_path: Path,
    title: str,
) -> None:
    width, height = 1500, 900
    margin_left, margin_right = 120, 60
    margin_bottom, margin_top = 120, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 30)
    axis_font = font(16)
    label_font = font(15)

    for tick in range(0, 11):
        y = margin_top + plot_h - (tick / 10) * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw.text((45, y - 9), f"{tick/10:.1f}", fill="#374151", font=axis_font)

    group_w = plot_w / len(labels)
    bar_w = group_w / (len(metrics) + 1)
    for i, label in enumerate(labels):
        x0 = margin_left + i * group_w
        for j, metric in enumerate(metrics):
            value = values[i][j]
            bar_x0 = x0 + (j + 0.5) * bar_w
            bar_x1 = bar_x0 + bar_w * 0.8
            bar_y0 = margin_top + plot_h - value * plot_h
            draw.rectangle((bar_x0, bar_y0, bar_x1, margin_top + plot_h), fill=colors[j])
            draw.text((bar_x0, bar_y0 - 24), fmt(value), fill="#111827", font=label_font)
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((x0 + group_w / 2 - (bbox[2] - bbox[0]) / 2, height - 90), label, fill="#111827", font=axis_font)

    legend_x = margin_left
    legend_y = height - 45
    for j, metric in enumerate(metrics):
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=colors[j])
        draw.text((legend_x + 26, legend_y - 2), metric, fill="#111827", font=axis_font)
        legend_x += 190

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#111827", width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill="#111827", width=2)
    img.save(output_path)


# Function: Draw a line chart with two y-series scaled to [0, 1].
# Inputs: epoch rows, output path, and title.
# Outputs: PNG file written to disk.
def draw_training_curve(epoch_rows: List[Dict[str, Any]], output_path: Path, title: str) -> None:
    width, height = 1500, 850
    margin_left, margin_right = 110, 70
    margin_bottom, margin_top = 100, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 30)
    axis_font = font(16)
    label_font = font(15)

    epochs = [int(row["epoch"]) for row in epoch_rows]
    bal = [float(row["metrics"]["balanced_accuracy"]) for row in epoch_rows]
    losses = [float(row["train_loss"]) for row in epoch_rows]
    max_loss = max(losses) if losses else 1.0
    loss_scaled = [loss / max_loss for loss in losses]

    for tick in range(0, 11):
        y = margin_top + plot_h - (tick / 10) * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw.text((42, y - 9), f"{tick/10:.1f}", fill="#374151", font=axis_font)

    def point(idx: int, value: float) -> Tuple[float, float]:
        x = margin_left + (idx / max(len(epoch_rows) - 1, 1)) * plot_w
        y = margin_top + plot_h - value * plot_h
        return x, y

    for series, color in [(bal, "#2563eb"), (loss_scaled, "#dc2626")]:
        pts = [point(i, value) for i, value in enumerate(series)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    for idx, epoch in enumerate(epochs):
        if idx % 2 == 0 or idx == len(epochs) - 1:
            x, _ = point(idx, 0)
            draw.text((x - 8, height - 78), str(epoch), fill="#374151", font=axis_font)

    legend_y = height - 42
    draw.line((margin_left, legend_y + 8, margin_left + 45, legend_y + 8), fill="#2563eb", width=4)
    draw.text((margin_left + 55, legend_y), "Balanced accuracy", fill="#111827", font=axis_font)
    draw.line((margin_left + 260, legend_y + 8, margin_left + 305, legend_y + 8), fill="#dc2626", width=4)
    draw.text((margin_left + 315, legend_y), f"Training loss (scaled; max={max_loss:.3f})", fill="#111827", font=axis_font)
    draw.text((width - 155, height - 78), "Epoch", fill="#111827", font=axis_font)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#111827", width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill="#111827", width=2)
    img.save(output_path)


# Function: Draw a horizontal bar chart for one metric.
# Inputs: row labels, values, output path, title, and x-axis label.
# Outputs: PNG file written to disk.
def draw_horizontal_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    width = 1500
    row_h = 62
    height = 170 + row_h * len(labels)
    margin_left, margin_right = 420, 80
    margin_top, margin_bottom = 125, 70
    plot_w = width - margin_left - margin_right
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 28)
    axis_font = font(16)
    label_font = font(17)

    for tick in range(0, 11):
        x = margin_left + tick / 10 * plot_w
        draw.line((x, margin_top - 10, x, height - margin_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 12, height - margin_bottom + 14), f"{tick/10:.1f}", fill="#374151", font=axis_font)

    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * row_h
        bar_w = value * plot_w
        color = "#2563eb" if value >= 0.75 else "#16a34a" if value >= 0.70 else "#dc2626"
        draw.text((30, y + 10), label, fill="#111827", font=label_font)
        draw.rectangle((margin_left, y + 8, margin_left + bar_w, y + 40), fill=color)
        draw.text((margin_left + bar_w + 10, y + 10), fmt(value), fill="#111827", font=label_font)

    draw.text((margin_left + plot_w / 2 - 90, height - 32), x_label, fill="#111827", font=axis_font)
    draw.line((margin_left, margin_top - 10, margin_left, height - margin_bottom), fill="#111827", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#111827", width=2)
    img.save(output_path)


# Function: Extract headline metric values from a result JSON.
# Inputs: result JSON path.
# Outputs: tuple of accuracy, balanced accuracy, and macro-F1.
def headline_metrics(path: Path) -> Tuple[float, float, float]:
    result = load_json(path)
    metrics = result["metrics"]
    return (
        float(metrics["accuracy"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["macro_f1"]),
    )


# Function: Generate all initial dissertation figures from current local results.
# Inputs: none.
# Outputs: PNG files in the figures directory.
def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    comparison_paths = [
        ("Llama 1B\nadapter", ADAPTER_1B),
        ("Llama 3B\nadapter", ADAPTER_3B),
    ]
    if LORA_3B.exists():
        comparison_paths.append(("Llama 3B\nadapter+LoRA", LORA_3B))

    labels = [label for label, path in comparison_paths if path.exists()]
    values = [headline_metrics(path) for _, path in comparison_paths if path.exists()]
    draw_grouped_bar_chart(
        labels=labels,
        metrics=["Accuracy", "Balanced accuracy", "Macro-F1"],
        values=values,
        output_path=FIGURES / "adapter_model_comparison.png",
        title="Adapter Model Comparison\nValidation metrics on binary ECG-QA",
    )

    adapter_3b = load_json(ADAPTER_3B)
    draw_training_curve(
        epoch_rows=adapter_3b["epoch_history"],
        output_path=FIGURES / "adapter_3b_training_curve.png",
        title="Llama-3.2-3B Adapter Training Curve\nBalanced accuracy and scaled training loss by epoch",
    )

    clinical_rows = sorted(
        load_csv(CLINICAL_GROUP_CSV),
        key=lambda row: float(row["balanced_accuracy"]),
    )
    draw_horizontal_bar_chart(
        labels=[row["clinical_group"] for row in clinical_rows],
        values=[float(row["balanced_accuracy"]) for row in clinical_rows],
        output_path=FIGURES / "adapter_3b_clinical_group_balanced_accuracy.png",
        title="Clinical Group Performance\nLlama-3.2-3B adapter validation balanced accuracy",
        x_label="Balanced accuracy",
    )

    localization_rows = sorted(
        load_csv(LOCALIZATION_CSV),
        key=lambda row: float(row["balanced_accuracy"]),
    )
    draw_horizontal_bar_chart(
        labels=[row["group"] for row in localization_rows],
        values=[float(row["balanced_accuracy"]) for row in localization_rows],
        output_path=FIGURES / "adapter_3b_localization_balanced_accuracy.png",
        title="Question Localization Performance\nWhole-ECG versus localized ECG-QA questions",
        x_label="Balanced accuracy",
    )

    print("Saved figures:")
    for path in sorted(FIGURES.glob("adapter_*.png")):
        print(path)


if __name__ == "__main__":
    main()
