from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures" / "mechanistic_diagnostics"
TABLES = OUTPUTS / "mechanistic_diagnostics"

TOKEN_ABLATION_SUMMARY = OUTPUTS / "ecg_token_ablation_analysis" / "ecg_token_ablation_summary.json"
TOKEN_ABLATION_BY_CLINICAL = OUTPUTS / "ecg_token_ablation_analysis" / "ecg_token_ablation_by_clinical_group.csv"
ADAPTER_PROBE_COMPARISON = OUTPUTS / "adapter_token_probe_3b_results_comparison.csv"
QUESTION_SWAP_SUMMARY = OUTPUTS / "ecg_question_swap_analysis" / "ecg_question_swap_summary.json"
QUESTION_SWAP_BY_CODE = OUTPUTS / "ecg_question_swap_analysis" / "ecg_question_swap_by_code.csv"
QUESTION_SWAP_PAIR_BY_CODE = OUTPUTS / "ecg_question_swap_analysis" / "ecg_question_swap_same_question_pairs_by_code.csv"


# Function: Load a JSON file from disk.
# Inputs: JSON path.
# Outputs: parsed JSON object.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Function: Load a CSV file into row dictionaries.
# Inputs: CSV path.
# Outputs: list of row dictionaries.
def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Function: Write dictionaries to a CSV file.
# Inputs: output path and row dictionaries.
# Outputs: None; writes CSV.
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Function: Write a markdown table.
# Inputs: output path, rows, column labels, and source row keys.
# Outputs: None; writes markdown table.
def write_markdown_table(path: Path, rows: List[Dict[str, Any]], headers: Sequence[str], keys: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


# Function: Format metric values for tables and labels.
# Inputs: value and decimal places.
# Outputs: formatted string.
def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return f"{float(value):.{digits}f}"


# Function: Calculate the median of a numeric sequence.
# Inputs: sequence of numeric values.
# Outputs: median value using the average of the two middle values for even-length sequences.
def median(values: Sequence[float]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        raise ValueError("Cannot calculate median of an empty sequence.")
    midpoint = n // 2
    if n % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


# Function: Draw centered title and subtitle text.
# Inputs: draw context, title text, canvas width, and starting y.
# Outputs: y coordinate after title block.
def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int, y: int) -> int:
    title_font = font(26)
    subtitle_font = font(16)
    for idx, line in enumerate(title.split("\n")):
        fnt = title_font if idx == 0 else subtitle_font
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), line, fill="#111827", font=fnt)
        y += bbox[3] - bbox[1] + (8 if idx == 0 else 4)
    return y


# Function: Draw a vertical grouped bar chart scaled to [0, 1].
# Inputs: group labels, metric labels, values, output path, and title.
# Outputs: None; writes PNG.
def draw_grouped_bar_chart(
    labels: Sequence[str],
    metric_labels: Sequence[str],
    values: Sequence[Sequence[float]],
    output_path: Path,
    title: str,
) -> None:
    width, height = 1700, 980
    margin_left, margin_right = 120, 70
    margin_top, margin_bottom = 135, 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 28)
    axis_font = font(16)
    label_font = font(14)

    for tick in range(0, 11):
        y = margin_top + plot_h - (tick / 10) * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        draw.text((44, y - 9), f"{tick / 10:.1f}", fill="#374151", font=axis_font)

    group_w = plot_w / len(labels)
    bar_w = group_w / (len(metric_labels) + 1)
    for i, label in enumerate(labels):
        x0 = margin_left + i * group_w
        for j, metric in enumerate(metric_labels):
            value = values[i][j]
            bar_x0 = x0 + (j + 0.45) * bar_w
            bar_x1 = bar_x0 + bar_w * 0.82
            bar_y0 = margin_top + plot_h - value * plot_h
            draw.rectangle((bar_x0, bar_y0, bar_x1, margin_top + plot_h), fill=colors[j])
            draw.text((bar_x0 - 5, bar_y0 - 22), fmt(value), fill="#111827", font=label_font)
        bbox = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((x0 + group_w / 2 - (bbox[2] - bbox[0]) / 2, height - 112), label, fill="#111827", font=axis_font)

    legend_x = margin_left
    legend_y = height - 58
    for j, metric in enumerate(metric_labels):
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=colors[j])
        draw.text((legend_x + 26, legend_y - 2), metric, fill="#111827", font=axis_font)
        legend_x += 250

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#111827", width=2)
    draw.line((margin_left, margin_top + plot_h, width - margin_right, margin_top + plot_h), fill="#111827", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# Function: Draw a horizontal bar chart scaled to [0, 1].
# Inputs: labels, values, output path, title, and x-axis label.
# Outputs: None; writes PNG.
def draw_horizontal_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    width = 1700
    row_h = 62
    height = 175 + row_h * len(labels)
    margin_left, margin_right = 430, 90
    margin_top, margin_bottom = 125, 74
    plot_w = width - margin_left - margin_right
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 28)
    axis_font = font(16)
    label_font = font(17)

    for tick in range(0, 11):
        x = margin_left + tick / 10 * plot_w
        draw.line((x, margin_top - 10, x, height - margin_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 12, height - margin_bottom + 14), f"{tick / 10:.1f}", fill="#374151", font=axis_font)

    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + i * row_h
        color = "#2563eb" if value >= 0.80 else "#16a34a" if value >= 0.70 else "#dc2626"
        draw.text((28, y + 9), label, fill="#111827", font=label_font)
        draw.rectangle((margin_left, y + 8, margin_left + value * plot_w, y + 40), fill=color)
        draw.text((margin_left + value * plot_w + 10, y + 10), fmt(value), fill="#111827", font=label_font)

    draw.text((margin_left + plot_w / 2 - 120, height - 33), x_label, fill="#111827", font=axis_font)
    draw.line((margin_left, margin_top - 10, margin_left, height - margin_bottom), fill="#111827", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#111827", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# Function: Draw a scatter plot with a y=x reference line.
# Inputs: points, labels, output path, title, x-axis label, and y-axis label.
# Outputs: None; writes PNG.
def draw_scatter(
    points: Sequence[tuple[float, float, str]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 1300, 1100
    margin_left, margin_right = 130, 70
    margin_top, margin_bottom = 130, 110
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width, 28)
    axis_font = font(16)
    point_font = font(13)
    xmin, xmax = 0.45, 1.02
    ymin, ymax = 0.45, 1.02

    def sx(x: float) -> float:
        return margin_left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return margin_top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    for tick_i in range(5, 11):
        tick = tick_i / 10
        x = sx(tick)
        y = sy(tick)
        draw.line((x, margin_top, x, margin_top + plot_h), fill="#e5e7eb", width=1)
        draw.line((margin_left, y, margin_left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((x - 14, margin_top + plot_h + 14), f"{tick:.1f}", fill="#374151", font=axis_font)
        draw.text((58, y - 9), f"{tick:.1f}", fill="#374151", font=axis_font)

    draw.line((sx(0.45), sy(0.45), sx(1.0), sy(1.0)), fill="#9ca3af", width=2)
    for x, y, label in points:
        color = "#2563eb" if y >= x - 0.02 else "#dc2626"
        px, py = sx(x), sy(y)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        if label in {"AFIB", "CRBBB", "PACE", "LVH", "STD_", "INVT", "NDT"}:
            draw.text((px + 8, py - 8), label, fill="#111827", font=point_font)

    draw.text((margin_left + plot_w / 2 - 100, height - 45), x_label, fill="#111827", font=axis_font)
    draw.text((24, margin_top + plot_h / 2), y_label, fill="#111827", font=axis_font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#111827", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="#111827", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# Function: Build summary tables from the three mechanistic diagnostics.
# Inputs: none; reads fixed local outputs.
# Outputs: dictionary containing table rows.
def build_tables() -> Dict[str, List[Dict[str, Any]]]:
    token = load_json(TOKEN_ABLATION_SUMMARY)
    swap = load_json(QUESTION_SWAP_SUMMARY)
    probe_rows = load_csv(ADAPTER_PROBE_COMPARISON)
    swap_code = load_csv(QUESTION_SWAP_BY_CODE)
    swap_pair = load_csv(QUESTION_SWAP_PAIR_BY_CODE)

    token_rows = []
    for mode in ["real", "zero", "mean", "shuffled", "random"]:
        m = token["metrics_by_mode"][mode]
        token_rows.append(
            {
                "condition": mode,
                "accuracy": fmt(m["accuracy"]),
                "balanced_accuracy": fmt(m["balanced_accuracy"]),
                "yes_recall": fmt(m["yes_recall"]),
                "no_recall": fmt(m["no_recall"]),
                "pred_yes": m["pred_yes_count"],
                "pred_no": m["pred_no_count"],
            }
        )

    probe_summary = []
    for feature in ["csfm", "adapter_mean", "adapter_flat"]:
        matching = [row for row in load_json(OUTPUTS / "adapter_token_probe_3b_results.json")["rows"] if row["feature_set"] == feature]
        aucs = [float(row["roc_auc"]) for row in matching if row["roc_auc"] is not None]
        bals = [float(row["balanced_accuracy"]) for row in matching]
        probe_summary.append(
            {
                "feature_set": feature,
                "codes": len(matching),
                "mean_auc": fmt(sum(aucs) / len(aucs)),
                "median_auc": fmt(median(aucs)),
                "mean_balanced_accuracy": fmt(sum(bals) / len(bals)),
            }
        )

    swap_rows = []
    for test_type, m in swap["metrics_by_type"].items():
        swap_rows.append(
            {
                "test_type": test_type,
                "n": m["n"],
                "accuracy": fmt(m["accuracy"]),
                "balanced_accuracy": fmt(m["balanced_accuracy"]),
                "yes_recall": fmt(m["yes_recall"]),
                "no_recall": fmt(m["no_recall"]),
            }
        )

    swap_code_rows = []
    pair_by_code = {row["scp_code"]: row for row in swap_pair}
    for row in swap_code:
        pair = pair_by_code.get(row["scp_code"], {})
        swap_code_rows.append(
            {
                "scp_code": row["scp_code"],
                "n": row["n"],
                "balanced_accuracy": fmt(row["balanced_accuracy"]),
                "yes_recall": fmt(row["yes_recall"]),
                "no_recall": fmt(row["no_recall"]),
                "pair_both_correct_rate": fmt(pair.get("both_correct_rate")),
                "pair_flip_rate": fmt(pair.get("prediction_flip_rate")),
            }
        )

    probe_drops = []
    for row in sorted(probe_rows, key=lambda item: float(item["adapter_mean_delta_auc_vs_csfm"]))[:10]:
        probe_drops.append(
            {
                "scp_code": row["scp_code"],
                "csfm_auc": fmt(row["csfm_roc_auc"]),
                "adapter_mean_auc": fmt(row["adapter_mean_roc_auc"]),
                "delta_auc": fmt(row["adapter_mean_delta_auc_vs_csfm"]),
            }
        )

    return {
        "token_ablation": token_rows,
        "adapter_probe_summary": probe_summary,
        "swap_by_type": swap_rows,
        "swap_by_code": swap_code_rows,
        "largest_adapter_probe_drops": probe_drops,
    }


# Function: Generate figures and tables for dissertation mechanistic diagnostics.
# Inputs: none.
# Outputs: PNG, CSV, and Markdown files.
def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tables = build_tables()

    write_csv(TABLES / "table_token_ablation.csv", tables["token_ablation"])
    write_csv(TABLES / "table_adapter_probe_summary.csv", tables["adapter_probe_summary"])
    write_csv(TABLES / "table_question_swap_by_type.csv", tables["swap_by_type"])
    write_csv(TABLES / "table_question_swap_by_code.csv", tables["swap_by_code"])
    write_csv(TABLES / "table_largest_adapter_probe_drops.csv", tables["largest_adapter_probe_drops"])

    write_markdown_table(
        TABLES / "table_token_ablation.md",
        tables["token_ablation"],
        ["Condition", "Accuracy", "Balanced Accuracy", "Yes Recall", "No Recall", "Pred Yes", "Pred No"],
        ["condition", "accuracy", "balanced_accuracy", "yes_recall", "no_recall", "pred_yes", "pred_no"],
    )
    write_markdown_table(
        TABLES / "table_adapter_probe_summary.md",
        tables["adapter_probe_summary"],
        ["Feature Set", "Codes", "Mean AUC", "Median AUC", "Mean Balanced Accuracy"],
        ["feature_set", "codes", "mean_auc", "median_auc", "mean_balanced_accuracy"],
    )
    write_markdown_table(
        TABLES / "table_question_swap_by_type.md",
        tables["swap_by_type"],
        ["Test Type", "n", "Accuracy", "Balanced Accuracy", "Yes Recall", "No Recall"],
        ["test_type", "n", "accuracy", "balanced_accuracy", "yes_recall", "no_recall"],
    )
    write_markdown_table(
        TABLES / "table_question_swap_by_code.md",
        tables["swap_by_code"],
        ["SCP Code", "n", "Balanced Accuracy", "Yes Recall", "No Recall", "Pair Both-Correct", "Pair Flip"],
        ["scp_code", "n", "balanced_accuracy", "yes_recall", "no_recall", "pair_both_correct_rate", "pair_flip_rate"],
    )

    token = load_json(TOKEN_ABLATION_SUMMARY)
    token_labels = ["Real", "Zero", "Mean", "Shuffled", "Random"]
    token_modes = ["real", "zero", "mean", "shuffled", "random"]
    draw_grouped_bar_chart(
        labels=token_labels,
        metric_labels=["Balanced accuracy", "Yes recall", "No recall"],
        values=[
            [
                float(token["metrics_by_mode"][mode]["balanced_accuracy"]),
                float(token["metrics_by_mode"][mode]["yes_recall"]),
                float(token["metrics_by_mode"][mode]["no_recall"]),
            ]
            for mode in token_modes
        ],
        output_path=FIGURES / "token_ablation_metrics.png",
        title="ECG Soft-Token Ablation\nReal ECG tokens outperform zero, mean, shuffled, and random controls",
    )

    probe_rows = load_csv(ADAPTER_PROBE_COMPARISON)
    draw_scatter(
        points=[
            (
                float(row["csfm_roc_auc"]),
                float(row["adapter_mean_roc_auc"]),
                row["scp_code"],
            )
            for row in probe_rows
        ],
        output_path=FIGURES / "csfm_vs_adapter_token_auc.png",
        title="SCP-Code Separability Before and After Adapter Projection\nPoints near y=x indicate preserved ECG information",
        x_label="CSFM embedding AUC",
        y_label="Adapter-token AUC",
    )

    swap_code = load_csv(QUESTION_SWAP_BY_CODE)
    sorted_swap = sorted(swap_code, key=lambda row: float(row["balanced_accuracy"]), reverse=True)
    draw_horizontal_bar_chart(
        labels=[row["scp_code"] for row in sorted_swap],
        values=[float(row["balanced_accuracy"]) for row in sorted_swap],
        output_path=FIGURES / "question_swap_balanced_accuracy_by_code.png",
        title="Counterfactual Swap Performance by SCP Code\nGrounding is strongest for rhythm, conduction, and device-like patterns",
        x_label="Balanced accuracy",
    )

    pair_rows = load_csv(QUESTION_SWAP_PAIR_BY_CODE)
    sorted_pair = sorted(pair_rows, key=lambda row: float(row["both_correct_rate"]), reverse=True)
    draw_grouped_bar_chart(
        labels=[row["scp_code"] for row in sorted_pair],
        metric_labels=["Both-correct rate", "Flip rate"],
        values=[
            [float(row["both_correct_rate"]), float(row["prediction_flip_rate"])]
            for row in sorted_pair
        ],
        output_path=FIGURES / "same_question_pair_flip_by_code.png",
        title="Same-Question ECG Swap Pair Metrics\nCorrect flips indicate ECG-specific grounding",
    )

    print(f"Saved tables: {TABLES}")
    print(f"Saved figures: {FIGURES}")


if __name__ == "__main__":
    main()
