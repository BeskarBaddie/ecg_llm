from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_PATH = Path("outputs/supervisor_progress_report_adapter_ecgqa.docx")


BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
LIGHT_GRAY = "F2F4F7"
BORDER = "D9E2F3"


# Function: Add or update a paragraph style.
# Inputs: document, style name, font size, color, bold flag, and spacing values.
# Outputs: None; mutates the document style definition.
def configure_paragraph_style(
    doc: Document,
    style_name: str,
    size_pt: float,
    color_hex: str,
    bold: bool = False,
    before_pt: float = 0,
    after_pt: float = 6,
    line_spacing: float = 1.1,
) -> None:
    style = doc.styles[style_name]
    style.font.name = "Calibri"
    style.font.size = Pt(size_pt)
    style.font.color.rgb = RGBColor.from_string(color_hex)
    style.font.bold = bold
    style.paragraph_format.space_before = Pt(before_pt)
    style.paragraph_format.space_after = Pt(after_pt)
    style.paragraph_format.line_spacing = line_spacing


# Function: Apply a solid background fill to a table cell.
# Inputs: table cell and hex color string.
# Outputs: None; mutates table cell XML.
def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


# Function: Set table cell margins in twentieths of a point.
# Inputs: table cell and top/bottom/start/end margin values.
# Outputs: None; mutates table cell XML.
def set_cell_margins(cell, top: int = 80, bottom: int = 80, start: int = 120, end: int = 120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, value in {"top": top, "bottom": bottom, "start": start, "end": end}.items():
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


# Function: Set explicit table width and column widths.
# Inputs: table and a list of column widths in inches.
# Outputs: None; mutates table geometry.
def set_table_widths(table, widths: list[float]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            cell = row.cells[idx]
            cell.width = Inches(width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


# Function: Add a formatted table with a shaded header row.
# Inputs: document, headers, rows, and column widths.
# Outputs: created table object.
def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_cells = table.rows[0].cells
    for idx, header in enumerate(headers):
        header_cells[idx].text = header
        shade_cell(header_cells[idx], LIGHT_GRAY)
        for paragraph in header_cells[idx].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
    for row_values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row_values):
            cells[idx].text = value
            for paragraph in cells[idx].paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    set_table_widths(table, widths)
    return table


# Function: Add controlled spacing after a table.
# Inputs: document and spacing size in points.
# Outputs: empty paragraph used only for visual separation.
def add_table_spacer(doc: Document, after_pt: float = 4):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(after_pt)
    paragraph.paragraph_format.line_spacing = 1
    return paragraph


# Function: Add a short bold-label paragraph.
# Inputs: document, label, and body text.
# Outputs: created paragraph.
def add_labeled_paragraph(doc: Document, label: str, body: str):
    p = doc.add_paragraph()
    p.style = doc.styles["Normal"]
    run = p.add_run(f"{label}: ")
    run.bold = True
    p.add_run(body)
    return p


# Function: Add a compact bullet paragraph.
# Inputs: document and bullet text.
# Outputs: created paragraph.
def add_bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    return p


# Function: Configure page setup and base styles.
# Inputs: document.
# Outputs: None; mutates document.
def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    configure_paragraph_style(doc, "Normal", 11, "000000", after_pt=6, line_spacing=1.1)
    configure_paragraph_style(doc, "Heading 1", 16, BLUE, bold=True, before_pt=16, after_pt=8)
    configure_paragraph_style(doc, "Heading 2", 13, BLUE, bold=True, before_pt=12, after_pt=6)
    configure_paragraph_style(doc, "Heading 3", 12, DARK_BLUE, bold=True, before_pt=8, after_pt=4)


# Function: Add a quiet running footer with page number placeholder text.
# Inputs: document.
# Outputs: None; mutates document footer.
def add_footer(doc: Document) -> None:
    footer = doc.sections[0].footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Progress report | ")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(90, 90, 90)
    run = paragraph.add_run("July 2026")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(90, 90, 90)


# Function: Build the supervisor progress report DOCX.
# Inputs: output path.
# Outputs: None; writes the DOCX file.
def build_report(output_path: Path) -> None:
    doc = Document()
    configure_document(doc)
    add_footer(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = title.add_run("Progress Report: ECG-QA Adapter Experiments")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor.from_string("0B2545")

    subtitle = doc.add_paragraph()
    subtitle.add_run(
        "Decoding Cardiac Languages with Advanced Large Language Models | Prepared for supervisor discussion"
    ).italic = True

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "The current pipeline now runs end-to-end on ARC: PTB-XL ECGs are encoded using a frozen pretrained CSFM-Tiny encoder, "
        "ECG-QA single-verify SCP-code questions are paired with those embeddings, and a lightweight adapter is trained to map one "
        "pooled ECG embedding into LLM soft tokens for yes/no answer generation."
    )
    doc.add_paragraph(
        "The best current adapter uses Llama-3.2-1B-Instruct as the frozen LLM and reaches 0.795 accuracy, 0.724 balanced accuracy, "
        "and 0.731 macro-F1 on the official ECG-QA validation split. More importantly, detailed SCP-code analysis shows that the "
        "model performs substantially better on broad/global ECG findings such as rhythm, conduction disturbance, and myocardial "
        "infarction/injury than on subtle ST/T, morphology, voltage, and lead-localized abnormalities."
    )

    doc.add_heading("Current Dataset And Embedding Setup", level=1)
    add_labeled_paragraph(
        doc,
        "Dataset",
        "ECG-QA PTB-XL subset restricted to single_verify yes/no questions with attribute_type = scp_code.",
    )
    add_labeled_paragraph(
        doc,
        "Split",
        "Official ECG-QA split preserved: 29,929 train rows, 4,789 validation rows, and 6,399 test rows.",
    )
    add_labeled_paragraph(
        doc,
        "Coverage",
        "41,117 total questions across 13,190 unique ECGs and all available SCP codes in the filtered subset.",
    )
    add_labeled_paragraph(
        doc,
        "CSFM features",
        "Pretrained CSFM-Tiny encoder, classification head replaced with identity, producing one 768-dimensional pooled CLS embedding per ECG.",
    )
    add_labeled_paragraph(
        doc,
        "Important caveat",
        "The current adapter uses one pooled representation per ECG, not a sequence of per-lead or per-patch latent tokens.",
    )

    doc.add_heading("Adapter Training Setup", level=1)
    add_bullet(doc, "Frozen ECG encoder: pretrained CSFM-Tiny.")
    add_bullet(doc, "Frozen LLM: initially SmolLM2-135M-Instruct, then Llama-3.2-1B-Instruct.")
    add_bullet(doc, "Trainable component: linear projection adapter mapping one 768-dimensional CSFM embedding into 8 soft tokens.")
    add_bullet(doc, "Input to LLM: ECG soft tokens prepended to a text prompt containing the ECG-QA question.")
    add_bullet(doc, "Training objective: generative yes/no answer training, evaluated using forced-choice yes/no scoring.")
    add_bullet(doc, "Compute: full embedding extraction and adapter training performed on ARC/HTC GPU resources.")

    doc.add_heading("Main Model Results", level=1)
    add_table(
        doc,
        ["Run", "Training data", "Validation data", "Accuracy", "Balanced accuracy", "Macro-F1"],
        [
            ["SmolLM2-135M", "Real ECG-QA train", "Real ECG-QA val", "0.759", "0.690", "0.692"],
            ["Llama-3.2-1B", "Real ECG-QA train", "Real ECG-QA val", "0.795", "0.724", "0.731"],
            ["SmolLM + synthetic v1", "Real + synthetic train", "Real ECG-QA val", "0.769", "0.671", "0.683"],
        ],
        [1.45, 1.35, 1.25, 0.8, 1.0, 0.8],
    )
    add_table_spacer(doc)
    doc.add_paragraph(
        "The Llama-3.2-1B run is the current best model. Synthetic v1 did not improve validation performance, suggesting that the first synthetic data construction was not sufficient to improve generalization."
    )

    doc.add_heading("Clinical Group Performance", level=1)
    doc.add_paragraph(
        "Performance varies strongly by clinical category. Confidence intervals were estimated using ECG-level clustered bootstrap resampling, so repeated questions from the same ECG are resampled together."
    )
    add_table(
        doc,
        ["Clinical group", "n", "Balanced accuracy", "95% CI", "Yes recall", "No recall"],
        [
            ["MI/injury", "159", "0.883", "[0.827, 0.927]", "0.949", "0.817"],
            ["Rhythm", "399", "0.850", "[0.807, 0.886]", "0.807", "0.893"],
            ["Conduction disturbance", "387", "0.814", "[0.767, 0.854]", "0.786", "0.841"],
            ["Normal", "60", "0.775", "[0.654, 0.884]", "0.700", "0.850"],
            ["Hypertrophy", "300", "0.723", "[0.667, 0.776]", "0.689", "0.757"],
            ["Form/morphology", "2411", "0.693", "[0.663, 0.720]", "0.495", "0.891"],
            ["ST/T change", "1073", "0.668", "[0.619, 0.722]", "0.430", "0.906"],
        ],
        [1.6, 0.5, 1.0, 1.05, 0.85, 0.85],
    )
    add_table_spacer(doc)

    doc.add_paragraph(
        "The central scientific observation is that broad/global categories are easier for this architecture, while subtle morphology and ST/T abnormalities are harder. The confidence intervals suggest this is not simply random validation-set noise."
    )

    doc.add_heading("Error Direction And Localization", level=1)
    doc.add_paragraph(
        "The weaker categories mostly fail through false negatives: the model under-calls true abnormalities rather than over-calling abnormalities. This is clinically important because the main failure mode is missed subtle disease signal."
    )
    add_table(
        doc,
        ["Question type", "n", "Balanced accuracy", "Yes recall", "No recall"],
        [
            ["Whole ECG", "1780", "0.781", "0.717", "0.845"],
            ["Localized any", "3009", "0.679", "0.458", "0.899"],
            ["Explicit lead", "2369", "0.668", "0.409", "0.926"],
            ["Regional leads", "640", "0.723", "0.645", "0.800"],
        ],
        [1.65, 0.6, 1.1, 0.9, 0.9],
    )
    add_table_spacer(doc)
    doc.add_paragraph(
        "SCP codes with a higher proportion of localized questions tend to perform worse. The correlation between localized-question fraction and per-code balanced accuracy was Pearson r = -0.535 and Spearman rho = -0.525. However, localization alone is not the full explanation: localized MI/injury codes can still perform well, while the weakest localized codes tend to be subtle ST/T, voltage, and waveform morphology findings."
    )

    doc.add_heading("Sanity Check Against The CSFM Paper", level=1)
    doc.add_paragraph(
        "The embedding extraction matches the CSFM feature-extraction setup described in the CSFM repository: pretrained encoder weights are loaded, the classification head is replaced with identity, and the model output is used as a feature vector. In our implementation this gives a frozen 768-dimensional pooled CSFM-Tiny representation for each 12-lead ECG."
    )
    doc.add_paragraph(
        "The CSFM paper does address lead-configuration transfer and includes lead-related ECG-QA experiments. Therefore, the claim should not be that lead-specific ECG-QA has not been studied. The distinct question in this project is whether a frozen pooled CSFM embedding, passed through a lightweight adapter into a frozen LLM, exposes enough fine-grained lead-specific waveform evidence for question answering. Current results suggest that this pooled bottleneck is useful but limited."
    )

    doc.add_heading("Current Interpretation", level=1)
    doc.add_paragraph(
        "The adapter is learning clinically meaningful ECG-QA behaviour above chance. The most defensible interpretation is that the frozen CSFM pooled embedding contains useful global ECG information, but the current single-vector adapter is less reliable for subtle, localized waveform abnormalities. This supports a move from simply chasing global balanced accuracy toward analysing what kinds of ECG information survive the frozen encoder + lightweight alignment bottleneck."
    )

    doc.add_heading("Proposed Next Discussion Points", level=1)
    add_bullet(doc, "Should the next adapter use per-lead or token-sequence CSFM representations rather than one pooled embedding?")
    add_bullet(doc, "Should the dissertation emphasize clinical category-level failure analysis as a core result?")
    add_bullet(doc, "Is the Llama-3.2-1B adapter result sufficient as the main proof-of-concept before testing a more complex adapter?")
    add_bullet(doc, "Should the held-out test set remain untouched until a final architecture choice is made?")
    add_bullet(doc, "How much additional effort should be spent on synthetic ECG-QA generation, given that v1 did not improve validation performance?")

    doc.add_heading("Reproducibility Artifacts", level=1)
    add_bullet(doc, "Embedding extraction: extract_ptbxl_csfm_embeddings.py")
    add_bullet(doc, "Dataset construction: build_ecgqa_dataset_from_embeddings.py")
    add_bullet(doc, "Adapter training: train_ecg_soft_prompt_adapter.py")
    add_bullet(doc, "Best Llama predictions: outputs/20260704_183337_..._predictions.jsonl")
    add_bullet(doc, "Clinical group/bootstrap/localization analyses: analyze_* scripts in ProjectCode")
    add_bullet(doc, "Raw prediction JSONL is the audit trail for recomputing metrics in Python or R.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


if __name__ == "__main__":
    build_report(OUT_PATH)
    print(OUT_PATH)
