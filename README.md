# ECG–LLM: Integrating Biosignal Foundation Models with Large Language Models

## Overview

This project explores how to integrate biosignal foundation models with large language models (LLMs).

Specifically, it investigates whether a **frozen LLM** can reason over ECG signals using embeddings extracted from a pretrained cardiac foundation model (CSFM), without retraining either model.

This work sits at the intersection of:
- Multimodal machine learning
- Foundation models
- Biomedical signal processing

---

## Project Goal

The primary objective is:

> To design and train a lightweight alignment interface that enables a frozen LLM to interpret ECG signals via learned embeddings.

Instead of converting ECG signals into text, this project focuses on enabling **direct reasoning over signal representations**.

---

## Pipeline

The current system pipeline is:

```
ECG-QA (question, answer, ecg_id)
        ↓
PTB-XL (ecg_id → ECG waveform)
        ↓
Preprocessing (resampling, cleaning, normalization)
        ↓
CSFM (ECG → embedding)
        ↓
Dataset construction (embedding + question → answer)
        ↓
Model training (baseline → adapter → LLM)
```

---

## Current Progress

- [x] Load ECG-QA dataset
- [x] Load PTB-XL ECG signals
- [x] Map ECG-QA to PTB-XL using ecg_id
- [x] Preprocess ECG signals using CSFM preprocessing
- [x] Extract embeddings using CSFM
- [x] Build CSFM embedding datasets for ECG-QA single-verify questions
- [x] Probe SCP-code recoverability from frozen CSFM embeddings
- [x] Build a clean binary ECG-QA subset for baseline experiments

---

## Project Structure

```
ProjectCode/
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralised paths and settings
│   ├── ecgqa_loader.py        # Loads ECG-QA dataset
│   ├── ptbxl_loader.py        # Loads PTB-XL ECG signals
│   ├── ecg_feature_extractor.py
│   ├── ecg_prompt_builder.py
│
├── outputs/                   # Generated datasets
├── logs/                      # Logs (for cluster use later)
├── build_preview_dataset.py   # Extracts CSFM embeddings for ECG-QA rows
├── build_scp_ecg_dataset.py   # Builds ECG-level SCP-labelled embedding data
├── build_ecgqa_scp_subset.py  # Builds the current binary ECG-QA subset
├── evaluate_scp_code_panel.py # Ranks SCP codes by CSFM probe performance
├── train_embedding_interpreter.py
├── train_baseline.py
├── test_ecg_llm.py
├── README.md
```

---

## Data

This project uses two datasets:

### ECG-QA
- Provides question-answer pairs linked to ECG signals
- Format: `(question, answer, ecg_id)`
- Used as supervision for training

### PTB-XL
- Large ECG dataset containing raw waveforms
- Provides signal data for each `ecg_id`
- Used as input to CSFM

---

## Environment Setup

Activate your conda environment:

```bash
conda activate csfm
```

Install required dependencies:

```bash
pip install numpy pandas wfdb torch neurokit2
```

---

## Running the Pipeline

To generate a small dataset (e.g. 100 samples):

```bash
python build_preview_dataset.py
```

This script will:
- load ECG-QA data
- map each `ecg_id` to PTB-XL
- preprocess ECG signals
- extract embeddings using CSFM
- save results to the `outputs/` folder

---

## Current Experimental Subset

The current baseline experiments use a controlled binary subset of ECG-QA.

Filtering criteria:

- `question_type == "single-verify"`
- `attribute_type == "scp_code"`
- answer is binary: `yes` or `no`
- the question targets one of the selected SCP codes
- every row has a CSFM ECG embedding

Selected SCP codes:

| Code | Description |
| --- | --- |
| AFIB | atrial fibrillation |
| LAFB | left anterior fascicular block |
| LVH | left ventricular hypertrophy |
| NORM | normal ECG |
| CLBBB | complete left bundle branch block |
| CRBBB | complete right bundle branch block |
| ASMI | anteroseptal myocardial infarction |

These codes were selected after probing which PTB-XL SCP labels are recoverable
from frozen CSFM embeddings with a simple logistic-regression classifier. The
panel results are saved in:

```bash
outputs/scp_code_panel_results.json
outputs/scp_code_panel_results.csv
```

Current subset outputs:

```bash
outputs/ecgqa_scp_binary_subset.jsonl
outputs/ecgqa_scp_binary_train.jsonl
outputs/ecgqa_scp_binary_val.jsonl
outputs/ecgqa_scp_binary_subset_stats.json
```

Current subset statistics:

| Statistic | Count |
| --- | ---: |
| Total questions | 938 |
| Unique ECGs | 885 |
| Train questions | 745 |
| Validation questions | 193 |
| Train unique ECGs | 708 |
| Validation unique ECGs | 177 |
| Train/validation ECG overlap | 0 |
| Yes answers | 310 |
| No answers | 628 |

Questions per selected code:

| Code | Questions |
| --- | ---: |
| LVH | 378 |
| CLBBB | 102 |
| NORM | 97 |
| CRBBB | 94 |
| AFIB | 91 |
| LAFB | 90 |
| ASMI | 86 |

The train/validation split is assigned by `ecg_id`, not by row. This prevents
the same ECG embedding from appearing in both train and validation sets.

To regenerate the subset after creating the CSFM embedding file:

```bash
python build_ecgqa_scp_subset.py
```

---

## Output Format

Each CSFM embedding sample is saved as a JSON object:

```json
{
  "ecg_id": 4803,
  "question": "Does this ECG show symptoms of non-diagnostic t abnormalities?",
  "answer": ["yes"],
  "question_type": "single-verify",
  "attribute_type": "scp_code",
  "attribute": ["non-diagnostic t abnormalities"],
  "embedding": [0.12, -0.43, ..., 768 values],
  "embedding_dim": 768,
  "signal_shape": [12, 2500]
}
```

The current binary ECG-QA subset adds SCP-code metadata and a fixed split:

```json
{
  "ecg_id": 837,
  "question": "Does this ECG show symptoms of left ventricular hypertrophy?",
  "answer": "no",
  "label": 0,
  "question_type": "single-verify",
  "attribute_type": "scp_code",
  "attribute": "left ventricular hypertrophy",
  "target_scp_code": "LVH",
  "target_scp_statement": {
    "code": "LVH",
    "description": "left ventricular hypertrophy"
  },
  "ptbxl_scp_codes": {
    "NORM": 100.0,
    "SR": 0.0
  },
  "embedding": [0.12, -0.43, "... 768 values"],
  "embedding_dim": 768,
  "signal_shape": [12, 2500],
  "split": "train"
}
```

---

## Key Insight

The project converts:

```
Raw ECG signal → compact embedding → language-compatible input
```

This enables:
- efficient learning
- scalable datasets
- integration with LLMs

---

## Next Steps

- Implement embedding-based baselines on the binary ECG-QA subset:
  - ECG-only: CSFM embedding → yes/no
  - Text-only: question embedding → yes/no
  - Combined: CSFM embedding + question embedding → yes/no

- Refine and evaluate a simple LLM baseline:
  - raw ECG → hand-crafted ECG features/text description
  - ECG description + question → frozen LLM → yes/no

- Develop adapter module:
  - Convert embeddings → LLM tokens

- Integrate with LLM:
  - Enable ECG-based question answering

- Add uncertainty estimation:
  - Confidence thresholds
  - "I don’t know" responses

---

## Research Context

This work is inspired by:

- BLIP-2 — frozen encoder + LLM alignment
- SensorLM / SleepLM — signal-to-language models

Key distinction:

> This project focuses on enabling LLM reasoning over biosignal embeddings without converting signals to text.

---

## Author

Tariro Nathan Banganayi  
Oxford MSc Dissertation
