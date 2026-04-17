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
- [x] Build initial dataset (100 samples)

---

## Project Structure

```
ProjectCode/
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralised paths and settings
│   ├── ecgqa_loader.py        # Loads ECG-QA dataset
│   ├── ptbxl_loader.py        # Loads PTB-XL ECG signals
│   ├── dataset_builder.py     # Builds embedding dataset
│
├── outputs/                   # Generated datasets
├── logs/                      # Logs (for cluster use later)
├── build_preview_dataset.py   # Script to generate dataset
├── test_one_sample.py         # Single-sample test script
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

## Output Format

Each sample is saved as a JSON object:

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

- Train baseline models:
  - ECG-only (embedding → answer)
  - Question-only
  - Combined model

- Evaluate performance:
  - Accuracy
  - Macro-F1
  - AUROC

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