# Model Technical Specification - 2026-07-21

## 1. Dataset And Task Specification

### Dataset

The adapter experiments use the ECG-QA subset constructed from PTB-XL ECGs and ECG-QA question-answer pairs.

Filtering criteria:

- Question type: `single_verify`.
- Attribute type: `scp_code`.
- Answer type: binary `yes`/`no`.
- Split: official ECG-QA train/validation/test split.
- ECG representation: precomputed CSFM embedding indexed by ECG identifier.

Dataset sizes:

| Split | Rows |
| --- | ---: |
| Train | 29,929 |
| Validation | 4,789 |
| Test | 6,399 |

The adapter model is trained on train rows, selected by ECG-QA's official training split, and evaluated on validation unless explicitly stated otherwise.

## 2. Core Architecture

The main architecture has four distinct components:

1. ECG encoder.
2. Trainable ECG-to-LLM adapter.
3. Frozen LLM.
4. Forced-choice yes/no evaluator.

### 2.1 ECG Encoder

Model:

- Cardiac Sensing Foundation Model (CSFM).
- Main experiments use CSFM-Tiny pooled embeddings.
- Additional experiment tested CSFM-Base embeddings.

Embedding format:

- One pooled ECG embedding per ECG.
- Main CSFM-Tiny embedding dimension: 768.
- Embeddings were precomputed once on ARC and stored as JSONL embedding banks.

Frozen status:

- CSFM is not trained during adapter experiments.
- The training script reads existing embeddings from disk; raw ECG waveforms are not passed through CSFM during adapter training.

Rationale:

- Precomputing embeddings avoids repeatedly running the ECG foundation model.
- It keeps the ECG representation fixed, making the experiment specifically about ECG-to-language alignment.

### 2.2 Adapter

Main adapter type:

- Linear projection adapter.

Input:

- One CSFM ECG embedding: shape `[768]`.

Output:

- A sequence of soft prompt tokens in the LLM hidden space.
- Main setting: 8 soft tokens.

For Llama-3.2-3B-Instruct:

- LLM hidden size: 3072.
- Adapter output shape per ECG: `[8, 3072]`.
- Adapter trainable parameters: 18,898,944.

Conceptually:

```text
CSFM embedding [768]
    -> linear adapter
ECG soft tokens [8, LLM hidden size]
    -> prepended to text prompt embeddings
    -> frozen LLM
```

The adapter is the main trainable component and the central architectural contribution. It learns how to translate a fixed ECG representation into continuous tokens that the frozen LLM can condition on.

### 2.3 MLP Adapter Variant

An MLP adapter was also tested.

Configuration:

- Input: 768-dimensional CSFM embedding.
- Hidden layer: 2048.
- Activation: GELU.
- Output: 8 soft tokens in LLM hidden space.

Result:

- The MLP adapter did not clearly outperform the simpler linear adapter in the available experiments.
- The linear adapter remained the preferred main configuration because it is simpler and easier to interpret.

Source result:
`outputs/20260626_081745_20_epochs_batch_16_mlp_adapter_hidden_2048_8_soft_tokens_full_all_scp_codes_adapter_full_all_codes_results.json`

### 2.4 Frozen LLM

LLMs tested:

- SmolLM2-135M-Instruct.
- Llama-3.2-1B-Instruct.
- Llama-3.2-3B-Instruct.

Current best main model:

- Llama-3.2-3B-Instruct.

Frozen status:

- In the main adapter-only experiments, all LLM parameters are frozen.
- Only the adapter parameters receive gradient updates.

Why frozen:

- Keeps training computationally feasible on ARC/HTC.
- Tests whether pretrained ECG representations can be aligned to language without end-to-end multimodal pretraining.
- Follows the BLIP-2 style paradigm: frozen encoder, lightweight trainable bridge, frozen language model.
- Reduces the risk that improvements come from language-model fine-tuning rather than ECG-to-language alignment.

## 3. Forward Pass

For each ECG-QA example:

1. Retrieve the precomputed CSFM embedding for the ECG identifier.
2. Insert the ECG-QA question into a fixed instruction prompt.
3. Tokenize the prompt using the LLM tokenizer.
4. Pass prompt token IDs through the frozen LLM embedding layer to obtain text-token embeddings.
5. Pass the CSFM ECG embedding through the trainable adapter to obtain ECG soft tokens.
6. Concatenate ECG soft tokens before the text-token embeddings.
7. Feed the combined embedding sequence into the LLM using `inputs_embeds`.
8. Train/evaluate the model to answer `yes` or `no`.

Important implementation detail:

- The ECG soft tokens are continuous vectors, not discrete vocabulary tokens.
- They are injected directly into the LLM embedding stream.
- No new tokenizer vocabulary is added.
- The LLM architecture itself is not structurally changed.

## 4. Training Objective

Training is generative but constrained to binary answers.

The model is prompted with the ECG soft tokens plus the question prompt, followed by the target answer token sequence.

Loss:

- Cross-entropy language-modeling loss.
- Loss is applied only to the answer-token positions.
- Prompt positions are masked.
- ECG soft-token positions are masked.

This means the adapter is optimized to make the frozen LLM assign higher likelihood to the correct answer, either `yes` or `no`.

Gradient flow:

```text
Loss on answer tokens
    -> frozen LLM backward pass
    -> gradients through LLM computation graph
    -> update adapter parameters only
```

In adapter-only experiments:

- CSFM: frozen.
- LLM: frozen.
- Adapter: trainable.

The LLM still participates in backpropagation as a differentiable computation graph, but its own weights are not updated.

## 5. Evaluation Strategy

Evaluation uses forced-choice answer scoring.

For each example:

1. Score the candidate answer `yes`.
2. Score the candidate answer `no`.
3. Select the answer with the better candidate likelihood.

Default decision:

- Predict whichever candidate has higher log-likelihood.

Threshold-tuned decision:

- Compute a yes-vs-no score difference.
- Tune a threshold on validation to improve balanced accuracy.
- Apply the selected threshold to validation or held-out test.

Metrics:

- Accuracy.
- Balanced accuracy.
- Macro-F1.
- Yes recall.
- No recall.
- Confusion matrix.

Balanced accuracy is treated as the main metric because the dataset is imbalanced toward `no` answers. Raw accuracy can look high even when the model misses many positive cases.

## 6. Main Hyperparameters

### Best Adapter-Only Llama-3.2-3B Run

Source:
`outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_results.json`

Configuration:

| Parameter | Value |
| --- | --- |
| LLM | Llama-3.2-3B-Instruct |
| ECG encoder | CSFM-Tiny pooled embedding |
| CSFM dimension | 768 |
| Adapter type | Linear |
| Soft tokens | 8 |
| LLM hidden size | 3072 |
| Adapter dtype | float32 |
| LLM dtype | float16 |
| Epochs | 20 |
| Best epoch | 11 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Learning rate | 5e-6 |
| Gradient clipping | 0.25 |
| Trainable parameters | 18,898,944 |
| Frozen components | CSFM and LLM |
| Evaluation mode | Forced-choice yes/no |

Best validation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7895 |
| Balanced accuracy | 0.7380 |
| Macro-F1 | 0.7357 |
| Yes recall | 0.6260 |
| No recall | 0.8501 |

## 7. Model Variants Tested

### 7.1 Llama-3.2-1B-Instruct

Configuration:

- LLM: Llama-3.2-1B-Instruct.
- Adapter: linear.
- Soft tokens: 8.
- Learning rate: 1e-5.
- Batch size: 2.
- Gradient clipping: 0.5.
- Epochs: 20.

Best validation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7954 |
| Balanced accuracy | 0.7235 |
| Macro-F1 | 0.7311 |
| Yes recall | 0.5672 |
| No recall | 0.8798 |

Interpretation:

- Strong result for a smaller model.
- Lower yes recall than the 3B model.
- Suggests some scaling benefit from moving to 3B.

### 7.2 Llama-3.2-3B-Instruct

Configuration:

- LLM: Llama-3.2-3B-Instruct.
- Adapter: linear.
- Soft tokens: 8.
- Learning rate: 5e-6.
- Batch size: 1.
- Gradient accumulation: 4.
- Epochs: 20.

Best validation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7895 |
| Balanced accuracy | 0.7380 |
| Macro-F1 | 0.7357 |
| Yes recall | 0.6260 |
| No recall | 0.8501 |

Interpretation:

- Current best adapter-only validation run.
- Better positive recall than 1B.
- Chosen as the main model for mechanistic ablations.

### 7.3 CSFM-Base Embedding Run

Configuration:

- ECG encoder: CSFM-Base.
- LLM: Llama-3.2-3B-Instruct.
- Adapter: linear.
- Soft tokens: 8.
- Learning rate: 5e-6.

Best validation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7956 |
| Balanced accuracy | 0.7283 |
| Macro-F1 | 0.7340 |
| Yes recall | 0.5819 |
| No recall | 0.8747 |

Interpretation:

- CSFM-Base did not improve balanced accuracy over CSFM-Tiny.
- The model became more conservative, with higher no recall and lower yes recall.

### 7.4 MLP Adapter

Configuration:

- ECG encoder: CSFM-Tiny.
- LLM: smaller local adapter-stage model in earlier experiments.
- Adapter: Linear -> GELU -> Linear.
- Hidden size: 2048.
- Soft tokens: 8.
- Epochs: 20.

Interpretation:

- More expressive than the linear adapter.
- Did not become the main configuration because it did not clearly outperform the simpler linear adapter.

## 8. LoRA Experiment Specification

Source:
`outputs/20260715_153929_20_epochs_llama_3_2_3b_csfm_tiny_linear_adapter_plus_lora_r8_alpha16_float16_llm_adapter_full_all_codes_results.json`

Purpose:

- Test whether limited LLM adaptation helps the LLM interpret ECG soft tokens better than adapter-only training.

Configuration:

| Parameter | Value |
| --- | --- |
| ECG encoder | CSFM-Tiny pooled embedding |
| LLM | Llama-3.2-3B-Instruct |
| Adapter | Linear |
| Soft tokens | 8 |
| LLM dtype | float16 |
| Adapter dtype | float32 |
| LoRA enabled | Yes |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Adapter parameters | 18,898,944 |
| LoRA parameters | 4,587,520 |
| Total trainable parameters | 23,486,464 |
| Learning rate | 5e-6 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Epochs requested | 20 |
| Best completed epoch | 13 |

How LoRA works here:

- The original LLM weights remain frozen.
- Low-rank trainable matrices are added to selected attention projection layers.
- During training, the adapter and LoRA matrices are updated.
- The goal is to let the LLM slightly adapt its attention computations to better use the ECG soft tokens, without full LLM fine-tuning.

Result:

| Metric | Adapter Only | Adapter + LoRA |
| --- | ---: | ---: |
| Accuracy | 0.7895 | 0.8033 |
| Balanced accuracy | 0.7380 | 0.7297 |
| Macro-F1 | 0.7357 | 0.7393 |
| Yes recall | 0.6260 | 0.5696 |
| No recall | 0.8501 | 0.8898 |

Interpretation:

- LoRA increased no recall and raw accuracy.
- LoRA reduced yes recall and balanced accuracy.
- It did not solve the positive-case sensitivity problem.
- Further LoRA work should investigate whether the model is overfitting to answer priors or becoming more conservative.

## 9. Class-Weighted Loss Specification

Source:
`outputs/20260720_191645_20_epochs_llama_3_2_3b_class_weighted_balanced_loss_float16_llm_float32_adapter_adapter_full_all_codes_results.json`

Purpose:

- Increase the loss contribution for the minority `yes` class.
- Test whether poor positive recall is mainly caused by answer imbalance.

Configuration:

| Parameter | Value |
| --- | --- |
| Model | Llama-3.2-3B + CSFM-Tiny + linear adapter |
| Class weight mode | `balanced` |
| No weight | 0.6959 |
| Yes weight | 1.7764 |
| Learning rate | 5e-6 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Gradient clipping | 0.25 |
| Epochs | 20 |

Loss propagation:

- The weighted loss is still applied only to answer-token positions.
- Prompt and ECG soft-token positions remain masked.
- The correct answer token receives the class-specific weight.
- Only the adapter parameters are updated.

Result:

| Metric | Unweighted Adapter | Class-Weighted Adapter |
| --- | ---: | ---: |
| Accuracy | 0.7895 | 0.7883 |
| Balanced accuracy | 0.7380 | 0.7352 |
| Macro-F1 | 0.7357 | 0.7335 |
| Yes recall | 0.6260 | 0.6198 |
| No recall | 0.8501 | 0.8506 |

Interpretation:

- Class weighting did not improve the main validation result.
- The yes-recall limitation is unlikely to be solved by simple answer-level reweighting alone.

## 10. Threshold Tuning Specification

Default forced-choice scoring chooses the answer with the higher candidate likelihood.

Threshold tuning instead uses a score difference:

```text
score = log P(yes) - log P(no)
```

The prediction is:

```text
yes if score >= threshold
no otherwise
```

Thresholds tested:

- Default implicit threshold.
- One global validation-tuned threshold.
- One per-SCP-code validation-tuned threshold.

Validation results:

| Method | Balanced Accuracy | Yes Recall | No Recall |
| --- | ---: | ---: | ---: |
| Default | 0.738 | 0.626 | 0.850 |
| Global threshold | 0.762 | 0.798 | 0.725 |
| Per-code threshold | 0.782 | 0.821 | 0.744 |

Held-out test results:

| Method | Test Balanced Accuracy | Test Yes Recall | Test No Recall |
| --- | ---: | ---: | ---: |
| Default | 0.724 | 0.614 | 0.835 |
| Global threshold | 0.745 | 0.782 | 0.708 |
| Per-code threshold | 0.733 | 0.769 | 0.698 |

Interpretation:

- Global threshold tuning is useful and generalizes better than per-code threshold tuning.
- Per-code tuning risks overfitting validation codes/sample sizes.
- The threshold analysis suggests the model has useful ranking signal that is not fully exploited by default forced-choice decoding.

## 11. Diagnostic Experiment Specifications

### 11.1 ECG Token Ablation

Script:
`evaluate_ecg_token_ablation.py`

Purpose:

- Test whether the ECG soft tokens are causally important.

Conditions:

- `real`: actual ECG embedding passed through adapter.
- `zero`: ECG soft tokens replaced by zeros.
- `mean`: all ECGs receive the average ECG token.
- `shuffled`: ECG tokens randomly assigned to other examples.
- `random`: random soft-token vectors.

Main result:

| Condition | Balanced Accuracy |
| --- | ---: |
| Real | 0.738 |
| Zero | 0.500 |
| Mean | 0.524 |
| Shuffled | 0.508 |
| Random | 0.498 |

Conclusion:

- The trained model depends on the correct ECG token.
- The result rules out a trivial text-only or prior-only explanation.

### 11.2 Adapter Token Probe

Script:
`probe_adapter_token_information.py`

Purpose:

- Test whether SCP-code information remains linearly recoverable after the adapter.

Feature sets:

- Original CSFM embedding.
- Adapter soft tokens averaged across the 8 token positions.
- Adapter soft tokens flattened into one vector.

Main result:

| Feature Set | Mean AUC | Mean Balanced Accuracy |
| --- | ---: | ---: |
| CSFM | 0.851 | 0.784 |
| Adapter mean | 0.840 | 0.780 |
| Adapter flat | 0.835 | 0.776 |

Conclusion:

- The adapter retains most of the original CSFM SCP-code information.
- The remaining performance bottleneck likely lies in LLM interpretation, calibration, or question-conditioned reasoning rather than complete loss of ECG signal.

### 11.3 ECG/Question Swap

Script:
`evaluate_ecg_question_swap.py`

Purpose:

- Test whether the model responds appropriately when the ECG changes while the question is fixed, or when the question changes while the ECG is fixed.

Main result:

| Test Type | Balanced Accuracy |
| --- | ---: |
| Same ECG, different question | 0.833 |
| Same question, different ECG | 0.813 |

Conclusion:

- The model uses both ECG information and question information.
- It is not only predicting from ECG abnormality or question priors.

## 12. Important Implementation Files

Core training:

- `train_ecg_soft_prompt_adapter.py`
  - Main adapter training script.
  - Loads embedding bank.
  - Loads frozen LLM.
  - Builds adapter.
  - Applies answer-token-only loss.
  - Supports LoRA, class weighting, gradient accumulation, timestamped run outputs.

Dataset construction:

- `build_ecgqa_dataset_from_embeddings.py`
  - Builds ECG-QA SCP-code dataset from ECG-QA rows and precomputed embedding availability.

Embedding extraction:

- `extract_ptbxl_csfm_embeddings.py`
  - Extracts CSFM embeddings for PTB-XL ECGs on ARC.

Evaluation and diagnostics:

- `tune_yes_no_thresholds.py`
  - Tunes global and per-code yes/no thresholds.

- `evaluate_adapter_on_test_with_thresholds.py`
  - Evaluates default and validation-tuned thresholds on held-out test data.

- `evaluate_ecg_token_ablation.py`
  - Runs ECG-token corruption/ablation experiment.

- `probe_adapter_token_information.py`
  - Tests how much SCP-code information is retained in adapter outputs.

- `evaluate_ecg_question_swap.py`
  - Tests whether predictions respond to ECG and question changes.

- `analyze_class_weighted_run.py`
  - Compares class-weighted training against the unweighted baseline.

- `tools/create_mechanistic_diagnostic_artifacts.py`
  - Creates summary tables and plots for mechanistic diagnostics.

## 13. Current Technical Conclusion

The strongest current technical configuration is:

```text
CSFM-Tiny pooled embedding
-> linear adapter
-> 8 soft prompt tokens
-> frozen Llama-3.2-3B-Instruct
-> forced-choice yes/no scoring
```

This setup gives the best adapter-only validation balanced accuracy so far: 0.738.

The main technical bottleneck is not simply that the adapter lacks ECG signal. Ablations show that ECG tokens matter, and linear probes show that adapter tokens preserve SCP-code information. The harder problem is converting that preserved ECG signal into reliable, calibrated, question-conditioned LLM decisions, especially for subtle or lead-localized ECG abnormalities.

