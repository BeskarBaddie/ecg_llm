# Supervisor Research Update - 2026-07-21

## 1. Current Research Position

The project has now moved beyond the initial baseline stage and into the main ECG-to-LLM adapter investigation. The current working system uses precomputed CSFM ECG embeddings as physiological representations of PTB-XL ECGs, maps each ECG embedding into continuous LLM soft prompt tokens using a trainable adapter, and conditions a frozen instruction-tuned LLM to answer ECG-QA binary `yes`/`no` questions.

The current main task remains deliberately constrained:

- ECG-QA `single_verify` questions only.
- `attribute_type = scp_code`.
- Binary `yes`/`no` answers.
- Official ECG-QA train/validation/test split preserved.
- CSFM and the LLM frozen for the main adapter-only experiments.
- Test set held out except for explicit final/diagnostic threshold evaluation.

The full ECG-QA SCP-code subset currently contains:

| Split | Questions |
| --- | ---: |
| Train | 29,929 |
| Validation | 4,789 |
| Test | 6,399 |

The full subset covers 41,117 questions across 13,190 unique ECGs.

## 2. Best Current Adapter Result

The best current validation result comes from the CSFM-Tiny pooled embedding plus linear soft-prompt adapter plus frozen Llama-3.2-3B-Instruct model:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSFM-Tiny + Linear Adapter + Llama-3.2-3B | 0.790 | 0.738 | 0.736 | 0.626 | 0.850 |

Source file:
`outputs/20260710_011154_20_epochs_llama_3_2_3b_instruct_float16_llm_float32_adapter_lr5e_6_batch1_accum4_adapter_full_all_codes_results.json`

This is the current strongest adapter-only proof of concept. It performs substantially above chance on balanced accuracy, but it is not yet clinically safe. The key limitation is still lower positive recall: the model is better at identifying when an attribute is absent than when it is present.

## 3. Experiments Run Since The Previous Update

### 3.1 LLM Scaling

We tested whether increasing LLM capacity improves the frozen-adapter setup.

| Configuration | Best Epoch | Accuracy | Balanced Accuracy | Macro-F1 | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B + linear adapter | 8 | 0.795 | 0.724 | 0.731 | 0.567 | 0.880 |
| Llama-3.2-3B + linear adapter | 11 | 0.790 | 0.738 | 0.736 | 0.626 | 0.850 |

Interpretation:

- The 3B model improved balanced accuracy over 1B.
- The 3B model also improved positive recall.
- This suggests there is some useful scaling with LLM size, though the gain is modest rather than transformative.
- The improvement appears to come mainly from a better yes/no trade-off rather than a large raw accuracy gain.

### 3.2 CSFM-Base Embeddings

We extracted CSFM-Base embeddings and ran the current best adapter setup with the larger CSFM representation.

| Configuration | Accuracy | Balanced Accuracy | Macro-F1 | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSFM-Tiny + Llama-3.2-3B | 0.790 | 0.738 | 0.736 | 0.626 | 0.850 |
| CSFM-Base + Llama-3.2-3B | 0.796 | 0.728 | 0.734 | 0.582 | 0.875 |

Interpretation:

- CSFM-Base did not improve balanced accuracy in the current setup.
- It increased no recall but reduced yes recall.
- This suggests that simply increasing the ECG encoder size is not enough; the alignment and decision threshold remain important bottlenecks.

### 3.3 LoRA Experiment

We tested whether allowing limited LLM adaptation helps by training the ECG adapter plus LoRA layers in the frozen Llama-3.2-3B model.

| Configuration | Accuracy | Balanced Accuracy | Macro-F1 | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapter only | 0.790 | 0.738 | 0.736 | 0.626 | 0.850 |
| Adapter + LoRA | 0.803 | 0.730 | 0.739 | 0.570 | 0.890 |

Interpretation:

- LoRA improved raw accuracy and no recall.
- LoRA did not improve balanced accuracy.
- The model became more conservative about predicting `yes`.
- This is important because raw accuracy is misleading in this dataset: `no` answers are more common.
- The result suggests that LoRA may be learning the majority-class structure or becoming better calibrated for absence, but not necessarily improving ECG-positive grounding.

Source file:
`outputs/20260715_153929_20_epochs_llama_3_2_3b_csfm_tiny_linear_adapter_plus_lora_r8_alpha16_float16_llm_adapter_full_all_codes_results.json`

### 3.4 Class-Weighted Loss

We tested whether increasing the loss weight for the minority `yes` class improves positive recall.

| Configuration | Accuracy | Balanced Accuracy | Macro-F1 | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapter only | 0.790 | 0.738 | 0.736 | 0.626 | 0.850 |
| Class-weighted adapter | 0.788 | 0.735 | 0.733 | 0.620 | 0.851 |

Interpretation:

- Class weighting did not improve validation performance.
- It produced almost the same yes/no trade-off as the unweighted model.
- The current bottleneck is probably not just class imbalance in the loss.

Source file:
`outputs/20260720_191645_20_epochs_llama_3_2_3b_class_weighted_balanced_loss_float16_llm_float32_adapter_adapter_full_all_codes_results.json`

### 3.5 Threshold Tuning

The model outputs log-likelihood scores for candidate answers `yes` and `no`. By default, the selected answer is whichever candidate has higher likelihood. We tested whether tuning the yes/no decision threshold on validation improves balanced accuracy.

| Evaluation | Balanced Accuracy | Yes Recall | No Recall |
| --- | ---: | ---: | ---: |
| Default validation decision | 0.738 | 0.626 | 0.850 |
| Global validation-tuned threshold | 0.762 | 0.798 | 0.725 |
| Per-code validation-tuned threshold | 0.782 | 0.821 | 0.744 |

Held-out test evaluation:

| Evaluation | Test Balanced Accuracy | Test Yes Recall | Test No Recall |
| --- | ---: | ---: | ---: |
| Default test decision | 0.724 | 0.614 | 0.835 |
| Global validation-tuned threshold | 0.745 | 0.782 | 0.708 |
| Per-code validation-tuned threshold | 0.733 | 0.769 | 0.698 |

Interpretation:

- A global validation-tuned threshold generalizes reasonably to test.
- Per-code thresholds look best on validation but lose some advantage on test, which suggests overfitting.
- This is useful evidence that the underlying model scores contain more signal than default argmax decoding reveals.
- For final reporting, default forced-choice and validation-tuned threshold results should be reported separately.

Sources:

- `outputs/threshold_tuning/threshold_tuning_results.json`
- `outputs/adapter_3b_test_threshold_evaluation_results.json`

## 4. Ablation And Mechanistic Experiments

### 4.1 ECG Token Ablation

We tested whether the model is genuinely using ECG information by replacing the ECG soft tokens with corrupted alternatives.

| Condition | Accuracy | Balanced Accuracy | Yes Recall | No Recall | Pred Yes | Pred No |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real ECG tokens | 0.789 | 0.738 | 0.625 | 0.850 | 1333 | 3456 |
| Zero ECG tokens | 0.270 | 0.500 | 1.000 | 0.000 | 4789 | 0 |
| Mean ECG tokens | 0.658 | 0.524 | 0.230 | 0.817 | 938 | 3851 |
| Shuffled ECG tokens | 0.653 | 0.508 | 0.192 | 0.823 | 868 | 3921 |
| Random ECG tokens | 0.279 | 0.498 | 0.974 | 0.022 | 4680 | 109 |

Interpretation:

- Real ECG tokens are necessary for good performance.
- Shuffling ECG tokens destroys most of the balanced accuracy.
- Mean tokens retain some majority-distribution behaviour but lose meaningful ECG specificity.
- This is strong evidence that the adapter is not only exploiting question text or answer priors.

Sources:

- `outputs/ecg_token_ablation_3b_adapter_results.json`
- `outputs/mechanistic_diagnostics/table_token_ablation.md`

### 4.2 Adapter Token Probe

We trained simple classifiers on three feature sets to test whether SCP-code information is retained after the adapter:

- Original CSFM embeddings.
- Mean-pooled adapter soft tokens.
- Flattened adapter soft-token sequence.

| Feature Set | Codes | Mean AUC | Median AUC | Mean Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| CSFM embedding | 38 | 0.851 | 0.873 | 0.784 |
| Adapter token mean | 38 | 0.840 | 0.865 | 0.780 |
| Adapter token flat | 38 | 0.835 | 0.859 | 0.776 |

Interpretation:

- The adapter mostly preserves the clinically relevant SCP-code signal from CSFM.
- The performance drop from CSFM to adapter tokens is small.
- This suggests the main bottleneck is not simply that the adapter destroys the ECG representation.
- The harder problem is likely the use of those continuous tokens inside the frozen LLM for question-conditioned yes/no reasoning.

Source:
`outputs/adapter_token_probe_3b_results.json`

### 4.3 ECG/Question Swap Experiment

We tested whether the model responds to both ECG changes and question changes.

| Test Type | n | Accuracy | Balanced Accuracy | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same ECG, different question | 282 | 0.837 | 0.833 | 0.779 | 0.887 |
| Same question, different ECG | 262 | 0.813 | 0.813 | 0.725 | 0.901 |

Interpretation:

- The model changes behaviour when the ECG changes while the question is fixed.
- The model also changes behaviour when the question changes while the ECG is fixed.
- This supports the claim that the model is doing some question-conditioned ECG grounding, not simply memorizing question priors or ECG abnormality priors.

Source:
`outputs/ecg_question_swap_3b_adapter_results.json`

## 5. SCP-Code And Clinical Category Findings

### 5.1 Best-Performing SCP Codes

Among codes with at least 30 validation examples, the strongest codes are mostly rhythm, conduction, pacemaker, and some localized ischemic/injury labels.

| SCP Code | Description | Clinical Group | n | Balanced Accuracy | Yes Recall | No Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CRBBB | Complete right bundle branch block | Conduction disturbance | 60 | 1.000 | 1.000 | 1.000 |
| PACE | Normal functioning artificial pacemaker | Rhythm | 51 | 1.000 | 1.000 | 1.000 |
| CLBBB | Complete left bundle branch block | Conduction disturbance | 51 | 0.963 | 1.000 | 0.925 |
| INJAS | Subendocardial injury in anteroseptal leads | Myocardial infarction/injury | 42 | 0.943 | 1.000 | 0.886 |
| AFIB | Atrial fibrillation | Rhythm | 60 | 0.938 | 0.950 | 0.925 |
| PVC | Ventricular premature complex | Form/morphology | 60 | 0.925 | 0.900 | 0.950 |
| STACH | Sinus tachycardia | Rhythm | 60 | 0.925 | 1.000 | 0.850 |
| AFLT | Atrial flutter | Rhythm | 36 | 0.900 | 0.833 | 0.967 |

Interpretation:

- The adapter performs best on conditions with strong, global, or rhythm-level ECG signatures.
- Bundle branch blocks and rhythm abnormalities are relatively easy because they affect recognizable temporal or conduction patterns.
- This supports the clinical interpretation that some ECG-QA categories are much more compatible with pooled ECG embeddings than others.

### 5.2 Worst-Performing SCP Codes

| SCP Code | Description | Clinical Group | n | Balanced Accuracy | Yes Recall | No Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| STE_ | Non-specific ST elevation | Form/morphology | 30 | 0.460 | 0.000 | 0.920 |
| TAB_ | T-wave abnormality | Form/morphology | 66 | 0.491 | 0.000 | 0.982 |
| HVOLT | High QRS voltage | Form/morphology | 60 | 0.500 | 0.100 | 0.900 |
| ISCIN | Ischemic in inferior leads | ST/T change | 30 | 0.540 | 0.200 | 0.880 |
| ISCLA | Ischemic in lateral leads | ST/T change | 48 | 0.562 | 0.250 | 0.875 |
| LVOLT | Low QRS voltages | Form/morphology | 107 | 0.572 | 0.444 | 0.700 |
| NST_ | Non-specific ST changes | ST/T change | 300 | 0.575 | 0.217 | 0.933 |

Interpretation:

- Poor-performing categories are dominated by form/morphology and ST/T abnormalities.
- These often depend on subtle amplitude, morphology, and lead-localized changes.
- A single pooled CSFM vector may compress or average away clinically relevant local information.
- The main failure pattern is low positive recall: the model often says `no` for subtle positive cases.

### 5.3 Clinical Group Performance

| Clinical Group | n | Balanced Accuracy | Yes Recall | No Recall |
| --- | ---: | ---: | ---: | ---: |
| Myocardial infarction/injury | 159 | 0.873 | 0.821 | 0.925 |
| Conduction disturbance | 387 | 0.833 | 0.795 | 0.870 |
| Rhythm | 399 | 0.831 | 0.815 | 0.846 |
| Form/morphology | 2411 | 0.715 | 0.604 | 0.826 |
| ST/T change | 1073 | 0.708 | 0.527 | 0.888 |
| Hypertrophy | 300 | 0.702 | 0.567 | 0.838 |
| Normal | 60 | 0.638 | 0.400 | 0.875 |

Interpretation:

- The best-performing categories are clinically coherent: rhythm and conduction disturbances are often visually/global ECG phenomena.
- The weakest categories are clinically coherent too: ST/T changes, voltage changes, and morphology findings can be subtle and lead-specific.
- The analysis suggests that model performance is not uniformly poor; it varies systematically by clinical signal type.
- This is likely an important dissertation result, not only an engineering limitation.

### 5.4 Confidence And Lead-Localization Evidence

ECG-level bootstrap confidence intervals were calculated by resampling ECG identifiers rather than individual questions. This avoids treating multiple questions from the same ECG as fully independent samples.

For the best 1B run's clinical group bootstrap analysis, the broad pattern was:

- Rhythm: balanced accuracy 0.850, 95% CI approximately 0.807-0.886.
- Conduction disturbance: 0.814, CI approximately 0.767-0.854.
- Form/morphology: 0.693, CI approximately 0.663-0.720.
- ST/T change: 0.668, CI approximately 0.619-0.722.

This supports the claim that the category-level pattern is unlikely to be only random validation noise.

Lead-localization burden analysis found:

- Pearson correlation between localized-question fraction and balanced accuracy: -0.535.
- Spearman correlation between localized-question fraction and balanced accuracy: -0.525.
- Pearson correlation between explicit-lead-question fraction and balanced accuracy: -0.475.
- Spearman correlation between explicit-lead-question fraction and balanced accuracy: -0.506.

Interpretation:

- SCP codes that rely more heavily on localized or lead-specific evidence tend to perform worse.
- This is consistent with the architectural limitation of using one pooled ECG embedding.

Source:
`outputs/llama_best_scp_localization_burden_summary.json`

## 6. Current Interpretation

The current adapter demonstrates meaningful ECG-to-language alignment, but the performance ceiling appears to depend strongly on clinical signal type.

The evidence now supports three claims:

1. The model is using ECG information.
   - Token ablation causes balanced accuracy to collapse.
   - Shuffled ECG tokens perform close to chance.

2. The adapter preserves clinically useful CSFM information.
   - Adapter-token probes retain almost the same SCP-classification signal as the original CSFM embeddings.

3. The hardest cases are clinically meaningful rather than random.
   - Rhythm/conduction/device-related questions perform strongly.
   - ST/T, voltage, and subtle morphology questions perform poorly.
   - Lead-localized burden correlates negatively with performance.

This makes the project more scientifically defensible: the central finding is not simply that the adapter reaches a single headline score, but that frozen ECG-to-LLM alignment works unevenly across clinical ECG phenomena.

## 7. Open Questions For Supervisor Discussion

1. Should the dissertation now foreground category-level performance variation as a main scientific finding?

2. Is the current pooled CSFM embedding sufficient for the final dissertation scope, or should the next stage test lead-specific or token-sequence CSFM representations?

3. Should LoRA be deprioritized unless we can explain why it shifts the model toward `no` predictions?

4. Should final reporting include validation-tuned global threshold results, or keep the main result as default forced-choice decoding?

5. For clinical safety framing, should we describe the model as a proof of concept only, with failure analysis showing why deployment-level performance would require lead-aware representations and stronger calibration?

6. Should further experiments focus on improving positive recall for subtle morphology and ST/T codes, rather than improving overall balanced accuracy alone?

