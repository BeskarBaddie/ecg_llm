# Final Report Clarifications And Highlights

This is a living checklist for points that must be clarified, defended, or highlighted in the dissertation and supervisor updates.

## Things To Clarify

### Train / Validation / Test Split

- Clarify that the model uses the official ECG-QA train/validation/test split for the full all-SCP-code dataset.
- Clarify that the test set is intended to be blind at the ECG/example level.
- Clarify whether test questions are novel at the wording/template level.
- Distinguish between:
  - unseen ECGs,
  - unseen exact ECG-question-answer rows,
  - repeated ECG-QA templates,
  - repeated SCP-code concepts.
- Confirm whether any ECG IDs overlap across train, validation, and test.
- Report whether the same question text appears across train/validation/test.
- Add a separate metric split for:
  - test examples whose question text/template appears in training,
  - test examples whose question text/template is novel relative to training.

### Unit Of Prediction

- Clarify that the model predicts at the ECG-question pair level, not the ECG-only level.
- Clarify that a single ECG can appear with multiple questions.
- Clarify that this is expected in ECG-QA because different SCP-code attributes can be queried for the same ECG.

### Dataset Inclusion / Exclusion

- Explain the ECG-QA subset filtering:
  - `single_verify` questions only,
  - `attribute_type = scp_code`,
  - binary `yes`/`no` answers,
  - ECG must have a corresponding precomputed CSFM embedding,
  - SCP code must be available/mappable.
- Report the resulting dataset sizes:
  - train rows,
  - validation rows,
  - test rows,
  - unique ECGs,
  - number of SCP codes,
  - yes/no label balance.

### Loss Calculation

- Explain that training uses generative token-level cross-entropy.
- Explain that loss is applied only to the answer tokens.
- Explain that prompt tokens are masked from the loss.
- Explain that ECG soft-token positions are masked from the loss.
- Clarify that the LLM participates in backpropagation as a frozen differentiable computation graph, but its parameters are not updated in adapter-only experiments.
- Clarify which parameters receive gradients:
  - adapter-only run: adapter parameters only,
  - LoRA run: adapter parameters plus LoRA parameters.

### Why Answer-Token Cross-Entropy

- Clarify that the objective is not to reconstruct the full prompt.
- Clarify that the model is trained to make the correct answer string more likely.
- Clarify that `yes` and `no` are scored as candidate generated answers.
- Explain why answer-token-only loss is appropriate for generative yes/no supervision.

### Yes / No Scores

- Explain that `yes_score` and `no_score` are log-likelihood scores from the LLM.
- Explain that these are not logistic-regression probabilities.
- Explain that the model computes:
  - `log P("yes" | ECG tokens, question prompt)`,
  - `log P("no" | ECG tokens, question prompt)`.
- Explain that default forced-choice prediction chooses the answer with the higher log-likelihood.

### Threshold Tuning

- Explain the score used for threshold tuning:
  - `score_diff = yes_score - no_score`.
- Explain default prediction:
  - predict `yes` if `score_diff >= 0`.
- Explain tuned prediction:
  - predict `yes` if `score_diff >= threshold`.
- Clarify that threshold tuning is deterministic post-processing, not model retraining.
- Clarify that threshold tuning is performed on validation data.
- Clarify that the held-out test set should only use thresholds chosen from validation.
- Discuss the difference between:
  - default threshold,
  - global validation-tuned threshold,
  - per-SCP-code validation-tuned threshold.
- Highlight that per-code threshold tuning may overfit when per-code validation sample sizes are small.

### Temperature And Evaluation

- Clarify that forced-choice log-likelihood scoring does not use stochastic generation.
- If generation is used for qualitative examples, clarify whether temperature is set to zero.
- Distinguish deterministic forced-choice evaluation from free-text generation.

### LoRA

- Explain exactly what LoRA changed:
  - low-rank trainable matrices added to selected LLM attention projections.
- Clarify that the base LLM weights remained frozen.
- Clarify that LoRA trained additional parameters alongside the adapter.
- Explain that LoRA improved raw accuracy/no recall but did not improve balanced accuracy.
- Investigate whether LoRA needs:
  - longer training,
  - checkpoint resume,
  - different class balance,
  - different learning rate,
  - different target modules.

### CSFM-Only Comparison

- Clarify whether strong and weak adapter SCP codes show the same pattern using CSFM-only classifiers.
- This will help separate:
  - ECG representation limitations,
  - adapter limitations,
  - LLM interpretation/calibration limitations.

### Clinical Interpretation

- Clarify why rhythm and conduction codes tend to perform well.
- Clarify why ST/T, voltage, and subtle morphology codes tend to perform worse.
- Highlight the hypothesis that poor-performing codes may require lead-specific information that is diluted by a single pooled ECG embedding.
- Treat this as a hypothesis unless supported by CSFM-only classifier and lead-localization analyses.

## Things To Highlight In The Final Report

### Main Scientific Framing

- The dissertation is a feasibility and failure-analysis study, not a clinical deployment study.
- The main research question is whether frozen ECG representations can be aligned with a frozen LLM through a lightweight adapter.
- The key result is not only the headline balanced accuracy.
- A central finding is that performance varies systematically by clinical ECG abnormality type.

### Best Current Result

- Best adapter-only validation result:
  - CSFM-Tiny pooled embedding,
  - linear adapter,
  - 8 soft tokens,
  - frozen Llama-3.2-3B-Instruct,
  - balanced accuracy approximately 0.738.
- Global validation-tuned threshold improved validation balanced accuracy to approximately 0.762.
- On held-out test, global validation-tuned threshold achieved approximately 0.745 balanced accuracy.

### Evidence That ECG Tokens Matter

- ECG token ablation should be highlighted:
  - real ECG tokens: balanced accuracy approximately 0.738,
  - shuffled ECG tokens: approximately 0.508,
  - zero ECG tokens: approximately 0.500,
  - random ECG tokens: approximately 0.498,
  - mean ECG tokens: approximately 0.524.
- This supports the claim that the model uses ECG-specific information rather than only question text or answer priors.

### Evidence That The Adapter Preserves ECG Signal

- Adapter-token linear probe should be highlighted:
  - CSFM embeddings mean AUC approximately 0.851,
  - adapter mean-token AUC approximately 0.840,
  - adapter flat-token AUC approximately 0.835.
- This suggests that the adapter does not simply destroy the SCP-code signal.
- The remaining bottleneck is likely downstream alignment, calibration, or LLM interpretation.

### Question-Conditioned Grounding

- ECG/question swap experiment should be highlighted:
  - same ECG, different question: balanced accuracy approximately 0.833,
  - same question, different ECG: balanced accuracy approximately 0.813.
- This supports the claim that the model responds to both ECG content and question content.

### SCP-Code And Clinical Group Findings

- Strong codes include:
  - CRBBB,
  - PACE,
  - CLBBB,
  - AFIB,
  - STACH,
  - AFLT.
- Weak codes include:
  - NST_,
  - STE_,
  - TAB_,
  - HVOLT,
  - LVOLT,
  - lead-localized ischemic/ST-T findings.
- Strongest clinical groups:
  - rhythm,
  - conduction disturbance,
  - myocardial infarction/injury in some cases.
- Weaker clinical groups:
  - ST/T change,
  - form/morphology,
  - hypertrophy,
  - normal ECG questions.

### Why Balanced Accuracy Matters

- The dataset is imbalanced toward `no`.
- Raw accuracy can be misleading because a conservative model can perform well by predicting mostly `no`.
- Balanced accuracy better reflects performance across positive and negative cases.
- Yes recall should be reported explicitly because missing positive ECG findings is clinically important.

### Limitations To State Clearly

- Current model is not clinically safe.
- The test set does not necessarily contain novel question templates.
- The current ECG representation is one pooled embedding per ECG.
- Pooled embeddings may lose lead-specific or morphology-specific information.
- The model currently handles only binary yes/no SCP-code questions.
- The LLM is mostly frozen; full LLM adaptation has not yet shown clear benefit.
- Threshold tuning improves calibration but does not solve the underlying representation/reasoning problem.

## Action Items To Track

- [ ] Verify ECG ID overlap across train/validation/test.
- [ ] Verify exact question text/template overlap across train/validation/test.
- [ ] Add novel-question metrics for the test set.
- [ ] Plot `yes_score`, `no_score`, and `score_diff` distributions by true answer.
- [ ] Plot score distributions by SCP code and clinical group.
- [ ] Compare top and worst SCP codes using CSFM-only classifiers.
- [ ] Export qualitative examples with ECG waveform, question, answer, prediction, and scores.
- [ ] Add checkpoint resume support if longer ARC jobs are required.
- [ ] Run or continue a more complete LoRA experiment only after checkpointing is robust.
- [ ] Try upsampling positive `yes` examples during adapter training.
- [ ] Consider rephrasing positive training questions while keeping validation/test unchanged.
- [ ] Consider adapter bottleneck experiments to limit ECG-token capacity.
- [ ] Decide whether final results should report default threshold only, global threshold tuning, or both.

