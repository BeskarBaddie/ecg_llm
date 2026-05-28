# Results Summary

This document summarizes the current ECG-QA subset and baseline results after the corrected CSFM embedding rebuild.

Important correction: earlier official-split baseline results were generated from embeddings produced by scripts that instantiated `CSFM_model("Tiny")` without loading the pretrained CSFM checkpoint consistently. Those numbers are now treated as invalid debugging outputs. The current results below use embeddings generated in one run from the pretrained CSFM Tiny checkpoint.

## Current Data

All current results use the official ECG-QA template train/validation split:

```text
outputs/ecgqa_scp_binary_train.jsonl
outputs/ecgqa_scp_binary_val.jsonl
```

The corrected dataset was built by:

```text
build_ecgqa_scp_official_embeddings.py
```

This script loads:

```text
/Users/tarironathanbanganayi/Desktop/Oxford/Dissertation/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth
```

and extracts train/validation embeddings using the same frozen pretrained CSFM instance.

## Part 1: Binary ECG-QA SCP Subset

Objective: create a controlled ECG-QA subset with binary `yes`/`no` answers, `single-verify` questions, `attribute_type == "scp_code"`, atrial fibrillation plus additional SCP codes, and the official ECG-QA train/validation split.

Selected codes:

| Code | Description |
| --- | --- |
| AFIB | atrial fibrillation |
| ASMI | anteroseptal myocardial infarction |
| CLBBB | complete left bundle branch block |
| CRBBB | complete right bundle branch block |
| LAFB | left anterior fascicular block |
| LVH | left ventricular hypertrophy |
| NORM | normal ECG |

Current dataset statistics:

| Statistic | Count |
| --- | ---: |
| Total questions | 3,496 |
| Unique ECGs | 2,868 |
| Train questions | 2,918 |
| Validation questions | 578 |
| Yes answers | 1,154 |
| No answers | 2,342 |
| Missing local signal rows skipped | 91 |

Questions per code:

| Code | Questions |
| --- | ---: |
| LVH | 1,412 |
| LAFB | 354 |
| NORM | 352 |
| CRBBB | 351 |
| AFIB | 346 |
| ASMI | 343 |
| CLBBB | 338 |

The label-alignment audit compares ECG-QA labels with PTB-XL SCP metadata. With `LVH -> VCLVH` aliasing and key-presence handling for AFIB/LVH, the overall agreement is:

| Metric | Value |
| --- | ---: |
| Agreement accuracy | 0.904 |
| Balanced accuracy | 0.908 |
| Macro-F1 | 0.895 |

The remaining disagreement is mainly LVH, where ECG-QA and PTB-XL metadata do not align perfectly.

Key files:

```text
outputs/ecgqa_scp_binary_subset.jsonl
outputs/ecgqa_scp_binary_train.jsonl
outputs/ecgqa_scp_binary_val.jsonl
outputs/ecgqa_scp_binary_subset_stats.json
outputs/ecgqa_label_alignment_corrected_results.json
```

## Part 2: Embedding-Based Baselines

Objective: evaluate simple non-LLM baselines on the binary ECG-QA subset:

- majority baseline
- ECG-only: CSFM ECG embedding -> classifier
- text-only: question embedding -> classifier
- combined: CSFM ECG embedding + question embedding -> classifier

The question encoder was:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Results on the official validation split:

| Method | Accuracy | Balanced Accuracy | Macro-F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.683 | 0.500 | 0.406 | 0.500 |
| ECG-only | 0.678 | 0.658 | 0.647 | 0.704 |
| Text-only | 0.422 | 0.494 | 0.422 | 0.477 |
| Combined | 0.692 | 0.663 | 0.656 | 0.721 |

Interpretation:

The ECG-only and combined baselines now perform above chance after correcting the CSFM embedding generation. Text-only remains near chance, which supports the claim that question wording alone is not solving this task. The combined model only modestly improves over ECG-only, so the current setup is mostly testing whether the ECG embedding supports the attribute label.

Key files:

```text
train_ecgqa_embedding_baselines.py
outputs/ecgqa_embedding_baseline_results.json
outputs/ecgqa_embedding_baseline_predictions.jsonl
outputs/ecgqa_embedding_baseline_models.joblib
```

## Part 3A: Hand-Crafted Feature-to-LLM Baseline

Objective: evaluate a simple LLM baseline where raw ECGs are converted into hand-crafted text features and passed to an LLM with the ECG-QA question.

The feature representation includes per-lead values such as:

- `n_rpeaks`
- `signal_mean`
- `signal_std`
- `rr_mean`
- `rr_std`
- `heart_rate_est`

Previous full-run result:

| Method | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Feature-to-LLM | 0.715 | 0.500 | 0.417 |

Interpretation:

This baseline collapses to majority-class behavior. The hand-crafted features are too weak for many SCP-code diagnostic questions because they do not capture enough morphology, voltage, conduction-pattern, or infarction-pattern information.

Key files:

```text
evaluate_feature_llm_baseline.py
outputs/feature_llm_baseline_results.json
outputs/feature_llm_baseline_predictions.jsonl
```

## Part 3B: Classifier-Output-to-LLM Baseline

Objective: implement a baseline closer to the ECG-QA paper's upper-bound-to-LLM setup:

1. Convert ECG-QA training rows into SCP attribute labels.
2. Train one attribute classifier per SCP code from CSFM embeddings.
3. Convert classifier probabilities into text.
4. Give classifier findings plus the ECG-QA question to the LLM.
5. Evaluate strict `yes`/`no` answers on the official validation split.

Results:

| Method | Accuracy | Balanced Accuracy | Macro-F1 | Invalid Outputs |
| --- | ---: | ---: | ---: | ---: |
| Direct attribute threshold | 0.817 | 0.788 | 0.788 | 0 |
| Tuned direct threshold | 0.830 | 0.816 | 0.808 | 0 |
| Classifier-to-LLM | 0.676 | 0.747 | 0.674 | 0 |

Per-code classifier-to-LLM balanced accuracy:

| Code | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: |
| CRBBB | 0.974 | 0.963 |
| AFIB | 0.947 | 0.940 |
| CLBBB | 0.950 | 0.890 |
| ASMI | 0.847 | 0.808 |
| LAFB | 0.813 | 0.762 |
| NORM | 0.717 | 0.704 |
| LVH | 0.560 | 0.385 |

Interpretation:

The direct classifier threshold is the upper-bound-style check requested by the supervisors. It is strong overall, especially for AFIB, CLBBB, CRBBB, ASMI, and LAFB. LVH is the weak point and is also the code with the largest label-alignment disagreement.

The LLM degrades performance relative to directly using the classifier output. It tends to over-predict `yes`, especially for LVH, so the classifier probabilities are more reliable than the current text-prompted LLM decision rule.

Key files:

```text
evaluate_classifier_llm_baseline.py
outputs/classifier_llm_baseline_results.json
outputs/classifier_llm_baseline_predictions.jsonl
outputs/classifier_llm_attribute_models.joblib
```

## Current Conclusion

The corrected results support these claims:

1. The official ECG-QA SCP subset is now built from pretrained CSFM embeddings and official train/validation splits.
2. ECG embeddings contain usable signal for the selected SCP-code yes/no questions.
3. Question text alone does not solve the task.
4. The upper-bound-style direct classifier is substantially stronger than the current LLM prompt baseline.
5. LVH is the main weak label/code and should be handled carefully in supervisor discussion.
6. The next modeling step should compare any projection/transformer approach against the direct classifier and ECG-only/combined embedding baselines, not against the earlier invalid chance-level outputs.

## Reproduction Commands

Build the corrected official ECG-QA SCP subset:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache /Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python build_ecgqa_scp_official_embeddings.py
```

Run label-alignment audit:

```bash
python analyze_ecgqa_label_alignment.py --results-path outputs/ecgqa_label_alignment_corrected_results.json --mismatches-path outputs/ecgqa_label_alignment_corrected_mismatches.csv --code-alias LVH=VCLVH --key-presence-code AFIB --key-presence-code VCLVH
```

Run embedding baselines:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache /Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python train_ecgqa_embedding_baselines.py --question-encoder sentence-transformer --question-model all-MiniLM-L6-v2
```

Run classifier-output-to-LLM baseline:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --model llama3.1 --num-predict 4
```

Run direct classifier threshold only:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --skip-llm --results-path outputs/classifier_llm_threshold_tuning_results.json --predictions-path outputs/classifier_llm_threshold_tuning_predictions.jsonl --model-bundle-path outputs/classifier_llm_attribute_models_threshold_tuning.joblib
```
