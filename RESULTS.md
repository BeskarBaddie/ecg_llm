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

Additional supervisor-requested checks used PCA to project ECG and text embeddings separately to 256 dimensions, then compared concatenation and mean-pooling fusion across three sentence embedding models:

| Text model | Method | Accuracy | Balanced Accuracy | Macro-F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | ECG-only, PCA-256 | 0.689 | 0.671 | 0.659 | 0.721 |
| all-MiniLM-L6-v2 | Text-only, PCA-256 | 0.543 | 0.510 | 0.505 | 0.523 |
| all-MiniLM-L6-v2 | ECG + text concat, PCA-256 | 0.514 | 0.515 | 0.497 | 0.544 |
| all-MiniLM-L6-v2 | ECG + text mean, PCA-256 | 0.690 | 0.671 | 0.660 | 0.728 |
| multi-qa-MiniLM-L6-cos-v1 | ECG-only, PCA-256 | 0.689 | 0.671 | 0.659 | 0.721 |
| multi-qa-MiniLM-L6-cos-v1 | Text-only, PCA-256 | 0.507 | 0.510 | 0.491 | 0.513 |
| multi-qa-MiniLM-L6-cos-v1 | ECG + text concat, PCA-256 | 0.543 | 0.509 | 0.504 | 0.541 |
| multi-qa-MiniLM-L6-cos-v1 | ECG + text mean, PCA-256 | 0.701 | 0.692 | 0.676 | 0.730 |
| all-mpnet-base-v2 | ECG-only, PCA-256 | 0.689 | 0.671 | 0.659 | 0.721 |
| all-mpnet-base-v2 | Text-only, PCA-256 | 0.505 | 0.507 | 0.489 | 0.517 |
| all-mpnet-base-v2 | ECG + text concat, PCA-256 | 0.510 | 0.511 | 0.493 | 0.541 |
| all-mpnet-base-v2 | ECG + text mean, PCA-256 | 0.687 | 0.677 | 0.661 | 0.731 |
| ncbi/MedCPT-Query-Encoder | ECG-only, PCA-256 | 0.689 | 0.671 | 0.659 | 0.721 |
| ncbi/MedCPT-Query-Encoder | Text-only, PCA-256 | 0.526 | 0.485 | 0.482 | 0.514 |
| ncbi/MedCPT-Query-Encoder | ECG + text concat, PCA-256 | 0.488 | 0.482 | 0.468 | 0.527 |
| ncbi/MedCPT-Query-Encoder | ECG + text mean, PCA-256 | 0.597 | 0.597 | 0.576 | 0.641 |

Interpretation:

The ECG-only and combined baselines now perform above chance after correcting the CSFM embedding generation. Text-only remains near chance, which supports the claim that question wording alone is not solving this task. The combined model only modestly improves over ECG-only, so the current setup is mostly testing whether the ECG embedding supports the attribute label.

The PCA/fusion check gives a more precise answer to the supervisor's concern. PCA-256 slightly improves ECG-only balanced accuracy from 0.658 to 0.671. Text-only remains near chance across all four text encoders, so the question embedding by itself still does not solve the task. Simple concatenation after PCA performs poorly, suggesting the classifier is not making good use of the two projected spaces when they are just appended. Mean fusion performs best with `multi-qa-MiniLM-L6-cos-v1`, where ECG + text reaches 0.692 balanced accuracy versus 0.671 for ECG-only. `ncbi/MedCPT-Query-Encoder` was tested as a biomedical query encoder, but it did not improve this baseline; its ECG + text mean-fusion balanced accuracy was 0.597.

Additional question-sample-size and one-hot checks kept the validation set fixed and changed only the training representation/sample count. The one-hot encoder represents each exact repeated ECG-QA question as a question ID rather than a dense sentence embedding.

| Train cap | Train Rows | ECG-only Balanced Accuracy | One-hot Text-only Balanced Accuracy | ECG + One-hot Concat Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 25/question | 250 | 0.617 | 0.508 | 0.612 |
| 50/question | 500 | 0.623 | 0.520 | 0.636 |
| 100/question | 1000 | 0.607 | 0.486 | 0.616 |
| 200/question | 2000 | 0.626 | 0.513 | 0.623 |
| Full, about 284-298/question | 2918 | 0.658 | 0.494 | 0.665 |

This suggests two things. First, using all available training rows is better than artificially reducing the per-question training sample size. Second, exact question identity is almost as useful as the original dense text embedding for concat fusion, which supports the suspicion that the text side is mostly identifying the target SCP/question rather than adding rich semantic information.

Key files:

```text
train_ecgqa_embedding_baselines.py
outputs/ecgqa_embedding_baseline_results.json
outputs/ecgqa_embedding_baseline_predictions.jsonl
outputs/ecgqa_embedding_baseline_models.joblib
outputs/ecgqa_embedding_baseline_minilm_pca256_results.json
outputs/ecgqa_embedding_baseline_multiqa_minilm_pca256_results.json
outputs/ecgqa_embedding_baseline_mpnet_pca256_results.json
outputs/ecgqa_embedding_baseline_medcpt_pca256_results.json
outputs/ecgqa_embedding_baseline_onehot_results.json
outputs/ecgqa_embedding_baseline_onehot_cap25_results.json
outputs/ecgqa_embedding_baseline_onehot_cap50_results.json
outputs/ecgqa_embedding_baseline_onehot_cap100_results.json
outputs/ecgqa_embedding_baseline_onehot_cap200_results.json
```

## Part 3A: Hand-Crafted Feature-to-LLM Baseline

Objective: evaluate a simple LLM baseline where raw ECGs are converted into hand-crafted text features and passed to an LLM with the ECG-QA question.

The first version used a compact feature representation with per-lead values such as:

- `n_rpeaks`
- `signal_mean`
- `signal_std`
- `rr_mean`
- `rr_std`
- `heart_rate_est`

The improved classifier-interpreted version uses the SignalMC-style 54-dimensional lead-II domain features. It also feeds the LLM the probabilities from domain-feature classifiers trained on those same features, plus short clinical sentences describing rhythm/timing features. This gives the LLM a stronger ECG representation than raw feature values alone, but it is not zero-shot because it relies on trained SCP-code classifiers.

Supervisor-requested zero-shot variants removed the classifier evidence from the prompt. The full zero-shot interpreted run used lead-II SignalMC features summarized as clinical rhythm/timing sentences. A smaller diagnostic run also listed all 54 feature names, descriptions, and values directly in the prompt.

| Method | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Compact feature-to-LLM | 0.715 | 0.500 | 0.417 |
| SignalMC interpreted feature-to-LLM, zero-shot | 0.654 | 0.500 | 0.457 |
| SignalMC classifier-interpreted feature-to-LLM | 0.742 | 0.682 | 0.689 |
| SignalMC described 54-feature prompt, zero-shot mixed-10 only | 0.500 | 0.500 | 0.333 |

Per-code balanced accuracy for the improved SignalMC classifier-interpreted feature-to-LLM baseline:

| Code | N | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| AFIB | 57 | 0.906 | 0.900 |
| ASMI | 59 | 0.500 | 0.398 |
| CLBBB | 50 | 0.800 | 0.740 |
| CRBBB | 59 | 0.735 | 0.726 |
| LAFB | 60 | 0.925 | 0.909 |
| LVH | 234 | 0.589 | 0.591 |
| NORM | 59 | 0.680 | 0.659 |

Interpretation:

The compact feature prompt collapses to majority-class behavior. Removing classifier evidence from the SignalMC prompt also collapses to near-majority behavior: the zero-shot interpreted prompt predicts mostly `no` and reaches only 0.500 balanced accuracy on the full validation set. The described 54-feature prompt was much slower and, on the mixed 10-row diagnostic subset, also predicted all rows as `no`. The classifier-interpreted prompt is materially better, but its performance depends on trained SCP-code classifiers, so it should be reported separately from the zero-shot textified-ECG baseline.

Key files:

```text
evaluate_feature_llm_baseline.py
outputs/feature_llm_baseline_results.json
outputs/feature_llm_baseline_predictions.jsonl
evaluate_domain_feature_llm_baseline.py
outputs/domain_feature_llm_baseline_interpreted54_zeroshot_full_results.json
outputs/domain_feature_llm_baseline_described54_zeroshot_mixed10_results.json
outputs/domain_feature_llm_baseline_classifier_interpreted_full_results.json
outputs/domain_feature_llm_baseline_classifier_interpreted_full_predictions.jsonl
```

## Domain Feature Classifier Validation

Objective: validate whether the SignalMC-style ECG domain features contain SCP-code signal before using them as text input to an LLM.

Setup:

- one classifier per SCP code
- official ECG-QA train/validation split
- lead II only
- logistic regression with class balancing
- features extracted from raw PTB-XL ECGs using the SignalMC-style extractor

Aggregate results:

| Feature Set | Mean Accuracy | Mean Balanced Accuracy | Mean Macro-F1 | Mean ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| First 10 SignalMC-style lead-II features | 0.723 | 0.725 | 0.698 | 0.779 |
| Full 54 SignalMC-style lead-II features | 0.774 | 0.765 | 0.749 | 0.833 |

Per-code ROC AUC:

| Code | 10 Features | 54 Features |
| --- | ---: | ---: |
| AFIB | 0.887 | 0.954 |
| ASMI | 0.804 | 0.755 |
| CLBBB | 0.800 | 0.898 |
| CRBBB | 0.726 | 0.801 |
| LAFB | 0.915 | 0.985 |
| LVH | 0.618 | 0.642 |
| NORM | 0.704 | 0.792 |

Interpretation:

The extracted domain features are not random: even the first 10 lead-II features produce above-chance SCP classifiers, and the full 54-feature set improves the aggregate AUC. This validates the extractor as a meaningful baseline input. However, the features are still weaker than CSFM embeddings and remain especially weak for LVH. The LLM feature-prompt failure is therefore not simply caused by useless features; it is also a prompt/model-conditioning problem and a limitation of single-lead handcrafted summaries.

Key files:

```text
train_domain_feature_classifiers.py
outputs/domain_feature_classifier_signalmc_leadII_10_logreg_results.json
outputs/domain_feature_classifier_signalmc_leadII_54_logreg_results.json
outputs/domain_feature_cache_signalmc_leadII_10.jsonl
outputs/domain_feature_cache_signalmc_leadII_54.jsonl
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
| Classifier-to-LLM, Llama 3.2 1B | 0.458 | 0.523 | 0.458 | 0 |
| Classifier-to-LLM, Llama 3.1 8B | 0.676 | 0.747 | 0.674 | 0 |
| Classifier-to-LLM, Llama 3.2 3B | 0.820 | 0.763 | 0.778 | 0 |

The original classifier-to-LLM run used the local Ollama model `llama3.1:latest`. `ollama show` reports this model as an 8.0B parameter Llama model using the Llama chat template. The smaller comparison runs use `llama3.2:3b` and `llama3.2:1b`.

Prediction counts:

| Model | no | yes |
| --- | ---: | ---: |
| Llama 3.2 1B | 192 | 386 |
| Llama 3.1 8B | 230 | 348 |
| Llama 3.2 3B | 435 | 143 |

Per-code classifier-to-LLM balanced accuracy:

| Code | Llama 3.2 1B | Llama 3.2 3B | Llama 3.1 8B |
| --- | ---: | ---: | ---: |
| AFIB | 0.500 | 0.876 | 0.947 |
| ASMI | 0.526 | 0.724 | 0.847 |
| CLBBB | 0.537 | 0.975 | 0.950 |
| CRBBB | 0.540 | 1.000 | 0.974 |
| LAFB | 0.587 | 0.912 | 0.813 |
| LVH | 0.527 | 0.621 | 0.560 |
| NORM | 0.500 | 0.742 | 0.717 |

Interpretation:

The direct classifier threshold is the upper-bound-style check requested by the supervisors. It is strong overall, especially for AFIB, CLBBB, CRBBB, ASMI, and LAFB. LVH is the weak point and is also the code with the largest label-alignment disagreement.

The LLM degrades performance relative to directly using the classifier output. The 1B model performs close to chance balanced accuracy and over-predicts `yes`. The 8B model also tends to over-predict `yes`, especially for LVH. The 3B model is more conservative, predicts `no` more often, and gives better overall accuracy/macro-F1 on this validation set. These runs do not show monotonic performance scaling with model size; prompt behavior and decision bias matter at least as much as parameter count for this baseline.

Key files:

```text
evaluate_classifier_llm_baseline.py
outputs/classifier_llm_baseline_results.json
outputs/classifier_llm_baseline_predictions.jsonl
outputs/classifier_llm_attribute_models.joblib
outputs/classifier_llm_baseline_llama3_2_1b_results.json
outputs/classifier_llm_baseline_llama3_2_1b_predictions.jsonl
outputs/classifier_llm_attribute_models_llama3_2_1b.joblib
outputs/classifier_llm_baseline_llama3_2_3b_results.json
outputs/classifier_llm_baseline_llama3_2_3b_predictions.jsonl
outputs/classifier_llm_attribute_models_llama3_2_3b.joblib
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

Run SignalMC-style domain feature classifier validation:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache /Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python train_domain_feature_classifiers.py --feature-set signalmc --lead II --max-features 54 --classifier logreg --feature-cache-path outputs/domain_feature_cache_signalmc_leadII_54.jsonl --results-path outputs/domain_feature_classifier_signalmc_leadII_54_logreg_results.json --predictions-path outputs/domain_feature_classifier_signalmc_leadII_54_logreg_predictions.jsonl --model-bundle-path outputs/domain_feature_classifier_signalmc_leadII_54_logreg_models.joblib
```

Run classifier-output-to-LLM baseline:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --model llama3.1 --num-predict 4
```

Run smaller Llama 3.2 3B classifier-output-to-LLM baseline:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --model llama3.2:3b --num-predict 4 --results-path outputs/classifier_llm_baseline_llama3_2_3b_results.json --predictions-path outputs/classifier_llm_baseline_llama3_2_3b_predictions.jsonl --model-bundle-path outputs/classifier_llm_attribute_models_llama3_2_3b.joblib
```

Run smaller Llama 3.2 1B classifier-output-to-LLM baseline:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --model llama3.2:1b --num-predict 4 --results-path outputs/classifier_llm_baseline_llama3_2_1b_results.json --predictions-path outputs/classifier_llm_baseline_llama3_2_1b_predictions.jsonl --model-bundle-path outputs/classifier_llm_attribute_models_llama3_2_1b.joblib
```

Run direct classifier threshold only:

```bash
/Users/tarironathanbanganayi/miniconda3/envs/csfm/bin/python evaluate_classifier_llm_baseline.py --skip-llm --results-path outputs/classifier_llm_threshold_tuning_results.json --predictions-path outputs/classifier_llm_threshold_tuning_predictions.jsonl --model-bundle-path outputs/classifier_llm_attribute_models_threshold_tuning.joblib
```
