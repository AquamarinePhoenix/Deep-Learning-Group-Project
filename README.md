# Deep Learning Group Project

---

> [!NOTE]
> This README is adjusted for the high-level presentation of the project.

## CONTENT LIST

---

- [INTRO](#abstract)
- [GOAL AND OBJECTIVES](#goal-and-objectives)
- [DATASET](#dataset-summary)
- [METHODOLOGY](#methodology)
- [PIPELINE DESIGN](#pipeline-design-conceptual)
- [PARAMETRIZATION](#parametrization)
- [RESULTS](#results-summary)
- [CONCLUSIONS](#conclusions)
- [FURTHER STUDY](#further-study)

> [!NOTE]
> Dataset of .csv and images folder should be inserted into data folder, that is, data/clickbait_dataset_vietnamese.csv and data/images/

## Abstract

This project implements a concise pipeline for detecting clickbait in Vietnamese text. It was designed to balance reproducibility, interpretability, and fast iteration: lightweight dataset sampling supports rapid experiments while a shard-based training strategy reduces memory pressure on single-machine environments. The goal is to compare two pretrained language model backbones on the same stratified splits and to evaluate their utility for binary clickbait classification.

## Goal and objectives

Goal: Produce a reliable, reproducible classifier that distinguishes clickbait from non-clickbait in Vietnamese text and quantify model performance using standard classification metrics.

Objectives:

- Curate and preprocess a representative Vietnamese headline dataset with stratified train/validation/test splits.
- Evaluate two pretrained sequence-classification backbones under identical data and tokenization regimes.
- Measure classifier quality with precision, recall, F1, and accuracy, and analyze trade-offs relevant to downstream deployment.
- Offer a compact workflow that permits small-sample experiments for rapid prototyping and full-data runs for final evaluation.

## Background & Motivation

Clickbait detection is a binary classification problem with asymmetric costs: false positives (mislabeling legitimate headlines as clickbait) and false negatives (missing clickbait) have different practical consequences depending on downstream applications (moderation, ranking, analytics). Vietnamese presents linguistic challenges (morphology, tokenization, domain-specific expressions) that motivated evaluation of multilingual vs. native pretrained models.

The project's design favors clarity and measurability: use stratified splitting to preserve label ratios, explicit timing and logging for reproducibility, and lightweight model checkpoints for portability between runs.

## Dataset Summary

- Source: Vietnamese clickbait dataset (repository contains a local CSV and JSONL mirror).
- Class balance: skewed; statistics are preserved via stratified splits so evaluation reflects real-world ratios.
- Splits: train, validation, test (stratified); a `primary_ratio` parameter supports sampling a smaller portion of data for quick experiments.

> [!CAUTION]
> **Class Imbalance**: The default `PRIMARY_DATA_RATIO=0.01` produces only ~3–4 positive examples in the training set (~272 total samples). This extreme imbalance can lead to convergence issues, underfitting, or majority-class prediction. Consider increasing `PRIMARY_DATA_RATIO` or using class-weighted loss for better positive-class learning.

## Methodology

Problem statement: given a headline text $x \in \mathcal{X}$, predict label $y \in \{0,1\}$ (0 = non-clickbait, 1 = clickbait). We model the conditional probability $p(y \mid x; \theta)$ with a parameterized classifier built on pretrained encoders and a lightweight classification head.

Model output: a two-dimensional logit vector $z(x; \theta) \in \mathbb{R}^2$. We convert logits to probabilities via softmax:

$$
\hat{p}_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \quad (i \in \{0,1\})
$$

Training objective: minimize the cross-entropy loss over the training set $\{(x_k, y_k)\}_{k=1}^N$:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{k=1}^N \log \hat{p}_{y_k}(x_k; \theta)
$$

Evaluation metrics used throughout:

- Accuracy: $\text{Acc} = \frac{TP+TN}{TP+TN+FP+FN}$
- Precision: $\text{Prec} = \frac{TP}{TP+FP}$
- Recall: $\text{Rec} = \frac{TP}{TP+FN}$
- F1: $F1 = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$

These metrics capture complementary aspects: precision prioritizes correctness among positive predictions, recall prioritizes coverage of positives, and F1 balances both.

**Model selection and ensembling rationale:**

- Two backbones were selected: a strong multilingual encoder and a Vietnamese-specialized encoder. Comparing them isolates the effect of pretraining language specificity versus multilingual breadth.
- Shard-based training (splitting training indices deterministically) is used to enable multiple independent runs on limited hardware; final aggregation weights worker metrics by shard size to approximate a single-model evaluation while still keeping per-shard checkpoints.

# TF-IDF (Term Frequency-Inverse Document Frequency)

- TF-IDF is a statistical method used in natural language processing and information retrieval to evaluate how important a word is to a document in relation to a larger collection of documents. TF-IDF combines two components:

1. Term Frequency (TF): Measures how often a word appears in a document. A higher frequency suggests greater importance. If a term appears frequently in a document, it is likely relevant to the document's content.

$$\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$$

2. Inverse Document Frequency (IDF): Reduces the weight of common words across multiple documents while increasing the weight of rare words. If a term appears in fewer documents, it is more likely to be meaningful and specific.

$$\text{IDF}(t, D) = \log\left(\frac{\text{Total number of documents in corpus } D}{\text{Number of documents containing term } t}\right)$$



D1: "I like it so so much"  
D2: "I do not like it"

---

Unique words from all documents:

| Term |
|------|
| I |
| like |
| it |
| so |
| much |
| do |
| not |

TF Vectors:

tf_D1 = [1, 1, 1, 2, 1, 0, 0]

tf_D2 = [1, 1, 1, 0, 0, 1, 1]

IDF = [0, 0 , 0, log(2), log(2), log(2), log(2)]

Final TF-IDF Vectors:

tf_idf_D1 = [0, 0, 0, 1.386, 0.693, 0, 0]

tf_idf_D2 = [0, 0, 0, 0, 0, 0.693, 0.693]


## Pipeline Design (Conceptual)

### Data Flow Diagram

```
┌─────────────────┐
│  Raw CSV/JSONL  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────────────┐
│  Data Loading   │──────▶│  Stratified Sampling     │
└────────┬────────┘       │  (primary_ratio split)   │
         │                └──────────────────────────┘
         ▼
┌─────────────────┐       ┌──────────────────────────┐
│  Train/Val/Test │──────▶|Preserve Label Ratios     │
│      Split      |       |                          |
|    (80/10/10)   │       │  (train/val/test split)  │
└────────┬────────┘       └──────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │       (via pretrained tokenizer)
│  & Encoding     │       (truncate/pad to MAX_LENGTH)
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────────────┐
│  Model Loading  │──────▶│  Classification Head     │
│  (xlm/phobert)  │       │  + Encoder Fine-tuning   │
└────────┬────────┘       └──────────────────────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────────────┐
│  Training Loop  │──────▶│  Adam optimizer,         │
│  (EPOCHS iter)  │       │  cross-entropy loss      │
└────────┬────────┘       └──────────────────────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────────────┐
│ Validation &    │──────▶│  Track metrics on val    │
│ Checkpoint Save │       │  (select best by acc)    │
└────────┬────────┘       └──────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Test Inference │       (on best checkpoint)
│  & Metrics      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Report Accuracy, Precision, Recall, F1, Loss   │
└─────────────────────────────────────────────────┘
```

### Module Breakdown

- **Data ingestion and preprocessing**: text normalization, tokenization via backbone tokenizer, and preservation of the label distribution via stratified sampling.
- **Tokenization / encoding**: use the pretrained tokenizer to map text to token sequences, truncating/padding to a fixed maximum length $L$ for batching.
- **Optimization**: fine-tune the classification head (and usually the final encoder layers) with standard Adam/variant optimizers and cross-entropy loss.
- **Evaluation**: run inference on validation and test splits; select the best checkpoint by validation accuracy (or other metric) and report test-set metrics for final comparison.

> [!WARNING]
> **Limited Epochs**: The default `EPOCHS=3` may not be sufficient for deep models to converge, especially on small datasets. Monitor validation metrics closely; consider increasing epochs or implementing early stopping if validation loss plateaus.

## Parametrization

The pipeline exposes the following parameters for fine-tuning and experimentation:

> [!NOTE]
> All parameters are editable in `_modules/config.py` for permanent changes or via CLI arguments in `main.py` and `_modules/train.py` for per-run overrides.

| Technical Name | Display Name | Purpose | Default Value |
|---|---|---|---|
| `LEARNING_RATE` | Learning Rate | Optimizer step size for fine-tuning | 2e-5 |
| `BATCH_SIZE` | Batch Size | Training batch size per device | 8 |
| `EPOCHS` | Number of Epochs | Training iterations over full dataset | 3 |
| `MAX_LENGTH` | Max Sequence Length | Tokenizer truncation/padding length | 256 |
| `WARMUP_RATIO` | Warmup Ratio | Fraction of training steps for learning rate warmup | 0.1 |
| `PRIMARY_DATA_RATIO` | Primary Data Ratio | Fraction of dataset to sample before train/val/test split | 0.01 |
| `RANDOM_STATE` | Random Seed | Seed for reproducibility and stratified splitting | 42 |
| `TEST_SIZE` | Test Split Ratio | Fraction of data reserved for final test split | 0.2 |
| `VAL_SIZE` | Validation Split Ratio | Fraction of training set reserved for validation | 0.5 |

All parameters are configurable via `_modules/config.py` or CLI overrides in `main.py` and `_modules/train.py`. Fine-tuning parameters (learning rate, batch size, epochs, max length, warmup ratio) control the model training dynamics, while data parameters (primary ratio, random state, test/val size) control dataset construction and sampling behavior.

## Results (Summary)

The latest runs show a clear difference between the two backbones:

- `xlm-roberta-base`
  - Precision: 0.0000
  - Recall: 0.0000
  - F1-score: 0.0000
  - Accuracy: 0.7143
- `vinai/phobert-base`
  - Precision: 0.6667
  - Recall: 0.2000
  - F1-score: 0.3077
  - Accuracy: 0.7429

Interpretation:

- `xlm-roberta-base` reached the same accuracy as a majority-class style baseline, which suggests it predicted the negative class almost exclusively on the test set.
- `vinai/phobert-base` produced a more balanced outcome, with non-zero precision, recall, and F1, although recall remains low.
- In this run, `vinai/phobert-base` is the stronger model overall because it improves both discriminative quality and accuracy over the multilingual baseline.

> [!WARNING]
> **Multilingual Model Underfitting**: The `xlm-roberta-base` model's zero precision/recall suggests it converged to a majority-class predictor on this small, imbalanced dataset. This is a known limitation of multilingual encoders on low-resource tasks; increasing training data or fine-tuning duration may help, but language-specialized pretraining appears advantageous here.

## Conclusions

The experiment suggests that a Vietnamese-specialized encoder is more suitable for this dataset than a general multilingual encoder, at least under the current training settings. The multilingual model did not learn enough positive-class signal to produce meaningful recall, while `vinai/phobert-base` captured some clickbait patterns and gave the best overall trade-off.

The practical implication is that future work should continue from the Vietnamese-specialized backbone and focus on improving positive-class recall through threshold tuning, longer training, class-aware sampling, or additional hyperparameter search. The current pipeline remains useful because it makes these comparisons reproducible and keeps the validation split available for model selection.

## Further Study

The current implementation provides a foundation for clickbait detection in Vietnamese, but several directions warrant further investigation:

1. **Data Augmentation & Sampling**:
   - Implement class-weighted loss or focal loss to penalize minority-class misclassification more heavily.
   - Use data augmentation (back-translation, paraphrasing) to increase the effective training set size and positive-class representation.
   - Consider oversampling positive examples or undersampling negative examples to rebalance the dataset.

2. **Hyperparameter Optimization**:
   - Perform systematic grid or Bayesian search over learning rate, batch size, and warmup ratio.
   - Extend training epochs incrementally and track validation metrics to identify optimal stopping points.
   - Experiment with higher `MAX_LENGTH` values if truncation is discarding critical clickbait signals.

3. **Threshold Tuning**:
   - Currently, predictions are binary (class 1 if logit > 0.5). Sweep decision thresholds on the validation set to balance precision and recall according to application needs.
   - Use precision-recall curves to identify operating points optimal for downstream use cases (e.g., high precision for user-facing filters, high recall for internal logging).

4. **Ensemble & Multi-Model Approaches**:
   - Combine predictions from `xlm-roberta-base` and `vinai/phobert-base` via voting or weighted averaging to exploit complementary strengths.
   - Experiment with other Vietnamese-specialized models (e.g., PhoBERT v2, XLMR-large) or domain-adapted encoders.

5. **Dataset & Domain Expansion**:
   - Collect additional labeled clickbait examples to increase the positive-class budget and improve statistical reliability.
   - Explore transfer learning from related clickbait datasets in other languages or domains to bootstrap cold-start scenarios.
   - Analyze false positives and false negatives to identify systematic blind spots (e.g., specific headline types or language patterns).

6. **Model Interpretation**:
   - Use attention visualization or feature importance techniques to understand which parts of headlines drive clickbait predictions.
   - Generate explanations for individual predictions to build user trust and guide refinement of training data or model architecture.

7. **Production Deployment**:
   - Implement confidence calibration so model scores reflect true probabilities, enabling downstream risk assessment.
   - Set up monitoring to detect performance degradation over time as headline distributions shift.
   - Create A/B tests to validate improvements in downstream metrics (e.g., user engagement, content quality ratings).
