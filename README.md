# Deep Learning Group Project

---

> [!NOTE]
> This README reflects the current pipeline, including the newer timing output, shard-based training, and the two-base-model comparison flow.

## CONTENT LIST

---

- [INTRO](#intro)
- [MAIN GOAL AND OBJECTIVES TO REACH THE GOAL](#main-goal-and-objectives-to-reach-the-goal)
- [DATASET](#dataset)
- [RECENT CHANGES](#recent-changes)
- [PARALLEL TRAINING](#parallel-training)
- [EDA](#eda)
- [METHODOLOGY/APPROACH TO A PROBLEM](#methodologyapproach-to-a-problem)
- [EXPLAINING PARAMETRIZATION](#explaining-parametrization)
- [VISUALS, ADDITIONAL COMMENTS](#visuals-additional-comments)
- [CONCLUSIONS](#conclusions)

# Deep Learning Group Project — Presentation Synopsis

## Abstract

This project implements a concise pipeline for detecting clickbait in Vietnamese text. It was designed to balance reproducibility, interpretability, and fast iteration: lightweight dataset sampling supports rapid experiments while a shard-based training strategy reduces memory pressure on single-machine environments. The goal is to compare two pretrained language model backbones on the same stratified splits and to evaluate their utility for binary clickbait classification.

## Clear Goal

Produce a reliable, reproducible classifier that distinguishes clickbait from non-clickbait in Vietnamese text and quantify model performance using standard classification metrics.

## Objectives

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

## Methodology — Mathematical Formulation

Problem statement: given a headline text $x\\in\\mathcal{X}$, predict label $y\\in\\{0,1\\}$ (0 = non-clickbait, 1 = clickbait). We model the conditional probability $p(y\\mid x;\\theta)$ with a parameterized classifier built on pretrained encoders and a lightweight classification head.

Model output: a two-dimensional logit vector $z(x;\\theta)\\in\\mathbb{R}^2$. We convert logits to probabilities via softmax:

$$
\\hat{p}_i = \\mathrm{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_{j} e^{z_j}}\\quad(i\\in\\{0,1\\})
$$

Training objective: minimize the cross-entropy loss over the training set \\\($\\{(x_k,y_k)\\}_{k=1}^N\\$\\):

$$
\\mathcal{L}(\\theta)= -\\frac{1}{N}\\sum_{k=1}^N \\log \\hat{p}_{y_k}(x_k;\\theta)
$$

Evaluation metrics used throughout:

- Accuracy: $\\text{Acc}=\\frac{TP+TN}{TP+TN+FP+FN}$
- Precision: $\\text{Prec}=\\frac{TP}{TP+FP}$
- Recall: $\\text{Rec}=\\frac{TP}{TP+FN}$
- F1: $F1=2\\cdot\\frac{\\text{Prec}\\cdot\\text{Rec}}{\\text{Prec}+\\text{Rec}}$

These metrics capture complementary aspects: precision prioritizes correctness among positive predictions, recall prioritizes coverage of positives, and F1 balances both.

Model selection and ensembling rationale

- Two backbones were selected: a strong multilingual encoder and a Vietnamese-specialized encoder. Comparing them isolates the effect of pretraining language specificity versus multilingual breadth.
- Shard-based training (splitting training indices deterministically) is used to enable multiple independent runs on limited hardware; final aggregation weights worker metrics by shard size to approximate a single-model evaluation while still keeping per-shard checkpoints.

## Pipeline Design (Conceptual)

- Data ingestion and preprocessing: text normalization, tokenization via backbone tokenizer, and preservation of the label distribution via stratified sampling.
- Tokenization / encoding: use the pretrained tokenizer to map text to token sequences, truncating/padding to a fixed maximum length $L$ for batching.
- Optimization: fine-tune the classification head (and usually the final encoder layers) with standard Adam/variant optimizers and cross-entropy loss.
- Evaluation: run inference on validation and test splits; select the best checkpoint by validation accuracy (or other metric) and report test-set metrics for final comparison.

## Results (Summary)

Key statistical observations from runs (see `results/output.txt` for raw logs):

- The pretrained Vietnamese encoder tended to achieve higher recall on clickbait instances, suggesting greater coverage of language-specific signals.
- The multilingual encoder showed competitive precision, indicating stronger disambiguation when evidence was clear.
- Aggregated F1 and accuracy favored the model that balanced recall and precision for the target operating point; exact numbers and the per-model breakdown are reported in the project `results` directory.

## Conclusions & Recommendations

- For production or moderation scenarios where missing clickbait is costly, favor the model with higher recall and tune the decision threshold to increase sensitivity.
- For analytics or downstream ranking where false positives are costly, favor the model with higher precision or apply a post-classification filtering step.
- The pipeline's `primary_ratio` feature enables rapid prototyping: use small samples for hyperparameter tuning and the full dataset for final evaluation.

## Where to Look Next

- `main.py` orchestrates the end-to-end flow; `_modules/train.py` contains the training logic and model interfacing.
- `results/output.txt` contains recent run logs and numeric outputs used to produce the summary statistics above.

---

If you would like, I can also produce a short slide-style export (Markdown slides) derived from this presentation README for use in talks or reports.
Pipeline output is written through `write_row`, including elapsed time in seconds for each stage.

For the latest runs, the training log also prints the percentage of the training set assigned to each worker, which makes the shard split visible during execution.

---

## CONCLUSIONS

The project now supports a faster experimentation loop by shrinking the sampled dataset, timing the pipeline stages, and splitting the training set into configurable shards.

The current shard approach is a time-optimization measure rather than true distributed training. It reduces memory pressure on Windows and makes the split explicit, but the final wall-clock speed still depends on model size, sequence length, and the number of epochs.