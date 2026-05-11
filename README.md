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

## INTRO

---

## MAIN GOAL AND OBJECTIVES TO REACH THE GOAL

Build a Vietnamese clickbait detection pipeline.
Evaluate two base models on the same dataset split and compare their metrics.

---

## DATASET

- Link to the dataset: ([Vietnamese clickbait dataset](https://data.mendeley.com/datasets/3wc46bfcjc/1)).

Data project structure:

```
project/
│
├── data/
│   ├── clickbait_dataset_vietnamese.csv
│   ├── clickbait_dataset_vietnamese.jsonl
│   └── splits/
│
├── _modules/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   └── plots.py
│   ├── train.py
│   └── write.py
│
├── main.py
├── models/
│   └── trained/
└── results/
```

> [!IMPORTANT]
> Target variable is imbalanced (70/30) for non-clickbait and clickbait, so stratified splitting is required.

The split step now supports `primary_ratio`, which lets you train on a fraction of the full dataset before creating train / validation / test splits.

---

## RECENT CHANGES

- Added two model presets in `_modules/config.py`: `MODEL_PRIMARY = "xlm-roberta-base"` and `MODEL_SECONDARY = "vinai/phobert-base"`.
- Added `PRIMARY_DATA_RATIO` so the main pipeline can sample a smaller training subset for faster experimentation.
- Added `RESULTS_DIR` and `RESULTS_FILENAME` so pipeline output is written to `results/output.txt`.
- Added `write_row(..., elapsed_seconds=None)` in `_modules/write.py` so timing information can be appended directly to the output log.
- Added shard-based training in `_modules/train.py` using `num_workers` to split the training indices into equally sized parts by default (`4` workers).
- Added per-worker shard reporting so the console shows how many rows and what percentage of the training set each worker receives.
- Added training-time logging in `main.py` for dataset load, preprocessing, splitting, saving, plotting, each model run, and total pipeline runtime.
- Added model-load inspection in `_modules/models.py` so missing classifier-head weights are reported as expected when loading a base checkpoint for sequence classification.
- Added `USE_BEST_MODEL_PTH` and `BEST_MODEL_FILENAME` in `_modules/config.py` so the pipeline can skip retraining and reuse a saved `best_model.pth` checkpoint from `models/saved/` when it already exists.

The missing classifier parameters reported for `xlm-roberta-base` and `vinai/phobert-base` are expected when the pretrained checkpoint does not already include a task-specific classification head:

- `classifier.out_proj.weight`
- `classifier.out_proj.bias`
- `classifier.dense.weight`
- `classifier.dense.bias`

Those keys are initialized for downstream training and do not indicate a broken load by themselves.

The repository now also ignores generated `models/` and `results/` outputs in `.gitignore`.

---

## PARALLEL TRAINING

The training code uses shard-based worker execution rather than true multi-process distributed training. The goal is to make the training set easier to manage, reduce memory pressure, and keep the worker assignment easy to follow in the logs.

How it works:

1. The main pipeline loads and preprocesses the dataset once.
2. The train / validation / test split is created once.
3. The tokenized training set is split into `num_workers` shards.
4. The default worker count is `4` unless you override it with `--num_workers`.
5. The shard boundaries are created with `numpy.array_split`, which spreads the row indices as evenly as possible across the requested number of workers.
6. Each worker receives a list of training indices rather than a new copy of the full dataset.
7. Each worker trains on its own subset, then the script collects its metrics and output checkpoint.

Worker assignment details:

- If the training set has 27 rows and `num_workers = 4`, the split becomes 7, 7, 7, and 6 rows.
- If the row count is not divisible by the worker count, the extra rows are distributed across the first shards by `numpy.array_split`.
- Each worker gets a contiguous chunk of the index array after splitting, which keeps the split deterministic for the same seed and dataset order.
- The console prints the number of examples and the percentage of the training set assigned to each worker so the split is visible during execution.

Execution behavior:

- The implementation is sequential shard training on this Windows environment, not simultaneous GPU/CPU distributed training.
- Each worker still loads its own model copy, trains on its shard, and writes its own checkpoint folder.
- Each training run also copies a `best_model.pth` file into `models/saved/<model_name>/`, and the config flag `USE_BEST_MODEL_PTH` can skip training and load that checkpoint directly.
- For speed, shard workers skip epoch-by-epoch evaluation and checkpoint selection.
- The pipeline then aggregates worker metrics by weighting each worker by its shard size.
- The best worker checkpoint is copied back into the final output folder after all shards finish.

Expected runtime implications:

- This approach improves clarity and can reduce memory spikes compared with launching several full training processes at once.
- It is still useful for experimentation, but it is not the same as a synchronized parallel optimizer.
- The biggest time costs remain model size, tokenization length, epochs, and the fact that each shard trains a separate model instance.

## EDA

---

## METHODOLOGY/APPROACH TO A PROBLEM

1. Load the CSV dataset.
2. Preprocess text and labels.
3. Split the dataset with stratification and optional `primary_ratio` sampling.
4. Save the train / validation / test CSV files.
5. Plot label distribution.
6. Train the selected base models.
7. Compare metrics across models.

Training details:

- `main.py` runs both model presets in sequence.
- `num_workers` controls how many training shards are created from the training split.
- Each shard uses a subset of the training indices, and the script logs the shard size plus its percentage of the training set.
- For speed, shard workers skip epoch-by-epoch evaluation and saving; the full pipeline still evaluates the final model metrics.
- The train-set shard split is deterministic for a fixed dataset order because the code uses `numpy.array_split` on the index array.

---

## EXPLAINING PARAMETRIZATION

Current main parameters in `_modules/config.py`:

- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2`
- `VAL_SIZE = 0.5`
- `PRIMARY_DATA_RATIO = 0.01`
- `MODEL_PRIMARY = "xlm-roberta-base"`
- `MODEL_SECONDARY = "vinai/phobert-base"`
- `USE_BEST_MODEL_PTH = False`
- `BEST_MODEL_FILENAME = "best_model.pth"`

CLI parameters in `main.py`:

- `--primary_ratio`: fraction of the dataset to keep before splitting.
- `--num_workers`: number of training shards to create from the training set.

CLI parameters in `_modules/train.py`:

- `--model_key`: model to train.
- `--output_dir`: destination folder for model artifacts.
- `--epochs`: number of epochs.
- `--batch_size`: per-device batch size.
- `--max_length`: tokenization max length.
- `--seed`: random seed.
- `--num_workers`: shard count for training.

---

## VISUALS, ADDITIONAL COMMENTS

Pipeline output is written through `write_row`, including elapsed time in seconds for each stage.

For the latest runs, the training log also prints the percentage of the training set assigned to each worker, which makes the shard split visible during execution.

---

## CONCLUSIONS

The project now supports a faster experimentation loop by shrinking the sampled dataset, timing the pipeline stages, and splitting the training set into configurable shards.

The current shard approach is a time-optimization measure rather than true distributed training. It reduces memory pressure on Windows and makes the split explicit, but the final wall-clock speed still depends on model size, sequence length, and the number of epochs.