"""
Training utilities moved into the `_modules` package.

This module exposes a `train()` CLI and `train_model()` function that
train a sequence classification model using the CSV splits located in
`data/splits/train.csv` and `data/splits/test.csv`.

Call from the repository root:
    python -m _modules.train --model_key xlm-r --epochs 3
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from inspect import signature
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed
from _modules.stats import compute_metrics

from _modules.config import (
    BATCH_SIZE,
    BEST_MODEL_FILENAME,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_PRIMARY,
    SAVED_MODELS_DIR,
    SPLIT_DIR,
    TEXT_COL,
    TRAINED_MODELS_DIR,
    USE_BEST_MODEL_PTH,
    WARMUP_RATIO,
)
from _modules.models import load_sequence_classification_model, load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _trainer_compute_metrics(eval_pred) -> Dict[str, float]:
    """Wrapper for Trainer which receives (logits, labels)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(labels, preds)


def _source_label(worker_id: int | None = None) -> str:
    return "main" if worker_id is None else f"worker {worker_id}"


def _log_info(message: str, *args, worker_id: int | None = None) -> None:
    logger.info("[%s] " + message, _source_label(worker_id), *args)


def _log_debug(message: str, *args, worker_id: int | None = None) -> None:
    logger.debug("[%s] " + message, _source_label(worker_id), *args)


def _log_error(message: str, *args, worker_id: int | None = None) -> None:
    logger.error("[%s] " + message, _source_label(worker_id), *args)

class TrainSubsetMetricsCallback(TrainerCallback):
    def __init__(self, train_dataset: object, worker_id: int | None = None) -> None:
        self.train_dataset = train_dataset
        self.worker_id = worker_id
        self.trainer: Trainer | None = None
        self.latest_loss: float | None = None

    def set_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.latest_loss = float(logs["loss"])
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or self.train_dataset is None:
            return control

        predictions = self.trainer.predict(self.train_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        metrics = compute_metrics(predictions.label_ids, preds)
        epoch_number = int(round(state.epoch)) if state.epoch is not None else int(state.global_step)
        loss_text = f"loss={self.latest_loss:.4f}, " if self.latest_loss is not None else ""
        _log_info(
            "Epoch %s train subset -> %sprecision=%.4f, recall=%.4f, f1=%.4f",
            epoch_number,
            loss_text,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            
            worker_id=self.worker_id,
        )
        return control


def _load_tokenized_datasets(
    model_key: str,
    max_length: int,
    train_indices: List[int] | None = None,
    worker_id: int | None = None,
) -> tuple[object, object]:
    train_csv = os.path.join(SPLIT_DIR, "train.csv")
    val_csv = os.path.join(SPLIT_DIR, "val.csv")
    test_csv = os.path.join(SPLIT_DIR, "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        _log_error("Expected CSV splits not found in %s. Run data split step first.", SPLIT_DIR, worker_id=worker_id)
        raise FileNotFoundError("CSV splits not found")

    _log_info("Loading datasets from CSV", worker_id=worker_id)
    data_files = {"train": train_csv, "validation": val_csv, "test": test_csv}
    raw_datasets = load_dataset("csv", data_files=data_files)

    if train_indices is not None:
        raw_datasets["train"] = raw_datasets["train"].select(train_indices)

    _log_info("Loading tokenizer for %s", model_key, worker_id=worker_id)
    tokenizer = load_tokenizer(model_key)

    def preprocess_function(examples):
        texts = examples[TEXT_COL]
        return tokenizer(texts, truncation=True, padding=False, max_length=max_length)

    tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=[TEXT_COL])
    return tokenized, tokenizer


def _build_training_args(
    output_dir: str,
    batch_size: int,
    epochs: int,
    evaluate_during_training: bool,
) -> TrainingArguments:
    ta_kwargs = {
        "output_dir": output_dir,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "warmup_ratio": WARMUP_RATIO,
        "logging_dir": "logs",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "dataloader_num_workers": max(1, min(2, os.cpu_count() or 1)),
    }

    if evaluate_during_training:
        ta_kwargs["evaluation_strategy"] = "epoch"
        ta_kwargs["save_strategy"] = "epoch"
        ta_kwargs["load_best_model_at_end"] = True
        ta_kwargs["metric_for_best_model"] = "f1"
    else:
        ta_kwargs["evaluation_strategy"] = "no"
        ta_kwargs["save_strategy"] = "no"
        ta_kwargs["load_best_model_at_end"] = False

    sig = signature(TrainingArguments.__init__)
    valid_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig.parameters and k != "self"}
    if len(valid_kwargs) != len(ta_kwargs):
        _log_info(
            "Filtered TrainingArguments params for compatibility: %s",
            set(ta_kwargs.keys()) - set(valid_kwargs.keys()),
        )

    if "evaluation_strategy" not in valid_kwargs:
        if evaluate_during_training:
            if "save_strategy" in sig.parameters:
                valid_kwargs["save_strategy"] = "best"
            valid_kwargs["metric_for_best_model"] = "f1"
        else:
            if "save_strategy" in sig.parameters:
                valid_kwargs["save_strategy"] = "no"
            valid_kwargs.pop("load_best_model_at_end", None)
            valid_kwargs.pop("metric_for_best_model", None)

    return TrainingArguments(**valid_kwargs)


def _saved_model_dir(model_key: str) -> str:
    return os.path.join(SAVED_MODELS_DIR, model_key.replace('/', '_'))


def _copy_best_model_pth(source_dir: str, model_key: str) -> str:
    saved_dir = _saved_model_dir(model_key)
    os.makedirs(saved_dir, exist_ok=True)
    source_path = os.path.join(source_dir, BEST_MODEL_FILENAME)
    destination_path = os.path.join(saved_dir, BEST_MODEL_FILENAME)
    if not os.path.exists(source_path):
        _log_info("Best model file not found at %s; skipping copy to saved models", source_path)
        return destination_path
    shutil.copy2(source_path, destination_path)
    _log_info("Copied best model %s -> %s", source_path, destination_path)
    return destination_path


def _load_best_model_state(model: object, model_key: str) -> None:
    best_model_path = os.path.join(_saved_model_dir(model_key), BEST_MODEL_FILENAME)
    state_dict = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(state_dict)


def _evaluate_saved_best_model(
    tokenized: object,
    tokenizer: object,
    model_key: str,
    batch_size: int,
    worker_id: int | None = None,
) -> Dict[str, float]:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    best_model_path = os.path.join(_saved_model_dir(model_key), BEST_MODEL_FILENAME)
    _log_info("Loading saved best model from %s", best_model_path, worker_id=worker_id)

    model = load_sequence_classification_model(model_key, num_labels=2, worker_id=worker_id)
    _load_best_model_state(model, model_key)

    training_args = TrainingArguments(
        output_dir=_saved_model_dir(model_key),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=max(1, min(2, os.cpu_count() or 1)),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": _trainer_compute_metrics,
    }

    sig_trainer = signature(Trainer.__init__)
    if "tokenizer" in sig_trainer.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    pred = trainer.predict(tokenized["test"])
    return compute_metrics(pred.label_ids, np.argmax(pred.predictions, axis=-1))


def _train_tokenized_model(
    tokenized: object,
    tokenizer: object,
    model_key: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    evaluate_during_training: bool = True,
    worker_id: int | None = None,
) -> Dict[str, float]:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    _log_info("Loading model for sequence classification: %s", model_key, worker_id=worker_id)
    model = load_sequence_classification_model(model_key, num_labels=2, worker_id=worker_id)

    training_args = _build_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        evaluate_during_training=evaluate_during_training,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": _trainer_compute_metrics,
    }

    sig_trainer = signature(Trainer.__init__)
    if "tokenizer" in sig_trainer.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        _log_info("Trainer.__init__ does not accept 'tokenizer'; attaching after init.", worker_id=worker_id)

    trainer = Trainer(**trainer_kwargs)
    if not hasattr(trainer, "tokenizer") or trainer.tokenizer is None:
        try:
            setattr(trainer, "tokenizer", tokenizer)
        except Exception:
            _log_debug("Could not attach tokenizer to Trainer instance; saving tokenizer separately.", worker_id=worker_id)

    epoch_metrics_callback = TrainSubsetMetricsCallback(
        train_dataset=tokenized["train"],
        worker_id=worker_id,
    )
    epoch_metrics_callback.set_trainer(trainer)
    trainer.add_callback(epoch_metrics_callback)

    _log_info("Starting training", worker_id=worker_id)
    trainer.train()
    pred = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics(pred.label_ids, np.argmax(pred.predictions, axis=-1))

    _log_info("Saving model and tokenizer to %s", output_dir, worker_id=worker_id)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save a lightweight state_dict as the best-model checkpoint inside the run output
    try:
        best_pth = os.path.join(output_dir, BEST_MODEL_FILENAME)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(trainer.model.state_dict(), best_pth)
        _log_info("Saved best model state to %s", best_pth, worker_id=worker_id)
    except Exception as _err:
        _log_info("Could not save best model state to %s: %s", output_dir, str(_err), worker_id=worker_id)

    return test_metrics


def _train_worker(
    worker_id: int,
    train_indices: List[int],
    tokenized: object,
    tokenizer: object,
    model_key: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    set_seed(seed + worker_id)
    worker_output_dir = f"{output_dir}_worker_{worker_id}"
    worker_tokenized = tokenized.copy()
    worker_tokenized["train"] = worker_tokenized["train"].select(train_indices)
    metrics = _train_tokenized_model(
        tokenized=worker_tokenized,
        tokenizer=tokenizer,
        model_key=model_key,
        output_dir=worker_output_dir,
        epochs=epochs,
        batch_size=batch_size,
        evaluate_during_training=False,
        worker_id=worker_id,
    )
    return {
        "worker_id": worker_id,
        "train_size": len(train_indices),
        "output_dir": worker_output_dir,
        "metrics": metrics,
    }


def _aggregate_worker_metrics(worker_results: List[Dict[str, object]]) -> Dict[str, float]:
    if not worker_results:
        raise ValueError("No worker results to aggregate")

    total_size = sum(int(result["train_size"]) for result in worker_results)
    if total_size <= 0:
        raise ValueError("Worker training shards are empty")

    aggregated: Dict[str, float] = {}
    for metric_name in ("precision", "recall", "f1"):
        aggregated[metric_name] = sum(
            float(result["metrics"][metric_name]) * int(result["train_size"])
            for result in worker_results
        ) / total_size

    aggregated["workers"] = float(len(worker_results))
    aggregated["train_size"] = float(total_size)
    return aggregated


def train_model(
    model_key: str = MODEL_PRIMARY,
    output_dir: str = TRAINED_MODELS_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    seed: int = 42,
    num_workers: int = 4,
) -> Dict[str, float]:
    set_seed(seed)
    tokenized, tokenizer = _load_tokenized_datasets(model_key=model_key, max_length=max_length, worker_id=None)

    best_model_path = os.path.join(_saved_model_dir(model_key), BEST_MODEL_FILENAME)
    if USE_BEST_MODEL_PTH and os.path.exists(best_model_path):
        _log_info("Loading saved best model from %s and skipping training", best_model_path)
        return _evaluate_saved_best_model(
            tokenized=tokenized,
            tokenizer=tokenizer,
            model_key=model_key,
            batch_size=batch_size,
            worker_id=None,
        )

    if num_workers <= 1:
        metrics = _train_tokenized_model(
            tokenized=tokenized,
            tokenizer=tokenizer,
            model_key=model_key,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            worker_id=None,
        )
        _copy_best_model_pth(output_dir, model_key)
        return metrics

    train_size = len(tokenized["train"])
    worker_count = min(num_workers, train_size)
    if worker_count <= 1:
        metrics = _train_tokenized_model(
            tokenized=tokenized,
            tokenizer=tokenizer,
            model_key=model_key,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            worker_id=None,
        )
        _copy_best_model_pth(output_dir, model_key)
        return metrics

    shard_indices = np.array_split(np.arange(train_size), worker_count)
    worker_jobs = [
        (worker_id, shard.astype(int).tolist())
        for worker_id, shard in enumerate(shard_indices)
        if len(shard) > 0
    ]

    _log_info("Starting shard-based training with %s workers and %s training examples", len(worker_jobs), train_size)
    for worker_id, train_indices in worker_jobs:
        shard_size = len(train_indices)
        shard_pct = (shard_size / train_size) * 100 if train_size else 0.0
        _log_info(
            "Worker %s receives %s examples (%.1f%% of the training set)",
            worker_id,
            shard_size,
            shard_pct,
            worker_id=None,
        )

    worker_results: List[Dict[str, object]] = []
    _log_info("Launching %s workers in parallel", len(worker_jobs))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _train_worker,
                worker_id,
                train_indices,
                tokenized,
                tokenizer,
                model_key,
                output_dir,
                epochs,
                batch_size,
                seed,
            )
            for worker_id, train_indices in worker_jobs
        ]
        for future in futures:
            worker_results.append(future.result())
    gc.collect()

    worker_results.sort(key=lambda result: int(result["worker_id"]))
    aggregated_metrics = _aggregate_worker_metrics(worker_results)

    best_worker = max(worker_results, key=lambda result: float(result["metrics"]["f1"]))
    best_output_dir = str(best_worker["output_dir"])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(best_output_dir, output_dir)

    _log_info("Shard worker metrics: %s", worker_results)
    _log_info("Aggregated metrics: %s", aggregated_metrics)
    _log_info("Copied best worker checkpoint from %s to %s", best_output_dir, output_dir)
    _copy_best_model_pth(best_output_dir, model_key)
    return aggregated_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default=MODEL_PRIMARY, help="Model key or HF model id to use")
    parser.add_argument("--output_dir", type=str, default="models/trained", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel training shards to use")
    # Use parse_known_args so kernel-injected args (from Jupyter) are ignored
    args, _unknown = parser.parse_known_args()

    train_model(
        model_key=args.model_key,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
