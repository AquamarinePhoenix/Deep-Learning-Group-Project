"""
Model loader helpers for the pipeline.

Provides simple functions to load tokenizers and models for:
- XLM-R (multilingual) - `xlm-roberta-base`
- PhoBERT (Vietnamese) - `vinai/phobert-base`

These helpers use Hugging Face `transformers`. They keep names flexible
so callers can pass short keys like "xlm-r" or "phobert".
"""
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Mapping of short keys to HF model IDs
MODEL_MAP: dict[str, str] = {
    "xlm-r": "xlm-roberta-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "phobert": "vinai/phobert-base",
    "vinai/phobert-base": "vinai/phobert-base",
}


def _resolve_model_name(key: str) -> str:
    return MODEL_MAP.get(key, key)


def _source_label(worker_id: int | None = None) -> str:
    return "main" if worker_id is None else f"worker {worker_id}"


def load_tokenizer(model_key: str, **kwargs) -> AutoTokenizer:
    """Load and return a tokenizer for `model_key`.

    Example keys: "xlm-r", "phobert", or full HF model ids.
    """
    name = _resolve_model_name(model_key)
    return AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)


def load_model_base(model_key: str, **kwargs) -> AutoModel:
    """Load and return the base (encoder) model for `model_key`."""
    name = _resolve_model_name(model_key)
    return AutoModel.from_pretrained(name, **kwargs)


def load_sequence_classification_model(
    model_key: str,
    num_labels: int = 2,
    worker_id: int | None = None,
    **kwargs,
) -> AutoModelForSequenceClassification:
    """Load a `AutoModelForSequenceClassification` for `model_key`.

    If the pretrained model doesn't include a classification head, a new
    randomly initialized head will be created with `num_labels`.
    """
    name = _resolve_model_name(model_key)
    try:
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=num_labels,
            output_loading_info=True,
            **kwargs,
        )
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels, **kwargs)

    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    expected_classifier_keys = {
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
        "classifier.dense.weight",
        "classifier.dense.bias",
    }

    if missing_keys:
        missing_set = set(missing_keys)
        if missing_set.issubset(expected_classifier_keys):
            print(
                f"[{_source_label(worker_id)}] Model loaded with a newly initialized classification head; missing keys: "
                + ", ".join(sorted(missing_set))
            )
        else:
            print(f"[{_source_label(worker_id)}] Model loaded with missing keys: " + ", ".join(sorted(missing_set)))

    if unexpected_keys:
        print(f"[{_source_label(worker_id)}] Model loaded with unexpected keys: " + ", ".join(sorted(unexpected_keys)))

    return model


def suggested_model_order() -> list[str]:
    """Return ordered list of model keys preferred by the pipeline.

    First is the primary multilingual model, second is the Vietnamese-focused one.
    """
    return ["xlm-r", "phobert"]


def fit_linear_regression(x, y):
    """Fit a linear regression model forced through origin (no intercept)."""
    
    import numpy as np
    
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    
    # Fit slope only (no intercept)
    slope = np.sum(x_arr * y_arr) / np.sum(x_arr * x_arr)
    
    # Predictions
    predicted_y = slope * x_arr
    
    return slope, 0.0, predicted_y
