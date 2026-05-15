"""
Classical baselines for Vietnamese clickbait detection.

This module provides a TF-IDF + Logistic Regression baseline that reuses
the same preprocessing and split logic as the transformer pipeline.

Example:
    python -m _modules.classical --primary_ratio 1.0
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _modules.config import (
    CLASSICAL_MODELS_DIR,
    LABEL_COL,
    LOGREG_MODEL_FILENAME,
    LOGREG_TOP_FEATURES_FILENAME,
    PRIMARY_DATA_RATIO,
    RANDOM_STATE,
    TEXT_COL,
    SPLIT_DIR,
)
from _modules.dataset import load_data, preprocess_data, save_splits, split_data
from _modules.write import write_row
from _modules.stats import compute_metrics

def _save_top_features(model: Pipeline, output_dir: str, top_k: int = 25) -> None:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    top_positive_idx = coefficients.argsort()[-top_k:][::-1]
    top_negative_idx = coefficients.argsort()[:top_k]

    feature_df = pd.DataFrame(
        {
            "feature": list(feature_names[top_positive_idx]) + list(feature_names[top_negative_idx]),
            "coefficient": list(coefficients[top_positive_idx]) + list(coefficients[top_negative_idx]),
            "direction": (["clickbait"] * top_k) + (["non-clickbait"] * top_k),
        }
    )
    feature_df.to_csv(os.path.join(output_dir, LOGREG_TOP_FEATURES_FILENAME), index=False)


def train_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, # is it needed?
    test_df: pd.DataFrame,
    output_dir: str = CLASSICAL_MODELS_DIR,
    max_features: int = 50000, # check how many we have now
) -> Dict[str, Dict[str, float]]:
    os.makedirs(output_dir, exist_ok=True)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2), # look through captions if it is enough
                    max_features=max_features,
                    min_df=2, # check if it is high enough
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    write_row("Fitting TF-IDF + Logistic Regression baseline...", source="logreg")
    fit_start = time.perf_counter()
    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
    write_row("Finished fitting logistic regression baseline", time.perf_counter() - fit_start, source="logreg")

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    split_frames = {
        "validation": val_df,
        "test": test_df,
    }

    for split_name, split_df in split_frames.items():
        predictions = model.predict(split_df[TEXT_COL])
        metrics = compute_metrics(split_df[LABEL_COL], predictions)
        metrics_by_split[split_name] = metrics

        write_row(f"{split_name.title()} metrics", source="logreg")
        write_row(f"Precision : {metrics['precision']:.4f}", source="logreg")
        write_row(f"Recall    : {metrics['recall']:.4f}", source="logreg")
        write_row(f"F1-Score  : {metrics['f1']:.4f}", source="logreg")

    model_path = os.path.join(output_dir, LOGREG_MODEL_FILENAME)
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    _save_top_features(model, output_dir)
    write_row(f"Saved logistic regression model to {model_path}", source="logreg")
    return metrics_by_split


def run_logistic_regression_pipeline(
    primary_ratio: float = PRIMARY_DATA_RATIO,
    output_dir: str = CLASSICAL_MODELS_DIR,
) -> Dict[str, Dict[str, float]]:
    # If splits already exist in `SPLIT_DIR`, load them and use directly.
    train_csv = os.path.join(SPLIT_DIR, "train.csv")
    val_csv = os.path.join(SPLIT_DIR, "val.csv")
    test_csv = os.path.join(SPLIT_DIR, "test.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        write_row("Loading dataset splits from disk...", source="logreg")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    else:
        write_row("Loading dataset...", source="logreg")
        df = load_data()

        write_row("Preprocessing dataset...", source="logreg")
        df = preprocess_data(df)

        write_row("Splitting dataset...", source="logreg")
        train_df, val_df, test_df = split_data(df, primary_ratio=primary_ratio)

        write_row("Saving dataset splits...", source="logreg")
        save_splits(train_df, val_df, test_df)

    return train_logistic_regression(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline")
    parser.add_argument(
        "--primary_ratio",
        type=float,
        default=PRIMARY_DATA_RATIO,
        help="Fraction of the dataset to use before splitting",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CLASSICAL_MODELS_DIR,
        help="Directory used to save the fitted baseline model",
    )
    args, _unknown = parser.parse_known_args()

    run_logistic_regression_pipeline(
        primary_ratio=args.primary_ratio,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
