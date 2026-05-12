#%%

import argparse
import time
from datetime import datetime
from typing import Dict

from _modules.config import (
    PRIMARY_DATA_RATIO,
    MODEL_PRIMARY,
    MODEL_SECONDARY
)

from _modules.dataset import (
    load_data,
    preprocess_data,
    split_data,
    save_splits
)

from _modules.plots import plot_label_distribution
from _modules.train import train_model
from _modules.write import *


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Vietnamese Clickbait Detection Pipeline"
    )

    parser.add_argument(
        "--primary_ratio",
        type=float,
        default=PRIMARY_DATA_RATIO,
        help="Fraction of the dataset to use before splitting"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of parallel training shards to use"
    )

    # Use parse_known_args to ignore extra args injected by interactive
    # environments (e.g. Jupyter / ipykernel passes `--f=...`).
    args, _unknown = parser.parse_known_args()

    overall_start = time.perf_counter()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_row("\n" + "=" * 60, source="main")
    write_row(f"Run started at: {run_timestamp}", source="main")
    write_row("=" * 60, source="main")
    
    write_row("\nLoading dataset...", source="main")
    load_start = time.perf_counter()
    df = load_data()
    write_row("Loaded dataset", time.perf_counter() - load_start, source="main")

    write_row("Preprocessing dataset...", source="main")
    preprocess_start = time.perf_counter()
    df = preprocess_data(df)
    write_row("Preprocessed dataset", time.perf_counter() - preprocess_start, source="main")

    write_row("Splitting dataset...", source="main")
    split_start = time.perf_counter()
    train_df, val_df, test_df = split_data(
        df,
        primary_ratio=args.primary_ratio
    )
    write_row("Split dataset", time.perf_counter() - split_start, source="main")

    write_row("Saving dataset splits...", source="main")
    save_start = time.perf_counter()
    save_splits(train_df, val_df, test_df)
    write_row("Saved dataset splits", time.perf_counter() - save_start, source="main")

    write_row("Generating label distribution plot...", source="main")
    plot_start = time.perf_counter()
    plot_label_distribution(train_df, val_df, test_df)
    write_row("Generated label distribution plot", time.perf_counter() - plot_start, source="main")

    models = [MODEL_PRIMARY, MODEL_SECONDARY]

    results: Dict[str, Dict[str, float]] = {}

    for model_name in models:

        output_dir = f"models/trained/{model_name.replace('/', '_')}"

        write_row("\n" + "=" * 60, source="main")
        write_row(f"Training model: {model_name}", source="main")
        write_row(f"Output directory: {output_dir}", source="main")
        write_row("=" * 60, source="main")

        train_start = time.perf_counter()
        metrics = train_model(
            model_key=model_name,
            output_dir=output_dir,
            num_workers=args.num_workers
        )
        write_row(f"Finished training model: {model_name}", time.perf_counter() - train_start, source="main")

        results[model_name] = metrics

    write_row("\nFinal Comparison Results", source="main")
    write_row("=" * 60, source="main")

    for model_name, metrics in results.items():

        write_row(f"\nModel: {model_name}", source="main")

        write_row(f"Precision : {metrics['precision']:.4f}", source="main")
        write_row(f"Recall    : {metrics['recall']:.4f}", source="main")
        write_row(f"F1-Score  : {metrics['f1']:.4f}", source="main")
        write_row(f"Accuracy  : {metrics['accuracy']:.4f}", source="main")

    write_row("\nTotal pipeline runtime", time.perf_counter() - overall_start, source="main")


if __name__ == "__main__":
    main()

# %%