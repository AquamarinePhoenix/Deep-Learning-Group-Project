from __future__ import annotations

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from _modules.config import *


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].map(LABEL_VALUE)

    df["title"] = df["title"].fillna("")
    df["lead_paragraph"] = df["lead_paragraph"].fillna("")

    df["text"] = df["title"] + " [SEP] " + df["lead_paragraph"]

    return df


def split_data(df: pd.DataFrame, primary_ratio: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optionally sample a fraction of the dataframe as primary data,
    then split into train / val / test.

    primary_ratio: fraction of rows to keep before splitting (0 < r <= 1.0)
    """
    if primary_ratio <= 0 or primary_ratio > 1.0:
        raise ValueError("primary_ratio must be in (0, 1].")

    if primary_ratio < 1.0:
        df = df.sample(frac=primary_ratio, random_state=RANDOM_STATE)

    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(SPLIT_DIR, exist_ok=True)

    cols = [TEXT_COL, LABEL_COL]

    train_df[cols].to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    val_df[cols].to_csv(f"{SPLIT_DIR}/val.csv", index=False)
    test_df[cols].to_csv(f"{SPLIT_DIR}/test.csv", index=False)
