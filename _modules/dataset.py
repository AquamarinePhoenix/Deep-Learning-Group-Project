import pandas as pd
from sklearn.model_selection import train_test_split
import os
from _modules.config import *

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].map(LABEL_VALUE)

    df["title"] = df["title"].fillna("")
    df["lead_paragraph"] = df["lead_paragraph"].fillna("")

    df["text"] = df["title"] + " [SEP] " + df["lead_paragraph"]

    return df


def split_data(df: pd.DataFrame):
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


def save_splits(train_df, val_df, test_df):
    os.makedirs(SPLIT_DIR, exist_ok=True)

    cols = [TEXT_COL, LABEL_COL]

    train_df[cols].to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    val_df[cols].to_csv(f"{SPLIT_DIR}/val.csv", index=False)
    test_df[cols].to_csv(f"{SPLIT_DIR}/test.csv", index=False)