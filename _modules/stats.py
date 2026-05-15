from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from typing import Dict


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute binary classification metrics.

    Returns a dict with `precision`, `recall`, and `f1`.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }