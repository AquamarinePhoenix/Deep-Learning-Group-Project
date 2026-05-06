from _modules.config import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


def plot_label_distribution(train_df, val_df, test_df, save_path=None):
    def get_dist(df):
        return df["label"].value_counts(normalize=True).sort_index()

    train_dist = get_dist(train_df)
    val_dist = get_dist(val_df)
    test_dist = get_dist(test_df)

    labels = ["Non-clickbait (0)", "Clickbait (1)"]
    x = np.arange(len(labels))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "train": PASTEL_COLORS["BLUE"],
        "val":   PASTEL_COLORS["RED"],
        "test":  PASTEL_COLORS["GREEN"]
    }

    bars_train = ax.bar(
        x - width,
        train_dist,
        width,
        label="Train",
        color=colors["train"],
        edgecolor="#444",
        linewidth=0.8
    )

    bars_val = ax.bar(
        x,
        val_dist,
        width,
        label="Validation",
        color=colors["val"],
        edgecolor="#444",
        linewidth=0.8
    )

    bars_test = ax.bar(
        x + width,
        test_dist,
        width,
        label="Test",
        color=colors["test"],
        edgecolor="#444",
        linewidth=0.8
    )

    ax.bar_label(bars_train, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars_val, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars_test, fmt="%.2f", padding=3, fontsize=9)

    for bars in [bars_train, bars_val, bars_test]:
        for bar in bars:
            bar.set_path_effects([
                pe.SimplePatchShadow(offset=(1, -1), alpha=0.25),
                pe.Normal()
            ])

    ax.set_title(
        "Label Distribution Across Dataset Splits (Train, Validation, Test)",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Relative Frequency")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)

    ax.legend(frameon=True)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()