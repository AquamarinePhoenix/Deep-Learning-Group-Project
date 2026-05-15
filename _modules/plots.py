from _modules.config import *
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


def plot_label_distribution(train_df, val_df, test_df, save_path=None):
    
    splits = [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]

    counts_non = []
    counts_click = []
    totals = []
    for _name, df in splits:
        total = len(df)
        click = int(df["label"].sum()) if total > 0 else 0
        non = total - click
        totals.append(total)
        counts_non.append(non)
        counts_click.append(click)

    y = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    bars_non = ax.barh(y, counts_non, color=PASTEL_COLORS["BLUE"], edgecolor="#444", linewidth=0.8, label="Non-clickbait")
    bars_click = ax.barh(y, counts_click, left=counts_non, color=PASTEL_COLORS["RED"], edgecolor="#444", linewidth=0.8, label="Clickbait")

    for i in range(len(splits)):
        total = totals[i]
        if total == 0:
            continue
        non = counts_non[i]
        click = counts_click[i]

        def annotate_segment(val, left, ypos, color):
            pct = val / total * 100 if total else 0
            x = left + val / 2
            # Dynamically scale font size based on segment width
            font_size = max(6, min(8, val / total * 12))
            if val >= max(1, total * 0.05):
                ax.text(x, ypos, f"{val}", ha="center", va="center", color="white", fontsize=font_size)
            else:
                ax.text(left + val + max(1, total * 0.01), ypos, f"{val}", ha="left", va="center", color="#222", fontsize=6)

        annotate_segment(non, 0, y[i], PASTEL_COLORS["BLUE"])
        annotate_segment(click, non, y[i], PASTEL_COLORS["RED"])

    for bars in (bars_non, bars_click):
        for bar in bars:
            bar.set_path_effects([
                pe.SimplePatchShadow(offset=(1, -1), alpha=0.25),
                pe.Normal()
            ])

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in splits])
    ax.set_xlabel("Number of examples")
    ax.set_title("Dataset split sizes and class composition")
    ax.legend(frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "label_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close()