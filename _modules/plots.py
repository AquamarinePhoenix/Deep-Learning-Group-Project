from _modules.config import *
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import seaborn as sns
from _modules.models import fit_linear_regression

def target_distr_plot(data, palette_dict):
    plt.figure(figsize=(8, 5), dpi=100)
    sns.set_style("whitegrid")
    
    ax = sns.countplot(data=data, x='label', palette=palette_dict, hue='label', alpha=0.85)
    
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Add shadow effect with semi-transparent bars
    for patch in ax.patches:
        patch.set_alpha(0.85)
    
    # Add labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=11, fontweight='bold')

    ax.set_title('Distribution of Clickbait vs Non-Clickbait', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Remove duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(palette_dict)], ['Non-Clickbait', 'Clickbait'], title='Category', fontsize=10)
    
    # padding at the top so the numbers don't hit the frame
    plt.ylim(0, data['label'].value_counts().max() * 1.1)
    plt.tight_layout()
    plt.show()
    
def top_N_news_cat(data, N=10):
    # Plot Top N Categories with professional styling
    plt.figure(figsize=(12, 6), dpi=100)
    sns.set_style("whitegrid")
    top_categories = data['category'].value_counts().head(N)
    ax1 = sns.barplot(x=top_categories.values, y=top_categories.index, palette='magma', alpha=0.85, hue=top_categories.index)
    
    for patch in ax1.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Add shadow effect
    for patch in ax1.patches:
        patch.set_alpha(0.85)
    
    for container in ax1.containers:
        ax1.bar_label(container, padding=5, fontsize=10, fontweight='bold')

    ax1.set_title(f'Top {N} News Categories', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Number of Articles', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_axisbelow(True)
    if ax1.legend_ is not None:
        ax1.legend_.remove()
    plt.tight_layout()
    plt.show()
    
def source_distr_plot(data):
    # Plot Distribution of News Sources with professional styling
    plt.figure(figsize=(10, 6), dpi=100)
    sns.set_style("whitegrid")
    source_counts = data['source'].value_counts()
    ax2 = sns.barplot(x=source_counts.index, y=source_counts.values, palette='magma', alpha=0.85, hue=source_counts.index)
    
    for patch in ax2.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Add shadow effect
    for patch in ax2.patches:
        patch.set_alpha(0.85)
    
    for container in ax2.containers:
        ax2.bar_label(container, padding=5, fontsize=10, fontweight='bold')

    ax2.set_title('Distribution of News Sources', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Source', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    plt.xticks(rotation=45, ha='right')
    if ax2.legend_ is not None:
        ax2.legend_.remove()
    plt.tight_layout()
    plt.show()
    
def title_length(data, palette_dict):
    plt.figure(figsize=(10, 6), dpi=100)
    sns.set_style("whitegrid")

    ax = sns.histplot(
        data=data,
        x='title_length',
        hue='label',
        stat='percent',
        common_norm=False,
        kde=True,
        element="step",
        palette=palette_dict,
        alpha=0.75,
        line_kws={'linewidth': 2.5},
        edgecolor='black',
        linewidth=1.2
    )
    
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    
    ax.set_title('Percentage Distribution of Title Lengths by Label', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Characters in Title', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Articles within Category (%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Use explicit proxy artists so the legend shows both fills and KDE lines cleanly.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    non_clickbait_color = palette_dict.get('Non-Clickbait', list(palette_dict.values())[0])
    clickbait_color = palette_dict.get('Clickbait', list(palette_dict.values())[-1])

    legend_handles = [
        Patch(facecolor=non_clickbait_color, edgecolor='black', label='Non-Clickbait'),
        Patch(facecolor=clickbait_color, edgecolor='black', label='Clickbait'),
        Line2D([0], [0], color=non_clickbait_color, linewidth=2.5, label='Non-Clickbait KDE'),
        Line2D([0], [0], color=clickbait_color, linewidth=2.5, label='Clickbait KDE'),
    ]
    ax.legend(handles=legend_handles, title='Category', fontsize=9, title_fontsize=10, ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()
    
def target_distr_by_source(palette_dict, source_label_pct):
    plt.figure(figsize=(10, 6), dpi=100)
    sns.set_style("whitegrid")
    # Map dictionary colors sequentially matching columns configuration for manual pandas stacking
    ordered_colors = [palette_dict[col] for col in source_label_pct.columns]
    ax = source_label_pct.plot(kind='bar', stacked=True, color=ordered_colors, alpha=0.85, ax=plt.gca())
    
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Add shadow effect to bars
    for patch in ax.patches:
        patch.set_alpha(0.85)
    
    ax.set_title('Percentage of Clickbait vs Non-Clickbait by Source', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Source', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-Clickbait', 'Clickbait'], title='Category', fontsize=10, title_fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def temporal_analysis(hourly_counts, palette_dict):
    plt.figure(figsize=(12, 6), dpi=100)
    sns.set_style("whitegrid")

    ax = sns.lineplot(
        data=hourly_counts,
        x='publish_hour',
        y='percentage',
        hue='label',
        marker='o',
        markersize=7,
        linewidth=2.5,
        palette=palette_dict,
        alpha=0.9
    )
    ax.set_title('Daily Distribution of Publishing Times (Normalized by Category)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Total Category Articles Published', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-Clickbait', 'Clickbait'], title='Category', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()
    
def image_dimension_analysis(widths, heights):
    plt.figure(figsize=(8, 6), dpi=100)
    sns.set_style("whitegrid")
    # Scatter plot with shadow effect
    ax = sns.scatterplot(x=widths, y=heights, alpha=0.65, s=80, color=PASTEL_COLORS['BLUE'], edgecolor='white', linewidth=0.5)
    ax.set_title('Scatter Plot of Image Dimensions', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Image Width (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Image Height (pixels)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add 1:1 aspect ratio line
    max_dim = max(max(widths), max(heights))
    ax.plot([0, max_dim], [0, max_dim], color=PASTEL_COLORS['RED'], linestyle='--', linewidth=2.5, label='1:1 Aspect Ratio', alpha=0.8)
    
    # Add linear regression line
    slope, intercept, _ = fit_linear_regression(widths, heights)
    regression_x = np.array([min(widths), max(widths)])
    regression_y = slope * regression_x + intercept
    ax.plot(regression_x, regression_y, color=PASTEL_COLORS['GREEN'], linestyle='-', linewidth=2.5, label=f'Linear Fit (y={slope:.2f}x)', alpha=0.8)
    
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()
    
def plot_capitals(data, palette_dict):
    plt.figure(figsize=(8, 5), dpi=100)
    sns.set_style("whitegrid")

    ax = sns.barplot(data=data, x='label', y='all_caps_words', palette=palette_dict, errorbar=None, hue='label', alpha=0.85)
    
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    
    # Add shadow effect
    for patch in ax.patches:
        patch.set_alpha(0.85)
    
    ax.set_title('Average Number of ALL CAPS Words in Titles', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Average ALL CAPS Words', fontsize=12, fontweight='bold')
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Remove duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(palette_dict)], ['Non-Clickbait', 'Clickbait'], title='Category', fontsize=10)
    plt.tight_layout()
    plt.show()
    
def number_of_words(data, palette_dict):
    plt.figure(figsize=(10, 6), dpi=100)
    sns.set_style("whitegrid")
    # Passed our uniform palette dictionary here
    ax = sns.boxplot(data=data, x='label', y='word_count', palette=palette_dict, hue='label', width=0.6)
    
    # Add shadow effect to boxes
    for patch in ax.patches:
        patch.set_alpha(0.85)
    
    ax.set_title('Distribution of Word Count in Titles by Label', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Words', fontsize=12, fontweight='bold')
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Remove duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(palette_dict)], ['Non-Clickbait', 'Clickbait'], title='Category', fontsize=10)
    plt.tight_layout()
    plt.show()

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