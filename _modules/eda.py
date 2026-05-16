#%%

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from _modules.config import *
from _modules.plots import (
    target_distr_plot, 
    top_N_news_cat, 
    source_distr_plot,
    title_length,
    target_distr_by_source,
    temporal_analysis,
    image_dimension_analysis,
    plot_capitals,
    number_of_words
    )
from _modules.write import write_row

def run_eda(data) -> None:

    ## GLOBAL COLOR PALETTE DESIGN
    # This dictionary maps colors directly to labels across all plot types.
    # Keys are integers that match the encoded label values (0 = non-clickbait, 1 = clickbait)
    palette_dict = {
        0: PASTEL_COLORS["GREEN"],  # non-clickbait
        1: PASTEL_COLORS["RED"]     # clickbait
    }

    ## Printing all columns of the dataset
    write_row(data.columns.tolist())

    ## Missing values
    write_row("Missing Values:")
    write_row(data.isnull().sum())

    #%%

    ## Label distribution - how many clickbait and non clickbait labels
    write_row("\nLabel Distribution:")
    write_row(data['label'].value_counts())

    ## Bar chart for label distribution
    target_distr_plot(data, palette_dict)

    #%%

    ## Source & Category Analysis
    write_row("\nTop 10 Categories")
    write_row(data['category'].value_counts().head(10))

    top_N_news_cat(data, N=10)

    #%%

    write_row("\nNews Sources")
    write_row(data['source'].value_counts())

    source_distr_plot(data)

    #%%

    ## Title Length Analysis (Scaled to Percentages - Vertical)
    write_row("\nTitle Length Analysis ")
    data['title_length'] = data['title'].apply(lambda x: len(str(x)))

    title_length(data, palette_dict)

    #%%

    ## Summary Statistics for Title Length
    write_row("\nSummary Statistics for Title Length:")
    length_stats = data.groupby('label')['title_length'].agg(
        Total_Articles='count',
        Average='mean',
        Median='median',
        Shortest='min',
        Longest='max',
        Std_Dev='std'
    ).round(2)
    write_row(length_stats)

    ## Verify Image Integrity

    data['image_exists'] = data['thumbnail_url'].apply(
        lambda x: os.path.exists(os.path.join(IMAGES_DIR, os.path.basename(str(x))))
    )
    write_row(f"\nImages found locally: {data['image_exists'].sum()} out of {len(data)}")

    ## Identify and print missing image info
    missing_data = data[data['image_exists'] == False].copy()
    write_row(f"\nDetailed Missing Image Report ({len(missing_data)} rows) ")

    if not missing_data.empty:
        only_nan_urls = missing_data[missing_data['thumbnail_url'].isna()]
        actual_missing_files = missing_data[missing_data['thumbnail_url'].notna()]

        write_row(f"Rows with empty URL (NaN): {len(only_nan_urls)}")
        if not only_nan_urls.empty:
            write_row(f"Article IDs with empty URLs: {only_nan_urls['id'].tolist()}")

        write_row(f"\nActual file path mismatches: {len(actual_missing_files)}")
        if not actual_missing_files.empty:
            for idx, row in actual_missing_files.iterrows():
                filename = os.path.basename(str(row['thumbnail_url']))
                write_row(f"ID: {row['id']} | Expected File: {filename}")

    #%%

    ## Clickbait Ratio by Source
    write_row("\nClickbait Proportion by Source ")
    source_label_pct = pd.crosstab(data['source'], data['label'], normalize='index') * 100

    if 'clickbait' in source_label_pct.columns:
        source_label_pct = source_label_pct.sort_values(by='clickbait', ascending=False)

    target_distr_by_source(palette_dict, source_label_pct)

    #%%

    ## Temporal Analysis (Distribution per Label over 24 Hours)
    write_row("\nPublishing Time Analysis (Hourly Distribution per Category)")
    data['publish_datetime'] = pd.to_datetime(data['publish_datetime'], errors='coerce')
    data['publish_hour'] = data['publish_datetime'].dt.hour

    hourly_counts = data.groupby(['publish_hour', 'label']).size().reset_index(name='count')
    hourly_counts['percentage'] = hourly_counts.groupby('label')['count'].transform(lambda x: (x / x.sum()) * 100)

    temporal_analysis(hourly_counts, palette_dict)

    #%%

    ## Image Dimension Analysis
    write_row("\nImage Dimension Analysis: ")
    widths, heights = [], []

    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            img_path = os.path.join(IMAGES_DIR, filename)
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except Exception as e:
                write_row(f"Could not read {filename}: {e}")

    if widths and heights:
        image_dimension_analysis(widths, heights)

        write_row(f"Total Images Analyzed: {len(widths)}")
        write_row(f"Average Width: {sum(widths) / len(widths):.1f}px")
        write_row(f"Average Height: {sum(heights) / len(heights):.1f}px")
    else:
        write_row("No images found in the directory. Please check the 'IMAGES_DIR' path.")

    #%%

    ## Text Feature Analysis: Punctuation and Caps
    write_row("\nText Feature Analysis")
    data['question_marks'] = data['title'].apply(lambda x: str(x).count('?'))
    data['exclamation_marks'] = data['title'].apply(lambda x: str(x).count('!'))
    data['all_caps_words'] = data['title'].apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
    )

    text_features = data.groupby('label')[['question_marks', 'exclamation_marks', 'all_caps_words']].mean()
    write_row("Average count per title:")
    write_row(text_features)

    #%%

    # Plotting the uppercase word usage
    plot_capitals(data, palette_dict)

    #%%
    ## Word Count Distribution
    data['word_count'] = data['title'].apply(lambda x: len(str(x).split()))

    number_of_words(data, palette_dict)
    # %%
