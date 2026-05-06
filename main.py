#%%

from _modules.dataset import load_data, preprocess_data, split_data, save_splits
from _modules.plots import plot_label_distribution

df = load_data()
df = preprocess_data(df)

train_df, val_df, test_df = split_data(df)

save_splits(train_df, val_df, test_df)

plot_label_distribution(train_df, val_df, test_df)

# %%
