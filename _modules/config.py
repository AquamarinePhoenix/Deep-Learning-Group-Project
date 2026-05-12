DATA_PATH = "data/clickbait_dataset_vietnamese.csv"
SPLIT_DIR = "data/splits"
LABEL_VALUE = {
    "non-clickbait": 0,
    "clickbait": 1
}
PASTEL_COLORS = {
    "BLUE": "#A7C7E7",
    "RED": "#F7B7B2",
    "GREEN": "#B7E4C7"
}
TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.5
# Model selection: primary multilingual and secondary Vietnamese-focused
# Primary: XLM-R (multilingual). Secondary: PhoBERT (Vietnamese-focused).
MODEL_PRIMARY = "xlm-roberta-base"
MODEL_SECONDARY = "vinai/phobert-base"
# Fraction of the full dataset to use as "primary" data before splitting.
# Set to 1.0 to use the entire dataset, or e.g. 0.5 to use half.
PRIMARY_DATA_RATIO = 0.1

# Saved model locations.
TRAINED_MODELS_DIR = "models/trained"
SAVED_MODELS_DIR = "models/saved"

# If True, the training pipeline skips fitting and loads a saved best_model.pth
# checkpoint from the saved-model directory instead.
USE_BEST_MODEL_PTH = False
BEST_MODEL_FILENAME = "best_model.pth"

RESULTS_DIR = "results/"
RESULTS_FILENAME = "output.txt"