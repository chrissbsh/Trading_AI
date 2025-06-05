import numpy as np
import random
import tensorflow as tf

# ───────────────────────── GLOBAL SETTINGS ────────────────────────── #
SEED = 42
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional: for full reproducibility (slower)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Call set_seeds at import time or explicitly in pipeline
set_seeds()
tf.keras.backend.set_floatx("float32")

# ───────────────────────── DATA PARAMETERS ─────────────────────── #
DATA_FILE_PATH = "csv_data/consolidated_data/normalized_complete_data.csv"
DATE_COL = "Date"
TARGET_PRICE_COL = "SP500_historical_data_Close" # Column used for future return calculation
EXCLUDE_COLS_FROM_FEATURES = {"Date", "ret_future", "target"} # Will be updated after feature selection

# ──────────────────── FEATURE SELECTION PARAMETERS ───────────────── #
TOP_N_FEATURES = 25

# ─────────────────── PREPROCESSING PARAMETERS ──────────────────── #
PRED_HORIZON = 7  # jours
N_CLASSES = 3
TIMESTEPS = 90  # taille séquence initial = 30
JITTER_RATIO = 0.05  # % du σ pour le jitter
THRESHOLD_STEP_MONTHS = 6 # For build_sliding_thresholds
THRESHOLD_STRATEGY = "adaptive" # "fixed" or "adaptive"
FIXED_THRESHOLDS = np.array([-0.0055, 0.0107])

# ──────────────── WALK-FORWARD PARAMETERS ───────────────── #
TRAIN_YEARS = 3
VAL_YEARS = 2
TEST_MONTHS = 12
BUFFER_DAYS = 14  # purge entre splits
HOLDOUT_START_DATE = "2023-01-03"
HOLDOUT_END_DATE = "2025-04-14"

# ─────────────────── MODEL TRAINING PARAMETERS ─────────────────── #
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 15
LEARNING_RATE = 0.01

# ───────────────────── OUTPUT PARAMETERS ────────────────────── #
MODEL_SAVE_DIR = "IA_training/model"
PREDICTION_SAVE_DIR = "prediction"
MODEL_VERSION = "18" # Example, can be incremented or passed as arg

# ─────────────────── BACKTESTING PARAMETERS ──────────────────── #
INITIAL_CASH = 1_000
TRANSACTION_FEE = 0.05 # in percentage
# For 3 classes: 0: short, 1: cash, 2: long
POSITION_SIZE_MAP_3_CLASSES = {0: -0.8, 1: 0.0, 2: 0.8}
# For 5 classes (example if you change N_CLASSES):
# POSITION_SIZE_MAP_5_CLASSES = {0: -0.8, 1: -0.4, 2: 0.0, 3: 0.4, 4: 0.8}
POSITION_SIZE_MAP = POSITION_SIZE_MAP_3_CLASSES # Default to 3 classes