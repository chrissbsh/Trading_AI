import numpy as np

# ───────────────────────── DATA PARAMETERS ─────────────────────── #
DATA_FILE_PATH = "csv_data/consolidated_data/final_complete_data.csv"
DATE_COL = "Date"
TARGET_PRICE_COL = "SP500_historical_data_Close" # Column used for future return calculation
EXCLUDE_COLS_FROM_FEATURES = {"Date", "ret_future", "target"} # Will be updated after feature selection

# ──────────────────── FEATURE SELECTION PARAMETERS ───────────────── #
TOP_N_FEATURES = 25

# ─────────────────── PREPROCESSING PARAMETERS ──────────────────── #
PRED_HORIZON = 7  # jours, combien de pas dans le futur on veut prédire (horizon = 5 → prédire t+5 à partir de t)
N_CLASSES = 3
SEQUENCE_LENGTH = 90  # nombre de pas de temps utilisés pour prédire la suite
STRIDE = 1  # pas de temps entre les séquences stride=1 → toutes les séquences se chevauchent, stride=sequence_length → elles ne se chevauchent pas.
THRESHOLD_STRATEGY = "fixed" # "fixed" or "adaptive"
FIXED_THRESHOLDS = np.array([-0.02, 0.02])

# ──────────────── WALK-FORWARD PARAMETERS ───────────────── #
TRAIN_YEARS = 3
VAL_YEARS = 2
TEST_MONTHS = 12
HOLDOUT_START_DATE = "2023-01-03"
HOLDOUT_END_DATE = "2025-04-14"

# ─────────────────── MODEL TRAINING PARAMETERS ─────────────────── #
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 20
LEARNING_RATE = 0.01

# ───────────────────── OUTPUT PARAMETERS ────────────────────── #
MODEL_SAVE_DIR = "Pipeline/model"
PREDICTION_SAVE_DIR = "Pipeline/prediction"
MODEL_VERSION = "2" # Example, can be incremented or passed as arg

# ─────────────────── BACKTESTING PARAMETERS ──────────────────── #
INITIAL_CASH = 1_000
TRANSACTION_FEE = 0.05 # in percentage
# For 3 classes: 0: short, 1: cash, 2: long
POSITION_SIZE_MAP_3_CLASSES = {0: -0.8, 1: 0.0, 2: 0.8}
# For 5 classes (example if you change N_CLASSES):
# POSITION_SIZE_MAP_5_CLASSES = {0: -0.8, 1: -0.4, 2: 0.0, 3: 0.4, 4: 0.8}
POSITION_SIZE_MAP = POSITION_SIZE_MAP_3_CLASSES # Default to 3 classes