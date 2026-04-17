import numpy as np

"""
Ce fichier de configuration centralise tous les paramètres utilisés dans le pipeline de prédiction 
du S&P500 (classification multi-classes). Il permet de piloter l’ensemble des étapes du processus, 
de la préparation des données à l’entraînement et à l’évaluation des modèles.

Contenu du fichier :
- Paramètres liés aux données (fichier source, colonne de date, colonne cible).
- Paramètres de sélection de features (nombre de variables à retenir).
- Paramètres de preprocessing temporel (horizon de prédiction, taille des séquences, seuils de classification).
- Paramètres du découpage temporel (périodes d’entraînement/validation, test).
- Paramètres d'entraînement du modèle (epochs, batch, learning rate, early stopping).
- Paramètres liés à Optuna (nombre d’essais).
- Répertoires de sauvegarde pour les modèles, prédictions et résultats.

Ce fichier assure la cohérence et la flexibilité du pipeline sans modifier le code principal.
"""

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

# ──────────────── SPLIT PARAMETERS ───────────────── #
HOLDOUT_START_DATE = "2023-01-03"
HOLDOUT_END_DATE = "2025-04-14"

# ─────────────────── MODEL TRAINING PARAMETERS ─────────────────── #
EPOCHS = 200
BATCH_SIZE = 128
PATIENCE = 50
LEARNING_RATE = 0.01
N_TRIALS = 50 # Nombre d'essais Optuna

# ───────────────────── OUTPUT PARAMETERS ────────────────────── #
MODEL_SAVE_DIR = "Pipeline/model"
PREDICTION_SAVE_DIR = "Pipeline/prediction"
OPTUNA_DIR = "Pipeline/optuna_results"
MODEL_VERSION = "13.1.2" # Example, can be incremented or passed as arg