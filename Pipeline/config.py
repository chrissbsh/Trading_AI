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
TOP_N_FEATURES = 30

# ─────────────────── PREPROCESSING PARAMETERS ──────────────────── #
PRED_HORIZON = 3  # jours — optimise par grid search target_grid_search.py (h=3 meilleur Sharpe/F1)
N_CLASSES = 3
SEQUENCE_LENGTH = 60  # optimise via Optuna BiLSTM (etait 90)
STRIDE = 5  # réduit les séquences corrélées pour limiter l'overfitting (était 1)
THRESHOLD_STRATEGY = "fixed"  # "fixed" — grid search montre que fixed est plus stable
FIXED_THRESHOLDS = np.array([-0.008, 0.008])  # grid search: meilleur equilibre classes (24/43/33%)

# ──────────────── SPLIT PARAMETERS ───────────────── #
HOLDOUT_START_DATE = "2023-01-03"
HOLDOUT_END_DATE = "2025-04-14"

# ─────────────────── MODEL TRAINING PARAMETERS ─────────────────── #
EPOCHS = 200
BATCH_SIZE = 128
PATIENCE = 20  # réduit pour stopper l'overfitting plus tôt (était 50)
LEARNING_RATE = 0.008  # optimise via Optuna BiLSTM (etait 0.01)
N_TRIALS = 150

# Multiplicateurs appliqués PAR-DESSUS le class_weight 'balanced' pour renforcer les classes rares.
CLASS_WEIGHT_BOOST = {0: 0.8, 1: 1, 2: 1.2}

# Seuil de confiance pour la décision finale (différence top1 - top2 probabilité)
ECART_MIN = 0.05

# ───────────────────── OUTPUT PARAMETERS ────────────────────── #
MODEL_SAVE_DIR = "Pipeline/model"
PREDICTION_SAVE_DIR = "Pipeline/prediction"
OPTUNA_DIR = "Pipeline/optuna_results"
MODEL_VERSION = "14.0.0" # Example, can be incremented or passed as arg