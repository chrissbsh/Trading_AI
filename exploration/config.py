import numpy as np

"""
Ce fichier de configuration regroupe les paramètres nécessaires à l’entraînement et à l’évaluation 
d’un modèle XGBoost pour la classification multi-classes des mouvements du S&P500.

Contenu du fichier :
- Paramètres liés aux données (chemin du fichier CSV, colonne de date, colonne de prix cible).
- Paramètres de preprocessing pour le calcul du rendement futur à horizon défini (`PRED_HORIZON`)
  et la création de la variable cible (`target`) selon une stratégie de seuils (`THRESHOLD_STRATEGY`).
- Configuration de la classification multi-classes avec `N_CLASSES` et `FIXED_THRESHOLDS`.
- Définition de la période de séparation temporelle entre les données d’entraînement et de test (`HOLDOUT_START_DATE` → `HOLDOUT_END_DATE`).

Ce fichier est utilisé pour entraîner un modèle XGBoost classique sur des données tabulaires, sans structure séquentielle.
"""

# ───────────────────────── DATA PARAMETERS ─────────────────────── #
DATA_FILE_PATH = "csv_data/consolidated_data/final_complete_data.csv"
DATE_COL = "Date"
TARGET_PRICE_COL = "SP500_historical_data_Close" # Column used for future return calculation
EXCLUDE_COLS_FROM_FEATURES = {"Date", "ret_future", "target"} # Will be updated after feature selection


# ─────────────────── PREPROCESSING PARAMETERS ──────────────────── #
PRED_HORIZON = 30  # jours, combien de pas dans le futur on veut prédire (horizon = 5 → prédire t+5 à partir de t)
N_CLASSES = 3
STRIDE = 1  # pas de temps entre les séquences stride=1 → toutes les séquences se chevauchent, stride=sequence_length → elles ne se chevauchent pas.
THRESHOLD_STRATEGY = "fixed" # "fixed" or "adaptive"
FIXED_THRESHOLDS = np.array([-0.02, 0.02])

# ──────────────── SPLIT PARAMETERS ───────────────── #
HOLDOUT_START_DATE = "2023-01-03"
HOLDOUT_END_DATE = "2025-04-14"