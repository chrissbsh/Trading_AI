# pipeline.py

import pandas as pd
import numpy as np
from config import *
from feature_selection import select_top_features
from preprocessing import prepare_data
from training import train_model
from model_architectures_v2 import lstm_model
from sklearn.preprocessing import StandardScaler

def load_data():
    print("🔹 Chargement des données...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    return df

def create_target(df):
    df['ret_future'] = df[TARGET_PRICE_COL].pct_change(periods=PRED_HORIZON).shift(-PRED_HORIZON)
    df.dropna(subset=['ret_future'], inplace=True)
    scaler = StandardScaler()
    df['ret_future'] = scaler.fit_transform(df[['ret_future']])
    return df


def sclale_sp500_close(df):
    sp5scaler = StandardScaler()
    df[TARGET_PRICE_COL] = scaler.fit_transform(df[TARGET_PRICE_COL])
    return df


def main():
    
    # 1. Charger les données
    df = load_data()

    df = create_target(df)

    df = df.drop(columns=["sp500_prev_close"])

    print(f"📊 Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")

    print("\n head: ")
    print(df.head())

    print("\n describe: ")
    print(df.describe())

    print("\n infos: ")
    print(df.info())

    print("\n Colonnes: ")
    print(df.columns)

    input("Press Enter to continue...")

    # 2. Sélection des features
    print("🔹 Sélection des features...")
    selected_features = select_top_features(df, target_col="ret_future", top_n=TOP_N_FEATURES)
    print(f"✅ Features sélectionnées : {selected_features}")

    # 3. Prétraitement (labeling, thresholds, jitter, split walk-forward)
    print("🔹 Prétraitement des données...")
    walk_forward_splits = prepare_data(
        df, selected_features,
        date_col=DATE_COL,
        target_price_col=TARGET_PRICE_COL,
        pred_horizon=PRED_HORIZON,
        n_classes=N_CLASSES,
        threshold_strategy=THRESHOLD_STRATEGY,
        threshold_step_months=THRESHOLD_STEP_MONTHS,
        fixed_thresholds=FIXED_THRESHOLDS,
        jitter_ratio=JITTER_RATIO,
        timesteps=TIMESTEPS,
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
        test_months=TEST_MONTHS,
        buffer_days=BUFFER_DAYS,
        holdout_start=HOLDOUT_START_DATE,
        holdout_end=HOLDOUT_END_DATE
    )

    # 4. Boucle d'entraînement walk-forward
    print("🔹 Démarrage de l'entraînement walk-forward...")
    for i, (X_train, y_train, X_val, y_val, X_test, y_test, test_dates) in enumerate(walk_forward_splits):
        print(f"📦 Split {i+1}/{len(walk_forward_splits)}")

        model = lstm_model()
        
        history, best_model = train_model(
            model, X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            lr=LEARNING_RATE,
            model_save_path=f"{MODEL_SAVE_DIR}/model_v{MODEL_VERSION}_split{i+1}.h5"
        )

        # Optionnel : évaluation ou prédictions ici

if __name__ == "__main__":
    main()
