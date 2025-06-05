# pipeline.py

import pandas as pd
import numpy as np
from config import *
from feature_selection import select_top_features_pca, select_top_features_shap, correlation_with_target
from preprocessing import prepare_data
from training import train_model
from model_architectures_v2 import lstm_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

def load_data():
    print("🔹 Chargement des données...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    return df

def create_target(df):
    """
    Ajoute une colonne 'ret_future' qui mesure l'évolution de TARGET_PRICE_COL comme (prix actuel - prix passé) / prix actuel.
    """
    df['ret_future'] = (df[TARGET_PRICE_COL] - df[TARGET_PRICE_COL].shift(PRED_HORIZON)) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)
    scaler = StandardScaler()
    df['ret_future'] = scaler.fit_transform(df[['ret_future']])
    return df


def sclale_sp500_close(df):
    sp500_scaler = StandardScaler()
    df[TARGET_PRICE_COL] = sp500_scaler.fit_transform(df[[TARGET_PRICE_COL]])
    return df

def main():
    
    # 1. Charger les données
    df = load_data()

    df = create_target(df)

    df = sclale_sp500_close(df)

    df = df.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"])

    print(f"📊 Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")

    # print("\n head: ")
    # print(df.head())

    # print("\n describe: ")
    # for col in df.columns:
    #     print(f"\n--- Statistiques pour '{col}' ---")
    #     desc = df[col].describe()
    #     print(desc)
    #     if 'std' in desc and desc['std'] <= 0.97:
    #         print(f"La colonne '{col}' a un écart-type ({desc['std']:.4f}) <= 0.97")

    # print("\n infos: ")
    # print(df.info())

    # print("\n Colonnes: ")
    # print(df.columns)

    # input("Press Enter to continue...")

    # corrélation avec la target
    print("🔹 Calcul de la corrélation avec la target ...")

    correlations = correlation_with_target(df, target_col=TARGET_PRICE_COL)

    print("moyenne des corrélations avec TARGET_PRICE_COL : ", round(correlations.mean(),3))

    print("✅ Corrélations avec la target :")
    # Plot the correlations
    plt.figure(figsize=(8, 6))
    sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
    plt.title(f'Correlation TARGET_PRICE_COL')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')

    # Tableau trié par valeur absolue croissante
    sorted_corr = correlations.reindex(correlations.abs().sort_values().index)
    print("\nTableau des corrélations avec TARGET_PRICE_COL (ordre croissant en valeur absolue) :")
    # print(sorted_corr)

    input("Press Enter to continue...")

    correlations = correlation_with_target(df, target_col="ret_future")

    print(f"moyenne des corrélations avec ret_future ({PRED_HORIZON} jours): ", round(correlations.mean(),3))

    print("✅ Corrélations avec la target :")
    # Plot the correlations
    plt.figure(figsize=(8, 6))
    sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
    plt.title(f'Correlation ret_future avec prediction sur {PRED_HORIZON} jours')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')

    # Tableau trié par valeur absolue croissante
    sorted_corr = correlations.reindex(correlations.abs().sort_values().index)
    print("\nTableau des corrélations avec ret_future (ordre croissant en valeur absolue) :")
    print(sorted_corr)

    # plt.show()
    plt.close("all")

    input("Press Enter to continue...")

    # 2. Sélection des features avec PCA
    print("🔹 Sélection des features avec PCA ...")
    # pca utilisé pour enelver les features corrélées entre elles
    nb_pca_features = 50
    selected_features = select_top_features_pca(df, top_n=nb_pca_features, target_col="ret_future")
    print(f"✅ Features sélectionnées avec PCA : {selected_features}")

    input("Press Enter to continue...")

    df = df[selected_features + ["Date", TARGET_PRICE_COL,"ret_future"]]

    # 2. Sélection des features avec SHAP
    print("🔹 Sélection des features avec shap ...")
    # shap utilisé pour garder les features les plus pertinentes
    selected_features = select_top_features_shap(df, top_n=TOP_N_FEATURES, target_col="ret_future")
    print(f"✅ Features sélectionnées avec shap : {selected_features}")

    df = df[selected_features + ["Date", TARGET_PRICE_COL, "ret_future"]]

    input("Press Enter to continue...")

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
