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
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
import tensorflow as tf

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

    # 2. Sélection des features avec PCA
    print("🔹 Sélection des features avec PCA ...")
    # pca utilisé pour enelver les features corrélées entre elles
    nb_pca_features = 50
    selected_features = select_top_features_pca(df, top_n=nb_pca_features, target_col="ret_future")
    print(f"✅ Features sélectionnées avec PCA : {selected_features}")

    df = df[selected_features + ["Date", TARGET_PRICE_COL,"ret_future"]]

    # 2. Sélection des features avec SHAP
    print("🔹 Sélection des features avec shap ...")
    # shap utilisé pour garder les features les plus pertinentes
    selected_features = select_top_features_shap(df, top_n=TOP_N_FEATURES, target_col="ret_future")
    print(f"✅ Features sélectionnées avec shap : {selected_features}")

    df = df[selected_features + ["Date", TARGET_PRICE_COL, "ret_future"]]

    input("Press Enter to continue...")
    
    # labelling target
    if THRESHOLD_STRATEGY == "fixed":

        quantiles = np.linspace(0, 1, N_CLASSES + 1)
        thresholds = df['ret_future'].quantile(quantiles).values[1:-1]

        print(f"🔹 Seuils fixes pour la classification : {thresholds}")

        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2

        df["target"] = df["ret_future"].apply(label_target)

    elif THRESHOLD_STRATEGY == "adaptive":
        raise NotImplementedError("Adaptive thresholds are not implemented in this version.")

    else:
        raise ValueError(f"Unknown threshold strategy: {THRESHOLD_STRATEGY}. Use 'fixed' or 'adaptive'.")
    
    # Vérification des classes
    class_counts = df["target"].value_counts()
    print("🔹 Distribution des classes :")
    print(class_counts)

    # print(df.head(20))

    plt.show()

    input("Press Enter to continue...")

    # Exemple de données
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)

    df_train = df.iloc[:train_size]
    df_val   = df.iloc[train_size:train_size + val_size]
    df_test  = df.iloc[train_size + val_size:]


    X_train = df_train[selected_features].to_numpy()
    y_train = df_train['target'].to_numpy()

    X_val = df_val[selected_features].to_numpy()
    y_val = df_val['target'].to_numpy()

    X_test = df_test[selected_features].to_numpy()
    y_test = df_test['target'].to_numpy()

    train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)
    val_gen   = TimeseriesGenerator(X_val, y_val,     length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)
    test_gen  = TimeseriesGenerator(X_test, y_test,   length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)

    model = lstm_model(input_shape=(SEQUENCE_LENGTH, len(selected_features)), n_classes=N_CLASSES)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
    train_gen,
    epochs=2,
    validation_data=val_gen,
    verbose=1
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Épochs")
    plt.ylabel("Loss")
    plt.show()

    loss = model.evaluate(test_gen)
    print(f"Test loss: {loss}")

    y_pred_proba = model.predict(test_gen)

    print(f"y_pred_proba: {y_pred_proba[-5:]}")  # Affiche les 5 dernières prédictions de probabilité

    # Utiliser la classe avec la probabilité la plus élevée
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"y_pred: {y_pred[-5:]}")

    y_true = y_test[SEQUENCE_LENGTH:]  # aligne avec les prédictions

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1}")

    # afficher la matrice de confusion
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(N_CLASSES), yticklabels=np.arange(N_CLASSES))
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.show()


if __name__ == "__main__":
    main()
