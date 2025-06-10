import pandas as pd
import numpy as np
from config import *
from feature_selection import select_top_features_pca, select_top_features_shap
from model_architectures_v2 import lstm_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

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
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)

    # print(df[["SP500_historical_data_Close", "ret_future"]].head(20))

    # input("Press Enter to continue...")

    # labelling target
    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        
        print(f"🔹 Seuils pour la classification : {thresholds}")

        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2

        df["target"] = df["ret_future"].apply(label_target)


    elif THRESHOLD_STRATEGY == "adaptive":

        quantiles = np.linspace(0, 1, N_CLASSES + 1)
        thresholds = df['ret_future'].quantile(quantiles).values[1:-1]

        print(f"🔹 Seuils pour la classification : {thresholds}")

        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2

        df["target"] = df["ret_future"].apply(label_target)

    else:
        raise ValueError(f"Unknown threshold strategy: {THRESHOLD_STRATEGY}. Use 'fixed' or 'adaptive'.")

    return df


def main():
    
    # 1. Charger les données
    df = load_data()

    df = df.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"])

    print(f"📊 Données chargées : {len(df)} lignes, {len(df.columns)} colonnes")

    df_holdout = df[(df[DATE_COL] >= HOLDOUT_START_DATE) & (df[DATE_COL] <= HOLDOUT_END_DATE)]

    df = df[df[DATE_COL] < HOLDOUT_START_DATE]

    df = create_target(df)

    df_holdout = create_target(df_holdout)
    
    # Vérification des classes
    class_counts = df["target"].value_counts()
    print("🔹 Distribution des classes :")
    print(class_counts)

    # plt.figure()
    # plt.plot(df[DATE_COL].values, df['ret_future'].values, label='ret_future')
    # plt.title("evolution de la target")
    # plt.xlabel("date")
    # plt.show()

    # 2. Sélection des features avec PCA
    print("🔹 Sélection des features avec PCA ...")
    # pca utilisé pour enelver les features corrélées entre elles
    nb_pca_features = 50
    selected_features = select_top_features_pca(df, top_n=nb_pca_features, target_col="ret_future")
    print(f"✅ Features sélectionnées avec PCA : {selected_features}")

    df = df[selected_features + ["ret_future"]]

    # 2. Sélection des features avec SHAP
    print("🔹 Sélection des features avec shap ...")
    # shap utilisé pour garder les features les plus pertinentes
    selected_features = select_top_features_shap(df, top_n=TOP_N_FEATURES, target_col="ret_future")
    print(f"✅ Features sélectionnées avec shap : {selected_features}")

    df = df[selected_features]
    df_holdout = df_holdout[selected_features]

    if 'target' not in df.columns:
        df = df[selected_features + ['target']]

    if 'target' not in df_holdout.columns:
        df_holdout = df_holdout[selected_features + ['target']]

    input("Press Enter to continue...")

    # Exemple de données
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.2)

    df_train = df.iloc[:train_size]
    df_val   = df.iloc[train_size:train_size + val_size]
    df_test  = df_holdout

    # Créer une liste de features finales SANS la cible
    final_features = [col for col in selected_features if col != 'target']

    X_train = df_train[final_features].to_numpy()
    y_train = df_train['target'].to_numpy()

    X_val = df_val[final_features].to_numpy()
    y_val = df_val['target'].to_numpy()

    X_test = df_test[final_features].to_numpy()
    y_test = df_test['target'].to_numpy()

    print(f"🔹 Taille des données d'entraînement : {X_train.shape}, {y_train.shape}")
    print(f"🔹 Taille des données de validation : {X_val.shape}, {y_val.shape}")
    print(f"🔹 Taille des données de test : {X_test.shape}, {y_test.shape}")

    train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)
    val_gen   = TimeseriesGenerator(X_val, y_val,     length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)
    test_gen  = TimeseriesGenerator(X_test, y_test,   length=SEQUENCE_LENGTH, stride=STRIDE, batch_size=BATCH_SIZE)

    model = lstm_model(input_shape=(SEQUENCE_LENGTH, len(final_features)), n_classes=N_CLASSES)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"🔹 Class weights: {class_weight_dict}")

    input("Press Enter to continue...")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
    train_gen,
    epochs=2,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Courbe d'apprentissage - Perte")
    plt.xlabel("Épochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("Courbe d'apprentissage - Accuracy")
    plt.xlabel("Épochs")
    plt.ylabel("Accuracy")

    plt.show()

    loss = model.evaluate(test_gen)
    print(f"Test loss: {loss}")

    y_pred_proba = model.predict(test_gen)

    print(f"y_pred_proba: {y_pred_proba[-5:]}")  # Affiche les 5 dernières prédictions de probabilité

    # Utiliser la classe avec la probabilité la plus élevée
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"y_pred: {y_pred[-5:]}")

    # Méthode robuste pour récupérer y_true
    y_true_list = []
    for i in range(len(test_gen)):
        _, labels = test_gen[i]
        y_true_list.extend(labels)
    y_true = np.array(y_true_list)

    print(f"y_true: {y_true[-5:]}")

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1}")

    print(classification_report(y_true, y_pred, zero_division=np.nan))
    print(cm)

    # Affichage de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(N_CLASSES), yticklabels=np.arange(N_CLASSES))
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.show()


if __name__ == "__main__":
    main()
