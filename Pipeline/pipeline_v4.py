import pandas as pd
import numpy as np
from config import *
from feature_selection import select_top_features_pca, select_top_features_shap
from model_architectures_v2 import lstm_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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

    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}. Utilisez 'fixed'.")

    return df


def main():
    
    # 1. Charger les données
    df_raw = load_data()

    df_raw = df_raw.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"])

    print(f"📊 Données chargées : {len(df_raw)} lignes, {len(df_raw.columns)} colonnes")

    df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)]

    df_main = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE]

    df_main = create_target(df_main)

    df_holdout = create_target(df_holdout)
    
    # Vérification des classes
    class_counts = df_main["target"].value_counts()
    print("🔹 Distribution des classes :")
    print(class_counts)

    # plt.figure()
    # plt.plot(df_main[DATE_COL].values, df_main['ret_future'].values, label='ret_future')
    # plt.title("evolution de la target")
    # plt.xlabel("date")
    # plt.show()

    # 2. Sélection des features avec PCA
    print("🔹 Sélection des features avec PCA ...")
    # pca utilisé pour enelver les features corrélées entre elles
    nb_pca_features = 50
    selected_features = select_top_features_pca(df_main, top_n=nb_pca_features, target_col="ret_future")
    print(f"✅ Features sélectionnées avec PCA : {selected_features}")

    df_main = df_main[selected_features + ["ret_future"]]

    # 2. Sélection des features avec SHAP
    print("🔹 Sélection des features avec shap ...")
    # shap utilisé pour garder les features les plus pertinentes
    selected_features = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
    print(f"✅ Features sélectionnées avec shap : {selected_features}")

    df_main = df_main[selected_features]
    df_holdout = df_holdout[selected_features]

    if 'target' not in df_main.columns:
        df_main = df_main[selected_features + ['target']]

    if 'target' not in df_holdout.columns:
        df_holdout = df_holdout[selected_features + ['target']]

    input("Press Enter to continue...")

    # 3. Mettre en place la Validation Croisée pour Séries Temporelles
    N_SPLITS = 5 # Nombre de plis pour la validation croisée
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Listes pour stocker les scores de chaque pli
    fold_accuracies = []
    fold_f1_scores_macro = []
    all_y_true = []
    all_y_pred = []

    # 4. Boucle de Validation Croisée
    for fold, (train_index, val_index) in enumerate(tscv.split(df_main)):
        print(f"\n{'='*20} FOLD {fold + 1}/{N_SPLITS} {'='*20}")

        # a. Séparer les données pour ce pli
        df_train, df_val = df_main.iloc[train_index].copy(), df_main.iloc[val_index].copy()
        print(f"Train: {len(df_train)} samples | Validation: {len(df_val)} samples")

        # # b. Sélection de features (faite UNIQUEMENT sur les données d'entraînement du pli)
        # # CELA CORRIGE LA FUITE DE DONNÉES
        # print("🔹 Sélection des features pour ce pli...")
        # # Note: on pourrait affiner en ne passant que `features_to_use`
        # features_pca = select_top_features_pca(df_train, top_n=50, target_col="ret_future")
        # final_features = select_top_features_shap(df_train[features_pca + ['ret_future']], top_n=TOP_N_FEATURES, target_col="ret_future")
        # print(f"✅ {len(final_features)} features sélectionnées pour ce pli.")

        final_features = [col for col in selected_features if col != 'target']

        # c. Préparer les données X et y pour ce pli
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(df_train[final_features].to_numpy())
        # X_val = scaler.transform(df_val[final_features].to_numpy())
        X_train = df_train[final_features].to_numpy()
        X_val = df_val[final_features].to_numpy()
        y_train = df_train['target'].to_numpy()
        y_val = df_val['target'].to_numpy()
        
        # d. Créer les générateurs pour ce pli
        train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
        val_gen = TimeseriesGenerator(X_val, y_val, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
        
        # Si le jeu de validation est trop petit pour créer un batch, on saute le pli
        if len(val_gen) == 0:
            print("⚠️ Pli sauté : pas assez de données de validation pour créer une séquence.")
            continue

        # e. Créer et entraîner un NOUVEAU modèle pour chaque pli
        model = lstm_model(input_shape=(SEQUENCE_LENGTH, len(final_features)), n_classes=N_CLASSES)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        model.fit(train_gen, validation_data=val_gen, epochs=20, 
                  callbacks=[early_stopping], class_weight=class_weight_dict, verbose=1)
        
        # f. Évaluer le modèle sur l'ensemble de validation du pli
        y_pred_proba = model.predict(val_gen)

        # Prédiction avec méthode simple
        # y_pred = np.argmax(y_pred_proba, axis=1)

        # Prédiction avec seuil de confiance
        ecart_min = 0.15
        y_pred_val_list = []
        for proba in y_pred_proba:
            sorted_proba = np.sort(proba)[::-1]
            if sorted_proba[0] - sorted_proba[1] >= ecart_min:
                y_pred_val_list.append(np.argmax(proba))
            else:
                y_pred_val_list.append(1) # Classe neutre par défaut pour incertitude (à adapter, 0, 1 ou 2)
        y_pred_final = np.array(y_pred_val_list)

        y_true_list = []
        for i in range(len(val_gen)):
            _, labels = val_gen[i]
            y_true_list.extend(labels)
        y_true = np.array(y_true_list)
        y_pred_final = y_pred_final[:len(y_true)] # S'assurer de l'alignement

        # g. Stocker les scores du pli
        acc = accuracy_score(y_true, y_pred_final)
        f1_macro = f1_score(y_true, y_pred_final, average='macro')
        fold_accuracies.append(acc)
        fold_f1_scores_macro.append(f1_macro)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred_final)
        
        print(f"Fold {fold + 1} -> Accuracy: {acc:.4f} | F1-Score Macro: {f1_macro:.4f}")


    # 5. Afficher les résultats de la validation croisée
    print(f"\n{'='*20} RÉSULTATS DE LA VALIDATION CROISÉE {'='*20}")
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores_macro)
    std_f1 = np.std(fold_f1_scores_macro)

    print(f"Accuracy Moyenne: {mean_acc:.4f} (std: {std_acc:.4f})")
    print(f"F1-Score Macro Moyen: {mean_f1:.4f} (std: {std_f1:.4f})")

    # Matrice de confusion globale sur tous les plis de validation
    print("\nMatrice de confusion et rapport de classification sur l'ensemble des prédictions de validation :")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(classification_report(all_y_true, all_y_pred))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion Globale (sur les ensembles de validation)')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités Terrain')
    plt.show()


    # 6. Évaluation sur l'ensemble de holdout
    print("\n🔹 Évaluation sur l'ensemble de holdout...")
    # b. Préparer les données finales
    X_train_final = df_main[final_features].to_numpy()
    y_train_final = df_main['target'].to_numpy()
    
    X_test_final = df_holdout[final_features].to_numpy()
    y_test_final = df_holdout['target'].to_numpy()

    # c. Créer les générateurs finaux
    train_gen_final = TimeseriesGenerator(X_train_final, y_train_final, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
    test_gen_final = TimeseriesGenerator(X_test_final, y_test_final, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

    if len(test_gen_final) == 0:
        print("❌ Erreur : Pas assez de données dans l'ensemble Hold-Out pour effectuer un test.")
        return # Quitter si le test n'est pas possible

    # d. Créer et entraîner le modèle FINAL
    # Il est important de créer une nouvelle instance du modèle
    final_model = lstm_model(input_shape=(SEQUENCE_LENGTH, len(final_features)), n_classes=N_CLASSES)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # On utilise les class_weights calculés sur tout l'ensemble d'entraînement
    class_weights_final = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
    class_weight_dict_final = dict(enumerate(class_weights_final))
    
    print("🔹 Entraînement du modèle final...")
    # Pour l'entraînement final, on n'utilise pas de données de validation.
    # On entraîne sur un nombre fixe d'époques. 20 est un exemple.
    early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    
    final_model.fit(train_gen_final, epochs=20, callbacks=[early_stopping], class_weight=class_weight_dict_final, verbose=1)

    # e. Évaluation sur l'ensemble Hold-Out
    print("\n--- Évaluation sur l'ensemble Hold-Out (données jamais vues) ---")
    y_pred_proba_final = final_model.predict(test_gen_final)
    
    # Seuil de confiance
    ecart_min = 0.15
    y_pred_final_list = []
    for proba in y_pred_proba_final:
        sorted_proba = np.sort(proba)[::-1]
        if sorted_proba[0] - sorted_proba[1] >= ecart_min:
            y_pred_final_list.append(np.argmax(proba))
        else:
            y_pred_final_list.append(1) # Classe neutre par défaut pour incertitude (à adapter, 0, 1 ou 2)
    y_pred_final = np.array(y_pred_final_list)

    # Récupération robuste de y_true
    y_true_final_list = []
    for i in range(len(test_gen_final)):
        _, labels = test_gen_final[i]
        y_true_final_list.extend(labels)
    y_true_final = np.array(y_true_final_list)
    y_pred_final = y_pred_final[:len(y_true_final)] # Alignement

    # f. Affichage des performances finales
    print("\n--- Performances sur l'ensemble Hold-Out ---")
    
    acc_final = accuracy_score(y_true_final, y_pred_final)
    f1_macro_final = f1_score(y_true_final, y_pred_final, average='macro')
    
    print(f"Accuracy sur Hold-Out: {acc_final:.4f}")
    print(f"F1-Score Macro sur Hold-Out: {f1_macro_final:.4f}")

    print("\nRapport de classification (Hold-Out):")
    print(classification_report(y_true_final, y_pred_final))

    cm_final = confusion_matrix(y_true_final, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', xticklabels=np.arange(N_CLASSES), yticklabels=np.arange(N_CLASSES))
    plt.title('Matrice de Confusion (Hold-Out)')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités Terrain')
    plt.show()


if __name__ == "__main__":
    main()
