import pandas as pd
import numpy as np
from config import *
import os
from feature_selection import select_top_features_pca, select_top_features_shap
# from model_architectures_v2 import *
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras import regularizers # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Lambda, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

def lstm_model_v2(input_shape, seq_len_keep, n_classes):
    """
    Parameters
    ----------
    input_shape : tuple
        (SEQUENCE_LENGTH + PRED_HORIZON, n_features)
    seq_len_keep : int
        Longueur réelle de la fenêtre (= SEQUENCE_LENGTH) à conserver
    n_classes : int
        Nombre de classes (3 ici)
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # ⬇️ On supprime les 'horizon' derniers pas (ici 30)
    model.add(Lambda(lambda z: z[:, :seq_len_keep, :],
                     name="truncate_future"))
    
    model.add(LSTM(64,
                   return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

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
    class_counts = df_main["target"].value_counts(normalize=True)
    print("🔹 Distribution des classes main:")
    print(class_counts)

    class_counts = df_holdout["target"].value_counts(normalize=True)
    print("🔹 Distribution des classes holdout:")
    print(class_counts)

    # plt.figure()
    # plt.plot(df_main[DATE_COL].values, df_main['ret_future'].values, label='ret_future')
    # plt.plot(df_holdout[DATE_COL].values, df_holdout['ret_future'].values, label='ret_future (Holdout)')
    # plt.title("evolution de la target")
    # plt.legend()
    # plt.xlabel("date")
    # plt.show()

    # # 2. Sélection des features avec PCA
    # print("🔹 Sélection des features avec PCA ...")
    # # pca utilisé pour enelver les features corrélées entre elles
    # nb_pca_features = 50
    # selected_features = select_top_features_pca(df_main, top_n=nb_pca_features, target_col="ret_future")
    # print(f"✅ Features sélectionnées avec PCA : {selected_features}")

    # df_main = df_main[selected_features + ["ret_future"]]

    # # 2. Sélection des features avec SHAP
    # print("🔹 Sélection des features avec shap ...")
    # # shap utilisé pour garder les features les plus pertinentes
    # selected_features = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
    # print(f"✅ Features sélectionnées avec shap : {selected_features}")

    # df_main = df_main[selected_features]
    # df_holdout = df_holdout[selected_features]

    # if 'target' not in df_main.columns:
    #     df_main = df_main[selected_features + ['target']]

    # if 'target' not in df_holdout.columns:
    #     df_holdout = df_holdout[selected_features + ['target']]

    # input("Press Enter to continue...")

    if cross_validation:
        print("🔹 Mode Validation Croisée activé.")

        # 3. Mettre en place la Validation Croisée pour Séries Temporelles
        N_SPLITS = 5 # Nombre de plis pour la validation croisée
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        # Listes pour stocker les scores de chaque pli
        fold_accuracies = []
        fold_accuracies_balanced = []
        fold_f1_scores_macro = []
        fold_f1_scores_weighted = []
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
            print("🔹 Sélection des features pour ce pli...")
            # Note: on pourrait affiner en ne passant que `features_to_use`
            # features_pca = select_top_features_pca(df_train, top_n=50, target_col="ret_future")

            final_features = select_top_features_shap(df_train, top_n=TOP_N_FEATURES, target_col="ret_future")

            # final_features = select_top_features_shap(df_train[features_pca + ['ret_future']], top_n=TOP_N_FEATURES, target_col="ret_future")
            print(f"✅ {len(final_features)} features sélectionnées pour ce pli.")

            final_features = [col for col in final_features if col != 'target' and col != 'ret_future']

            if 'SP500_historical_data_Close' not in final_features:
                final_features.append('SP500_historical_data_Close')

            print("Features final: \n", final_features)

            input("Enter to continue")

            # c. Préparer les données X et y pour ce pli
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train[final_features].to_numpy())
            X_val = scaler.transform(df_val[final_features].to_numpy())
            y_train = df_train['target'].to_numpy()
            y_val = df_val['target'].to_numpy()

            # e. Créer et entraîner un NOUVEAU modèle pour chaque pli
            full_seq_len   = SEQUENCE_LENGTH + PRED_HORIZON  # 90 + 30 = 120
            input_shape    = (full_seq_len, len(final_features))
            
            # d. Créer les générateurs pour ce pli
            train_gen = TimeseriesGenerator(X_train, y_train, length=full_seq_len, batch_size=BATCH_SIZE)
            val_gen = TimeseriesGenerator(X_val, y_val, length=full_seq_len, batch_size=BATCH_SIZE)

            model = lstm_model_v2(input_shape, seq_len_keep=SEQUENCE_LENGTH, n_classes=N_CLASSES)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = class_weights * 0.4
            class_weight_dict = dict(enumerate(class_weights))

            model.fit(train_gen, validation_data=val_gen, epochs=20, 
                    callbacks=[early_stopping], class_weight=class_weight_dict, verbose=1)
            
            # f. Évaluer le modèle sur l'ensemble de validation du pli
            y_pred_proba = model.predict(val_gen)

            # Prédiction avec méthode simple
            # y_pred = np.argmax(y_pred_proba, axis=1)

            # Prédiction avec seuil de confiance
            ecart_min = 0.1
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
            balanced_acc = balanced_accuracy_score(y_true, y_pred_final)
            f1_macro = f1_score(y_true, y_pred_final, average='macro')
            f1_weighted = f1_score(y_true, y_pred_final , average='weighted', zero_division=0)

            fold_accuracies.append(acc)
            fold_accuracies_balanced.append(balanced_acc)
            fold_f1_scores_macro.append(f1_macro)
            fold_f1_scores_weighted.append(f1_weighted)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred_final)
            
            print(f"Fold {fold + 1} -> Accuracy: {acc:.4f} | F1-Score Macro: {f1_macro:.4f}")


        # 5. Afficher les résultats de la validation croisée
        print(f"\n{'='*20} RÉSULTATS DE LA VALIDATION CROISÉE {'='*20}")
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        mean_balanced_acc = np.mean(fold_accuracies_balanced)
        std_balanced_acc = np.std(fold_accuracies_balanced)
        mean_f1 = np.mean(fold_f1_scores_macro)
        std_f1 = np.std(fold_f1_scores_macro)
        mean_f1_weighted = np.mean(fold_f1_scores_weighted)
        std_f1_weighted = np.std(fold_f1_scores_weighted)

        print(f"Accuracy Moyenne: {mean_acc:.4f} (std: {std_acc:.4f})")
        print(f"Balanced Accuracy Moyenne: {mean_balanced_acc:.4f} (std: {std_balanced_acc:.4f})")
        print(f"F1-Score Macro Moyen: {mean_f1:.4f} (std: {std_f1:.4f})")
        print(f"F1-Score Pondéré Moyen: {mean_f1_weighted:.4f} (std: {std_f1_weighted:.4f})")

        # Matrice de confusion globale sur tous les plis de validation
        print("\nMatrice de confusion et rapport de classification sur l'ensemble des prédictions de validation :")
        cm = confusion_matrix(all_y_true, all_y_pred)
        print(classification_report(all_y_true, all_y_pred, zero_division=np.nan))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion Globale (sur les ensembles de validation)')
        plt.xlabel('Prédictions')
        plt.ylabel('Vérités Terrain')
        plt.show()


    # 6. Évaluation sur l'ensemble de holdout
    # a. Sélection de features sur TOUT l'ensemble d'entraînement
    print("🔹 Sélection des features finales sur l'ensemble de df_main...")
    # features_pca_final = select_top_features_pca(df_main, top_n=50, target_col="ret_future")

    final_features_for_model = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")

    # final_features_for_model = select_top_features_shap(df_main[features_pca_final + ['ret_future']], top_n=TOP_N_FEATURES, target_col="ret_future")
    print(f"✅ {len(final_features_for_model)} features sélectionnées pour le modèle final.")

    final_features_for_model = [col for col in final_features_for_model if col != 'target' and col != 'ret_future']

    if 'SP500_historical_data_Close' not in final_features_for_model:
        final_features_for_model.append('SP500_historical_data_Close')

    # b. Préparer les données X et y finales
    X_train_final_df = df_main[final_features_for_model]
    X_test_final_df = df_holdout[final_features_for_model]

    y_train_final = df_main['target'].to_numpy()
    y_test_final = df_holdout['target'].to_numpy()

    print(f"🔹 Features finales pour le modèle : {final_features_for_model}")

    input("Press Enter to continue...")

    # c. Mettre à l'échelle les données finales
    print("🔹 Mise à l'échelle des données finales...")
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final_df.to_numpy())
    X_test_final_scaled = final_scaler.transform(X_test_final_df.to_numpy())

    full_seq_len   = SEQUENCE_LENGTH + PRED_HORIZON  # 90 + 30 = 120
    input_shape    = (full_seq_len, len(final_features_for_model))

    # d. Créer les générateurs finaux
    train_gen_final = TimeseriesGenerator(X_train_final_scaled, y_train_final, length=full_seq_len, batch_size=BATCH_SIZE)
    test_gen_final = TimeseriesGenerator(X_test_final_scaled, y_test_final, length=full_seq_len, batch_size=BATCH_SIZE)

    # d. Créer et entraîner le modèle FINAL
    # Il est important de créer une nouvelle instance du modèle
    # final_model = lstm_model_v2(input_shape=(SEQUENCE_LENGTH, len(final_features_for_model)), n_classes=N_CLASSES)

    final_model = lstm_model_v2(input_shape, seq_len_keep=SEQUENCE_LENGTH, n_classes=N_CLASSES)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # On utilise les class_weights calculés sur tout l'ensemble d'entraînement
    class_weights_final = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
    class_weights_final[0] = class_weights_final[0] * 0.3
    class_weights_final[1] = class_weights_final[1] * 1
    class_weights_final[2] = class_weights_final[2] * 0.5
    class_weight_dict_final = dict(enumerate(class_weights_final))

    print(f"🔹 Class weights pour l'entraînement final : {class_weight_dict_final}")

    # input("Press Enter to continue...")
    
    print("🔹 Entraînement du modèle final...")
    # Pour l'entraînement final, on n'utilise pas de données de validation.
    early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    
    final_model.fit(train_gen_final, epochs=20, callbacks=[early_stopping], class_weight=class_weight_dict_final, verbose=1)

    # e. Évaluation sur l'ensemble Hold-Out
    print("\n--- Évaluation sur l'ensemble Hold-Out (données jamais vues) ---")
    y_pred_proba_final = final_model.predict(test_gen_final)

    print("Probabilités brutes (premiers 10):")
    print(y_pred_proba_final[:30])
    
    # Seuil de confiance
    ecart_min = 0.05
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

    print("Probabilités finales: ")
    print(y_pred_final[:30])

    print("Vérités terrain finales: ")
    print(y_true_final[:30])

    # enregistre les prédictions et vérités terrain
    print("🔹 Enregistrement des prédictions et vérités terrain...")
    os.makedirs(PREDICTION_SAVE_DIR, exist_ok=True)
    predictions_df = pd.DataFrame({
        'Date': df_holdout[DATE_COL].values[:len(y_true_final)],
        'probabilities': [list(proba) for proba in y_pred_proba_final[:len(y_true_final)]],
        'y_true': y_true_final,
        'y_pred': y_pred_final
    })
    predictions_path = os.path.join(PREDICTION_SAVE_DIR, f"predictions_v{MODEL_VERSION}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ Prédictions et vérités terrain enregistrées à {predictions_path}")

    # f. Affichage des performances finales
    print("\n--- Performances sur l'ensemble Hold-Out ---")
    
    acc_final = accuracy_score(y_true_final, y_pred_final)
    balanced_acc = balanced_accuracy_score(y_true_final, y_pred_final)
    f1_macro_final = f1_score(y_true_final, y_pred_final, average='macro', zero_division=np.nan)
    f1_weighted_final = f1_score(y_true_final, y_pred_final, average='weighted', zero_division=np.nan)
    
    print(f"Accuracy sur Hold-Out: {acc_final:.4f}")
    print(f"Balanced Accuracy sur Hold-Out: {balanced_acc:.4f}")
    print(f"F1-Score Macro sur Hold-Out: {f1_macro_final:.4f}")
    print(f"F1-Score Pondéré sur Hold-Out: {f1_weighted_final:.4f}")

    print("\nRapport de classification (Hold-Out):")
    print(classification_report(y_true_final, y_pred_final, zero_division=np.nan))

    # affichage nombre de classes 0 prédite vs nombre de classes 0 réelles
    print(f"Nombre de classes 0 prédites: {np.sum(y_pred_final == 0)}")
    print(f"Nombre de classes 0 réelles: {np.sum(y_true_final == 0)}")

    print(f"Nombre de classes 1 prédites: {np.sum(y_pred_final == 1)}")
    print(f"Nombre de classes 1 réelles: {np.sum(y_true_final == 1)}")

    print(f"Nombre de classes 2 prédites: {np.sum(y_pred_final == 2)}")
    print(f"Nombre de classes 2 réelles: {np.sum(y_true_final == 2)}")


    cm_final = confusion_matrix(y_true_final, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', xticklabels=np.arange(N_CLASSES), yticklabels=np.arange(N_CLASSES))
    plt.title('Matrice de Confusion (Hold-Out)')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités Terrain')
    plt.show()

    # g. Sauvegarde du modèle final
    print("🔹 Sauvegarde du modèle final...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, f"model_v{MODEL_VERSION}.keras")
    final_model.save(model_path)
    print(f"✅ Modèle sauvegardé à {model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_final, label='Prédictions')
    plt.plot(y_true_final, label='Vérités Terrain', alpha=0.7)
    plt.title('Prédictions vs Vérités Terrain (Hold-Out)')
    plt.xlabel('Index')
    plt.ylabel('Classe')
    plt.legend()

    plt.show()

if __name__ == "__main__":

    cross_validation = True  # Set to False to run the full training pipeline without cross-validation

    version = MODEL_VERSION
    print(f"\n--- Running Training Pipeline for Model Version: {version} ---")

    if os.path.exists(f"Pipeline/model/model_v{version}.keras"):
        print(f"Warning: model_v{version}.keras already exists.")
        input("Press Enter to continue or Ctrl+C to cancel...")

    main()