import pandas as pd
import numpy as np
# Assurez-vous d'avoir un fichier config.py ou remplacez les variables par leurs valeurs
from config import * 
import os
import random
import tensorflow as tf

from feature_selection import select_top_features_pca, select_top_features_shap

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tf_metrics import *

## AJOUT OVERSAMPLING ##
# Importation de RandomOverSampler
from imblearn.over_sampling import SMOTE

from focal_loss import SparseCategoricalFocalLoss

# ==================== REPRODUCTIBILITÉ ====================

# Fixer toutes les graines aléatoires
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration TensorFlow pour la reproductibilité
tf.config.experimental.enable_op_determinism()

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

def lstm_model_v2(input_shape, seq_len_keep, n_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Lambda(lambda z: z[:, :seq_len_keep, :], name="truncate_future"))
    model.add(LSTM(64, return_sequences=False, 
                   kernel_regularizer=regularizers.l2(0.01)
                   ))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def load_data(debug_on):
    print("🔹 1) Chargement des données depuis :", DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    print("   → Aperçu des 5 premières lignes :")
    print(df.head())

    if debug_on:
        input("   [PAUSE] Vérifiez le DataFrame chargé puis appuyez sur Entrée...")

    return df

def create_target(df, tag="", debug_on=False):
    print(f"🔹 2) Création de la cible (horizon = {PRED_HORIZON}) {tag}")
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    print("   → Avant suppression des NaN, nombre de lignes :", len(df))
    df.dropna(subset=['ret_future'], inplace=True)
    print("   → Après suppression, nombre de lignes :", len(df))
    print("   → Aperçu ret_future :")
    print(df[[DATE_COL, TARGET_PRICE_COL, 'ret_future']].head(10))

    if debug_on:    
        input("   [PAUSE] Vérifiez ret_future puis Entrée...")

    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        print(f"   → Seuils utilisés pour labellisation : {thresholds}")
        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2
        df["target"] = df["ret_future"].apply(label_target)
        print("   → Répartition des classes après labellisation :")
        print(df["target"].value_counts(normalize=True))
    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}")

    if debug_on:
        input("   [PAUSE] Vérifiez la colonne 'target' puis Entrée...")

    return df

def main(debug_on, feature_selection_on):
    # -- Chargement et split --
    df_raw = load_data(debug_on)
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    print(f"🔹 Colonnes restantes : {df_raw.columns.tolist()}")
    print(f"🔹 Total lignes après nettoyage : {len(df_raw)}")

    if debug_on:
        input("   [PAUSE] Vérifiez le nettoyage initial puis Entrée...")

    df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
    df_main    = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    print(f"🔹 Lignes pour entraînement (avant target) : {len(df_main)}")
    print(f"🔹 Lignes pour holdout (avant target)    : {len(df_holdout)}")

    if debug_on:
        input("   [PAUSE] Vérifiez les splits temporels puis Entrée...")

    df_main    = create_target(df_main, tag="(train)", debug_on=debug_on)
    df_holdout = create_target(df_holdout, tag="(holdout)", debug_on=debug_on)

    # -- Distribution des classes --
    print("🔹 Distribution classes TRAIN :")
    print(df_main["target"].value_counts(normalize=True))
    print("🔹 Distribution classes HOLDOUT :")
    print(df_holdout["target"].value_counts(normalize=True))
    if debug_on:
        input("   [PAUSE] Vérifiez distributions puis Entrée...")

    if feature_selection_on:
        # -- Pipeline final sur holdout --
        print("\n🔹 Sélection finale des features sur tout df_main")
        raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
        final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
        if 'SP500_historical_data_Close' not in final_feats:
            final_feats.append('SP500_historical_data_Close')
        print("   → Features finales :", final_feats)

    else:
        final_feats = [f for f in df_main.columns if f not in ('target', 'ret_future')]
    
    if debug_on:
        input("   [PAUSE] Vérifiez les features finales puis Entrée...")

    # --- 1. Division des données en Train / Validation / Test ---
    print("🔹 Division chronologique des données...")
    split_idx = int(0.8 * len(df_main))
    train_df = df_main.iloc[:split_idx]
    val_df = df_main.iloc[split_idx:]
    test_df = df_holdout

    print(f"Shapes des DataFrames : Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")

    # --- 2. Préparation des X/y et mise à l'échelle ---
    X_train = train_df[final_feats]
    y_train = train_df['target'].to_numpy()
    X_val = val_df[final_feats]
    y_val = val_df['target'].to_numpy()
    X_test = test_df[final_feats]
    y_test = test_df['target'].to_numpy()
    date_test = test_df[DATE_COL].values

    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_val_scaled   = final_scaler.transform(X_val)
    X_test_scaled  = final_scaler.transform(X_test)

    print("🔹 Shapes finales X (scaled) :", X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)
    print("🔹 Shapes finales y :", y_train.shape, y_val.shape, y_test.shape)

    if debug_on:
        input("   [PAUSE] Vérifiez les données train/val/test puis Entrée...")

    # --- 3. Création des TimeseriesGenerators ---
    full_seq = SEQUENCE_LENGTH + PRED_HORIZON
    gen_train = TimeseriesGenerator(X_train_scaled, y_train, length=full_seq, batch_size=BATCH_SIZE)
    gen_val   = TimeseriesGenerator(X_val_scaled, y_val, length=full_seq, batch_size=BATCH_SIZE)
    gen_test  = TimeseriesGenerator(X_test_scaled, y_test, length=full_seq, batch_size=BATCH_SIZE)

    print("🔹 Batches finaux train/validation/test :", len(gen_train), len(gen_val), len(gen_test))
    
    # --- 4. ## AJOUT OVERSAMPLING ## - Application de RandomOverSampler ---
    print("\n🔹 Application de l'Oversampling sur les données d'entraînement...")
    
    # Étape A : Extraire toutes les séquences du générateur d'entraînement.
    # C'est nécessaire car imblearn a besoin de l'ensemble des données en mémoire.
    X_train_sequences_list = []
    y_train_sequences_list = []
    for i in range(len(gen_train)):
        x, y = gen_train[i]
        X_train_sequences_list.append(x)
        y_train_sequences_list.append(y)

    X_train_sequences = np.concatenate(X_train_sequences_list)
    y_train_sequences = np.concatenate(y_train_sequences_list)

    print(f"   → Shape des séquences d'entraînement AVANT oversampling : {X_train_sequences.shape}")
    print(f"   → Répartition des classes AVANT oversampling :\n{pd.Series(y_train_sequences).value_counts()}")

    # Étape B : Aplatir les séquences pour que RandomOverSampler puisse les traiter.
    # Le sur-échantillonneur attend des données 2D (samples, features).
    n_samples, seq_len, n_features = X_train_sequences.shape
    X_train_reshaped = X_train_sequences.reshape(n_samples, seq_len * n_features)

    # Étape C : Appliquer l'oversampling
    # SMOTE ne fonctionne pas directement sur les données 3D, donc l'aplatissement est correct.
    smote = SMOTE(random_state=SEED)
    X_resampled_flat, y_resampled = smote.fit_resample(X_train_reshaped, y_train_sequences)

    # Étape D : Redonner aux données leur forme de séquence 3D pour le LSTM
    X_train_resampled = X_resampled_flat.reshape(-1, seq_len, n_features)

    print(f"   → Shape des séquences d'entraînement APRÈS oversampling : {X_train_resampled.shape}")
    print(f"   → Répartition des classes APRÈS oversampling :\n{pd.Series(y_resampled).value_counts()}")
    
    if debug_on:
        input("   [PAUSE] Vérifiez les données sur-échantillonnées puis Entrée...")

    # --- 5. Entraînement du modèle ---
    final_model = lstm_model_v2((full_seq, len(final_feats)), SEQUENCE_LENGTH, N_CLASSES)
    metrics = ['accuracy', F1Macro(3), BalancedAcc(3)]

    gamma = 5
    ## AJOUT OVERSAMPLING ##: On retire les class_weight de la loss, 
    # car le dataset est maintenant équilibré. Appliquer les deux serait redondant.
    loss_fn = SparseCategoricalFocalLoss(gamma=gamma)

    final_model.compile(optimizer='adam',
                        loss=loss_fn,
                        metrics=metrics)

    early_stop = EarlyStopping(
        monitor='val_f1_macro',
        mode='max',
        patience=PATIENCE,
        restore_best_weights=True)

    ## AJOUT OVERSAMPLING ##: On utilise les données sur-échantillonnées pour l'entraînement.
    # La validation se fait toujours sur gen_val pour une évaluation non biaisée.
    final_model.fit(X_train_resampled, y_resampled,
                    batch_size=BATCH_SIZE, # Il faut spécifier batch_size ici
                    validation_data=gen_val,
                    epochs=EPOCHS,
                    callbacks=[early_stop], 
                    verbose=1)

    if debug_on:
        input("   [PAUSE] Entraînement final OK. Entrée pour évaluation...")

    # --- 6. Évaluation sur le Holdout ---
    print("\n--- Évaluation HOLDOUT ---")
    # L'évaluation se fait sur gen_test, qui n'a JAMAIS été sur-échantillonné.
    proba_final = final_model.predict(gen_test)
    y_true_final = np.concatenate([gen_test[i][1] for i in range(len(gen_test))])
    # S'assurer que les probabilités correspondent aux étiquettes extraites du générateur
    proba_final = proba_final[:len(y_true_final)]

    # Exemples de dates + prédictions (Hold-Out)
    print("🔹 Quelques exemples de fenêtres et dates (Hold-Out):")
    for j in range(min(3, len(y_true_final))):
        start_idx = j
        end_input_idx = start_idx + SEQUENCE_LENGTH - 1
        pred_idx = start_idx + full_seq -1 # L'étiquette y_true[j] correspond à la fin de la fenêtre de prédiction
        
        # S'assurer que les indices sont valides
        if pred_idx < len(date_test):
            start_date     = pd.to_datetime(date_test[start_idx])
            end_input_date = pd.to_datetime(date_test[end_input_idx])
            # La date de prédiction correspond au point de données pour lequel on prédit
            pred_date      = pd.to_datetime(date_test[pred_idx])
            print(f"  Sample {j}: Input de {start_date.date()} à {end_input_date.date()}, Prédiction pour {pred_date.date()}, y_true={y_true_final[j]}")
        
    if debug_on:    
        input("   [PAUSE] Vérifiez dates des fenêtres HoldOut puis Entrée...")

    # Seuil confiance & y_pred_final
    ecart_min = ECART_MIN
    y_pred_final_list = []
    for proba in proba_final:
        sorted_proba = np.sort(proba)[::-1]
        if sorted_proba[0] - sorted_proba[1] >= ecart_min:
            y_pred_final_list.append(np.argmax(proba))
        else:
            y_pred_final_list.append(1) # Classe neutre par défaut
    y_pred_final = np.array(y_pred_final_list)

    # Sauvegarde des probabilités, prédictions et valeurs réelles
    proba_df = pd.DataFrame(proba_final, columns=[f"proba_class_{i}" for i in range(proba_final.shape[1])])
    proba_df["y_pred"] = y_pred_final
    proba_df["y_true"] = y_true_final
    proba_df["date"] = date_test[full_seq:]  # Décalage dû au TimeseriesGenerator
    proba_df = proba_df.reset_index(drop=True)
    output_csv_path = os.path.join(PREDICTION_SAVE_DIR, f"predictions_v{MODEL_VERSION}.csv")
    proba_df.to_csv(output_csv_path, index=False)
    print(f"🔹 Fichier CSV sauvegardé : {output_csv_path}")

    # Scores finaux
    acc_f  = accuracy_score(y_true_final, y_pred_final)
    bal_f  = balanced_accuracy_score(y_true_final, y_pred_final)
    f1m_f  = f1_score(y_true_final, y_pred_final, average='macro')
    f1w_f  = f1_score(y_true_final, y_pred_final, average='weighted')
    print(f"   → Accuracy final : {acc_f:.4f}")
    print(f"   → Balanced Acc. : {bal_f:.4f}")
    print(f"   → F1-macro      : {f1m_f:.4f}")
    print(f"   → F1-weighted      : {f1w_f:.4f}")


    print(f"Nombre de classe 0 prédite (y_pred_final): {(y_pred_final == 0).sum()}")
    print(f"Nombre de classe 0 réelle (y_true_final): {(y_true_final == 0).sum()}")

    print(f"Nombre de classe 1 prédite (y_pred_final): {(y_pred_final == 1).sum()}")
    print(f"Nombre de classe 1 réelle (y_true_final): {(y_true_final == 1).sum()}")

    print(f"Nombre de classe 2 prédite (y_pred_final): {(y_pred_final == 2).sum()}")
    print(f"Nombre de classe 2 réelle (y_true_final): {(y_true_final == 2).sum()}")



    print("\n" + classification_report(y_true_final, y_pred_final, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_final, y_pred_final))

    if debug_on:
        input("   [PAUSE] Vérifiez le rapport de classification puis Entrée...")

    # ... (suite du code pour sauvegarde, etc.)

if __name__ == "__main__":
    debug_on = False
    feature_selection_on = True 
    ECART_MIN = 0.05

    print(f"\n--- Lancement pipeline version : {MODEL_VERSION} ---")
    if os.path.exists(f"Pipeline/model/model_v{MODEL_VERSION}.keras"):
        print("⚠️  Le modèle existe déjà :", f"model_v{MODEL_VERSION}.keras")
        input("   Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")

    main(debug_on, feature_selection_on)
