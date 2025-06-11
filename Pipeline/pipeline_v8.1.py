import pandas as pd
import numpy as np
from config import *
import os
from feature_selection import select_top_features_pca, select_top_features_shap
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda # type: ignore
from tensorflow.keras import regularizers # type: ignore
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class_weight_list = [1, 1, 1]
# class_weight_list = [0.3, 1, 0.5]

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

def lstm_model_v2(input_shape, seq_len_keep, n_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Lambda(lambda z: z[:, :seq_len_keep, :], name="truncate_future"))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
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

def main(cross_validation, debug_on):
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

    # -- Cross-validation ou pipeline complet --
    if cross_validation:
        print("🔹 Mode Validation Croisée activé")
        tscv = TimeSeriesSplit(n_splits=5)
        fold_stats = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_main)):
            print(f"\n========== FOLD {fold+1} ==========")
            df_train = df_main.iloc[train_idx].copy()
            df_val   = df_main.iloc[val_idx].copy()
            date_train = df_train[DATE_COL].values
            date_val   = df_val[DATE_COL].values
            print(f"👉 Train: {len(df_train)}, Val: {len(df_val)}")
            if debug_on:
                input("   [PAUSE] Vérifiez les indexes de split puis Entrée...")

            # Sélection de features sur df_train uniquement
            print("🔹 Sélection features SHAP...")
            raw_features = select_top_features_shap(df_train, top_n=TOP_N_FEATURES, target_col="ret_future")
            features = [f for f in raw_features if f not in ('target', 'ret_future')]
            if 'SP500_historical_data_Close' not in features:
                features.append('SP500_historical_data_Close')
            print(f"   → features finales utilisées : {features}")

            if debug_on:
                input("   [PAUSE] Vérifiez features puis Entrée...")

            # Préparation X/y
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train[features])
            X_val   = scaler.transform(df_val[features])
            y_train = df_train['target'].to_numpy()
            y_val   = df_val['target'].to_numpy()
            print("🔹 Shape X_train, X_val, y_train, y_val :", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            
            if debug_on: 
                input("   [PAUSE] Vérifiez les shapes puis Entrée...")

            # Générateurs
            full_seq = SEQUENCE_LENGTH + PRED_HORIZON
            train_gen = TimeseriesGenerator(X_train, y_train, length=full_seq, batch_size=BATCH_SIZE)
            val_gen   = TimeseriesGenerator(X_val,   y_val,   length=full_seq, batch_size=BATCH_SIZE)
            print("🔹 Nombre de batches train/val :", len(train_gen), len(val_gen))
            
            if debug_on:
                input("   [PAUSE] Vérifiez les generators puis Entrée...")

            # Modèle
            model = lstm_model_v2((full_seq, len(features)), SEQUENCE_LENGTH, N_CLASSES)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("🔹 Modèle compilé. Architecture :")
            model.summary()

            if debug_on:
                input("   [PAUSE] Vérifiez l'architecture puis Entrée...")

            # Entraînement
            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

            cw[0] *= class_weight_list[0]
            cw[1] *= class_weight_list[1]
            cw[2] *= class_weight_list[2]

            cw_dict = dict(enumerate(cw))

            cw_dict = None

            print("🔹 Class weights pli :", cw_dict)
            early = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
            model.fit(train_gen, validation_data=val_gen, epochs=20,
                      callbacks=[early], class_weight=cw_dict, verbose=1)
            
            if debug_on:    
                input("   [PAUSE] Entraînement terminé. ⇨ Entrée pour prédictions...")

            # Prédiction
            y_proba = model.predict(val_gen)
            # On reconstruit y_true de la même façon que plus bas
            y_true = np.concatenate([val_gen[i][1] for i in range(len(val_gen))])
            y_proba = y_proba[:len(y_true)]

            # Exemples de dates + prédictions
            print(f"🔹 Quelques exemples de fenêtres et dates (Fold {fold+1}):")
            for j in range(min(3, len(y_true))):
                start_date      = pd.to_datetime(date_val[j])
                end_input_date  = pd.to_datetime(date_val[j + SEQUENCE_LENGTH - 1])
                pred_date       = pd.to_datetime(date_val[j + full_seq])
                print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}, y_true={y_true[j]}")
            
            if debug_on:    
                input("   [PAUSE] Vérifiez dates des fenêtres CV puis Entrée...")

            # Seuil de confiance & y_pred
            ecart_min = 0.1
            y_pred = []
            for p in y_proba:
                top2 = np.sort(p)[::-1][:2]
                y_pred.append(np.argmax(p) if (top2[0]-top2[1]>=ecart_min) else 1)
            y_pred = np.array(y_pred)

            # Scores
            acc     = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1_m    = f1_score(y_true, y_pred, average='macro')
            f1_w    = f1_score(y_true, y_pred, average='weighted')
            print(f"   → Fold {fold+1} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | F1-macro: {f1_m:.4f} | F1-weighted: {f1_w:.4f}")
            fold_stats.append((acc, bal_acc, f1_m, f1_w))

        # Résumé CV
        accs, bals, f1s, f1w = zip(*fold_stats)
        print("\n===== RÉSULTATS CV =====")
        print(f"Acc Moyenne: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"BalAcc Moyenne: {np.mean(bals):.4f} ± {np.std(bals):.4f}")
        print(f"F1-macro Moyenne: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"F1-weighted Moyenne: {np.mean(f1w):.4f} ± {np.std(f1w):.4f}")
        
        input("   [PAUSE] Fin validation croisée. Entrée pour pipeline final...")

    # -- Pipeline final sur holdout --
    print("\n🔹 Sélection finale des features sur tout df_main")
    raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
    final_feats = [f for f in raw_feats if f not in ('target', 'ret_future')]
    if 'SP500_historical_data_Close' not in final_feats:
        final_feats.append('SP500_historical_data_Close')
    print("   → Features finales :", final_feats)
    
    if debug_on:
        input("   [PAUSE] Vérifiez les features finales puis Entrée...")

    # Préparation X/y finales
    final_scaler = StandardScaler()
    X_tr = final_scaler.fit_transform(df_main[final_feats])
    X_te = final_scaler.transform(df_holdout[final_feats])
    y_tr = df_main['target'].to_numpy()
    y_te = df_holdout['target'].to_numpy()
    date_te = df_holdout[DATE_COL].values
    print("🔹 Shapes finales X_tr, X_te, y_tr, y_te :", X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
        
    if debug_on:    
        input("   [PAUSE] Vérifiez les données finales puis Entrée...")

    # Générateurs finaux
    full_seq = SEQUENCE_LENGTH + PRED_HORIZON
    gen_tr   = TimeseriesGenerator(X_tr, y_tr, length=full_seq, batch_size=BATCH_SIZE)
    gen_te   = TimeseriesGenerator(X_te, y_te, length=full_seq, batch_size=BATCH_SIZE)
    print("🔹 Batches finaux train/holdout :", len(gen_tr), len(gen_te))
        
    if debug_on:    
        input("   [PAUSE] Vérifiez les generators finaux puis Entrée...")

    # Entraînement final
    final_model = lstm_model_v2((full_seq, len(final_feats)), SEQUENCE_LENGTH, N_CLASSES)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cw_f = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)

    cw_f[0] *= class_weight_list[0]
    cw_f[1] *= class_weight_list[1]
    cw_f[2] *= class_weight_list[2]
    
    cw_dict_f = dict(enumerate(cw_f))

    cw_dict_f = None

    print("🔹 Class weights final :", cw_dict_f)
    early_f = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    final_model.fit(gen_tr, epochs=20, callbacks=[early_f], class_weight=cw_dict_f, verbose=1)
            
    if debug_on:    
        input("   [PAUSE] Entraînement final OK. Entrée pour évaluation...")

    # Évaluation holdout
    print("\n--- Évaluation HOLDOUT ---")
    proba_final = final_model.predict(gen_te)
    y_true_final = np.concatenate([gen_te[i][1] for i in range(len(gen_te))])
    proba_final = proba_final[:len(y_true_final)]

    # Exemples de dates + prédictions (Hold-Out)
    print("🔹 Quelques exemples de fenêtres et dates (Hold-Out):")
    for j in range(min(3, len(y_true_final))):
        start_date     = pd.to_datetime(date_te[j])
        end_input_date = pd.to_datetime(date_te[j + SEQUENCE_LENGTH - 1])
        pred_date      = pd.to_datetime(date_te[j + full_seq])
        print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}, y_true={y_true_final[j]}")
        
    if debug_on:    
        input("   [PAUSE] Vérifiez dates des fenêtres HoldOut puis Entrée...")

    # Seuil confiance & y_pred_final
    ecart_min = 0.05
    y_pred_final_list = []
    for proba in proba_final:
        sorted_proba = np.sort(proba)[::-1]
        if sorted_proba[0] - sorted_proba[1] >= ecart_min:
            y_pred_final_list.append(np.argmax(proba))
        else:
            y_pred_final_list.append(1) # Classe neutre par défaut pour incertitude (à adapter, 0, 1 ou 2)
    y_pred_final = np.array(y_pred_final_list)

    # Scores finaux
    acc_f  = accuracy_score(y_true_final, y_pred_final)
    bal_f  = balanced_accuracy_score(y_true_final, y_pred_final)
    f1m_f  = f1_score(y_true_final, y_pred_final, average='macro')
    f1w_f  = f1_score(y_true_final, y_pred_final, average='weighted')
    print(f"   → Accuracy final : {acc_f:.4f}")
    print(f"   → Balanced Acc. : {bal_f:.4f}")
    print(f"   → F1-macro      : {f1m_f:.4f}")
    print(f"   → F1-weighted      : {f1w_f:.4f}")
    # print(classification_report(y_true_final, y_pred_final, zero_division=0))
    print(confusion_matrix(y_true_final, y_pred_final))

    print(f"Nombre de classe 0 prédite (y_pred_final): {(y_pred_final == 0).sum()}")
    print(f"Nombre de classe 0 réelle (y_true_final): {(y_true_final == 0).sum()}")

    print(f"Nombre de classe 1 prédite (y_pred_final): {(y_pred_final == 1).sum()}")
    print(f"Nombre de classe 1 réelle (y_true_final): {(y_true_final == 1).sum()}")

    print(f"Nombre de classe 2 prédite (y_pred_final): {(y_pred_final == 2).sum()}")
    print(f"Nombre de classe 2 réelle (y_true_final): {(y_true_final == 2).sum()}")


    if debug_on:
        input("   [PAUSE] Vérifiez le rapport de classification puis Entrée...")

    # ... (suite sauvegarde prédictions, modèle, affichages graphiques identiques)
    # Je vous laisse la partie finale de sauvegarde telle quelle, avec les prints / input déjà présents.

if __name__ == "__main__":
    cross_validation = True  # False pour pipeline complet sans CV
    debug_on = False

    print(f"\n--- Lancement pipeline version : {MODEL_VERSION} ---")
    if os.path.exists(f"Pipeline/model/model_v{MODEL_VERSION}.keras"):
        print("⚠️  Le modèle existe déjà :", f"model_v{MODEL_VERSION}.keras")
        input("   Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")

    main(cross_validation, debug_on)
