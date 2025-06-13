import pandas as pd
import numpy as np
from config import * 
import os
import random
import optuna
import warnings
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # 0 = tout, 1 = INFO, 2 = WARNING, 3 = ERROR
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')            # supprime les dépréciations tf.placeholder, NodeDef, etc.

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

from imblearn.over_sampling import RandomOverSampler

from focal_loss import SparseCategoricalFocalLoss

# ==================== REPRODUCTIBILITÉ ====================

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

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
    df.dropna(subset=['ret_future'], inplace=True)
    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2
        df["target"] = df["ret_future"].apply(label_target)
    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}")
    return df

def main(debug_on, feature_selection_on, fine_tune_on=False):
    # -- Chargement et split --
    df_raw = load_data(debug_on)
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
    df_main    = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()

    df_main    = create_target(df_main, tag="(train)", debug_on=debug_on)
    df_holdout = create_target(df_holdout, tag="(holdout)", debug_on=debug_on)

    # Affichage simple de la répartition des classes (en pourcentage)
    print("\nRépartition des classes (en pourcentage) :")
    for name, df in [("Train/Val", df_main), ("Holdout", df_holdout)]:
        counts = df["target"].value_counts(normalize=True).sort_index() * 100
        print(f"  {name} :")
        for cls, pct in counts.items():
            print(f"    Classe {cls}: {pct:.2f}%")

    if feature_selection_on:
        raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
        final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
        if 'SP500_historical_data_Close' not in final_feats:
            final_feats.append('SP500_historical_data_Close')
    else:
        final_feats = [f for f in df_main.columns if f not in ('target', 'ret_future')]

    # 1. Division des données
    split_idx = int(0.8 * len(df_main))
    train_df = df_main.iloc[:split_idx]
    val_df = df_main.iloc[split_idx:]
    test_df = df_holdout

    # 2. Préparation X/y + Scaling
    X_train = train_df[final_feats]; y_train = train_df['target'].to_numpy()
    X_val   = val_df[final_feats];   y_val   = val_df['target'].to_numpy()
    X_test  = test_df[final_feats];  y_test  = test_df['target'].to_numpy()
    date_test = test_df[DATE_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # 3. TimeseriesGenerators
    full_seq = SEQUENCE_LENGTH + PRED_HORIZON
    gen_train = TimeseriesGenerator(X_train_scaled, y_train, length=full_seq, batch_size=BATCH_SIZE)
    gen_val   = TimeseriesGenerator(X_val_scaled,   y_val,   length=full_seq, batch_size=BATCH_SIZE)
    gen_test  = TimeseriesGenerator(X_test_scaled,  y_test,  length=full_seq, batch_size=BATCH_SIZE)

    # 4. Oversampling
    X_seq_list, y_seq_list = [], []
    for i in range(len(gen_train)):
        x, y = gen_train[i]
        X_seq_list.append(x); y_seq_list.append(y)
    X_seqs = np.concatenate(X_seq_list); y_seqs = np.concatenate(y_seq_list)
    n_samples, seq_len, n_features = X_seqs.shape
    X_flat = X_seqs.reshape(n_samples, seq_len * n_features)
    ros = RandomOverSampler(random_state=SEED)
    X_res_flat, y_res = ros.fit_resample(X_flat, y_seqs)
    X_train_res = X_res_flat.reshape(-1, seq_len, n_features)

    # Métriques et constantes communes
    metrics = ['accuracy', F1Macro(3), BalancedAcc(3)]
    input_shape = (full_seq, len(final_feats))
    seq_len_keep = SEQUENCE_LENGTH
    n_classes = N_CLASSES

    log_filename = f"{OPTUNA_DIR}/optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # --- Fine-tuning Optuna ---
    if fine_tune_on:
        def objective(trial):
            print(f"▶ Trial {trial.number+1}/{N_TRIALS}")
            # Suggestions d'hyperparamètres
            units        = trial.suggest_int('lstm_units',   32, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            l2_reg       = trial.suggest_float('l2_reg',       1e-4, 1e-1, log=True)
            lr           = trial.suggest_float('lr',      1e-5, 1e-2, log=True)
            gamma        = trial.suggest_int('gamma',          1, 10)

            # Construction du modèle
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Lambda(lambda z: z[:, :seq_len_keep, :], name="truncate_future"))
            model.add(LSTM(units, return_sequences=False,
                           kernel_regularizer=regularizers.l2(l2_reg)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(n_classes, activation='softmax'))

            loss_fn   = SparseCategoricalFocalLoss(gamma=gamma)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            early = EarlyStopping(monitor='val_f1_macro', mode='max',
                                  patience=PATIENCE, restore_best_weights=True)

            # Entraînement silencieux
            model.fit(X_train_res, y_res,
                      validation_data=gen_val,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=[early],
                      verbose=0)

            # Évaluation F1-macro sur validation
            proba_val = model.predict(gen_val)
            y_true_val = np.concatenate([gen_val[i][1] for i in range(len(gen_val))])
            proba_val = proba_val[:len(y_true_val)]

            # Seuil de confiance
            y_pred_val = []
            for p in proba_val:
                top2 = np.sort(p)[::-1][:2]
                if top2[0] - top2[1] >= ECART_MIN:
                    y_pred_val.append(np.argmax(p))
                else:
                    y_pred_val.append(1)
            score = f1_score(y_true_val, y_pred_val, average='macro')
            print(f"✓ Trial {trial.number+1} → f1_macro = {score:.4f}")
            
            # Enregistrement dans le fichier
            with open(log_filename, "a") as f:
                f.write(f"Trial {trial.number+1} | "
                        f"units={units}, dropout={round(dropout_rate, 3)}, l2={round(l2_reg, 6)}, "
                        f"lr={round(lr, 6)}, gamma={gamma} → f1_macro={round(score, 3)}\n")
                
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        print("✨ Best hyperparameters :", study.best_trial.params)

        input("Entrer pour continuer")

        # Reconstruction du modèle final avec les meilleurs paramètres
        best = study.best_trial.params
        final_model = Sequential()
        final_model.add(Input(shape=input_shape))
        final_model.add(Lambda(lambda z: z[:, :seq_len_keep, :], name="truncate_future"))
        final_model.add(LSTM(best['lstm_units'], return_sequences=False,
                             kernel_regularizer=regularizers.l2(best['l2_reg'])))
        final_model.add(Dropout(best['dropout_rate']))
        final_model.add(Dense(32, activation='relu'))
        final_model.add(Dense(n_classes, activation='softmax'))
        loss_fn   = SparseCategoricalFocalLoss(gamma=best['gamma'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=best['lr'])
        final_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    else:
        # Modèle standard
        final_model = lstm_model_v2(input_shape, seq_len_keep, n_classes)
        loss_fn = SparseCategoricalFocalLoss(gamma=5)
        final_model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)


    # Early stopping et entraînement final
    early_stop = EarlyStopping(
        monitor='val_f1_macro',
        mode='max',
        patience=PATIENCE,
        restore_best_weights=True)

    final_model.fit(
        X_train_res, y_res,
        batch_size=BATCH_SIZE,
        validation_data=gen_val,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=0)

    # --- Évaluation sur le Holdout ---
    proba_final = final_model.predict(gen_test)
    y_true_final = np.concatenate([gen_test[i][1] for i in range(len(gen_test))])
    proba_final = proba_final[:len(y_true_final)]

    # Application du seuil de confiance
    y_pred_final = []
    for proba in proba_final:
        top2 = np.sort(proba)[::-1][:2]
        if top2[0] - top2[1] >= ECART_MIN:
            y_pred_final.append(np.argmax(proba))
        else:
            y_pred_final.append(1)
    y_pred_final = np.array(y_pred_final)

    # Sauvegarde et rapport
    proba_df = pd.DataFrame(proba_final, columns=[f"proba_class_{i}" for i in range(proba_final.shape[1])])
    proba_df["y_pred"] = y_pred_final
    proba_df["y_true"] = y_true_final
    proba_df["date"]   = date_test[full_seq:]
    proba_df.reset_index(drop=True, inplace=True)
    output_csv_path = os.path.join(PREDICTION_SAVE_DIR, f"predictions_v{MODEL_VERSION}.csv")
    proba_df.to_csv(output_csv_path, index=False)
    print(f"🔹 Fichier CSV sauvegardé : {output_csv_path}")

    acc_f  = accuracy_score(y_true_final, y_pred_final)
    bal_f  = balanced_accuracy_score(y_true_final, y_pred_final)
    f1m_f  = f1_score(y_true_final, y_pred_final, average='macro')
    f1w_f  = f1_score(y_true_final, y_pred_final, average='weighted')
    print(f"   → Accuracy final : {acc_f:.4f}")
    print(f"   → Balanced Acc. : {bal_f:.4f}")
    print(f"   → F1-macro      : {f1m_f:.4f}")
    print(f"   → F1-weighted   : {f1w_f:.4f}")

    print(f"Nombre de classe 0 prédite (y_pred_final): {(y_pred_final == 0).sum()}")
    print(f"Nombre de classe 0 réelle (y_true_final): {(y_true_final == 0).sum()}")

    print(f"Nombre de classe 1 prédite (y_pred_final): {(y_pred_final == 1).sum()}")
    print(f"Nombre de classe 1 réelle (y_true_final): {(y_true_final == 1).sum()}")

    print(f"Nombre de classe 2 prédite (y_pred_final): {(y_pred_final == 2).sum()}")
    print(f"Nombre de classe 2 réelle (y_true_final): {(y_true_final == 2).sum()}")

    print("\n" + classification_report(y_true_final, y_pred_final, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_final, y_pred_final))


if __name__ == "__main__":
    debug_on = False
    feature_selection_on = True
    fine_tune_on = True
    ECART_MIN = 0.05

    print(f"\n--- Lancement pipeline version : {MODEL_VERSION} ---")
    if os.path.exists(f"Pipeline/model/model_v{MODEL_VERSION}.keras"):
        print("⚠️  Le modèle existe déjà :", f"model_v{MODEL_VERSION}.keras")
        input("   Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")

    main(debug_on, feature_selection_on, fine_tune_on)