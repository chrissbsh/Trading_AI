import pandas as pd
import numpy as np
from config import * 
import os
import random
import optuna
import warnings
from datetime import datetime
import re
import ast

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

from optuna.trial import TrialState

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

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("✅ GPU détecté :", gpus)
#     try:
#         # Alloue la mémoire GPU de façon progressive (pas tout d’un coup)
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print("Erreur de config GPU :", e)
# else:
#     print("❌ Aucun GPU détecté. L'entraînement se fera sur CPU.")

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

def parse_previous_trials(folder_path):
    print(folder_path)
    trials = []
    pattern = r"f1_macro=([0-9.eE+-]+) \| params=({.*})"
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        score = float(match.group(1))
                        params = ast.literal_eval(match.group(2))
                        trials.append((params, score))
                        print("score: ", score)
        else:
            print("vide")

    input("entrer pour continuer")
    return trials

def create_fixed_distributions(params):
    """Crée une distribution bloquée à la valeur observée (min=max)."""
    distributions = {}
    for k, v in params.items():
        if isinstance(v, int):
            distributions[k] = optuna.distributions.IntDistribution(v, v)
        elif isinstance(v, float):
            distributions[k] = optuna.distributions.FloatDistribution(v, v)
    return distributions


def load_data(debug_on):
    print("🔹 1) Chargement des données depuis :", DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    if debug_on:
        print("   → Aperçu des 5 premières lignes :")
        print(df.head())
        input("   [PAUSE] Vérifiez le DataFrame chargé puis appuyez sur Entrée...")
    return df

def create_target(df, pred_horizon, fixed_threshold, tag=""):
    print(f"🔹 2) Création de la cible (horizon = {pred_horizon}, seuil={fixed_threshold}) {tag}")
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-pred_horizon) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)
    
    thresholds = [-fixed_threshold, fixed_threshold]
    def label_target(x):
        if x < thresholds[0]:
            return 0  # Baisse
        elif x <= thresholds[1]:
            return 1  # Neutre
        else:
            return 2  # Hausse
    df["target"] = df["ret_future"].apply(label_target)
    return df


def main(debug_on, feature_selection_on, recup_past_trials):
    # -- Chargement et split initial --
    df_raw = load_data(debug_on)
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    df_holdout_raw = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
    df_main_raw = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()

    # --- Création du fichier de log unique pour cette exécution ---
    if not os.path.exists(OPTUNA_DIR):
        os.makedirs(OPTUNA_DIR)
    date_file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(OPTUNA_DIR, f"optuna_run_{date_file_name}.txt")
    print(f"\n📝 Les résultats seront enregistrés dans : {log_filename}")


    # ========================================================================
    # =========== BOUCLE PRINCIPALE SUR LES HORIZONS DE PRÉDICTION ===========
    # ========================================================================
    
    horizons_to_test = [1] # <--- DÉFINISSEZ ICI LES HORIZONS À TESTER

    top_n_features = 25

    ecart_min = 0.05

    for current_pred_horizon in horizons_to_test:
        
        with open(log_filename, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"===== DÉBUT DE L'OPTIMISATION POUR PRED_HORIZON = {current_pred_horizon} =====\n")
            f.write("="*80 + "\n\n")

        print("\n" + "="*80)
        print(f"🚀 Lancement de l'optimisation pour PRED_HORIZON = {current_pred_horizon}")
        print("="*80)
        
        # --- Définition de la fonction Objective pour Optuna ---
        # Elle est définie à l'intérieur de la boucle pour capturer `current_pred_horizon`
        def objective(trial):
            print(f"\n▶ Trial {trial.number+1}/{N_TRIALS} pour horizon={current_pred_horizon}")
            # ==================== 1. HYPERPARAMÈTRES À TESTER ====================
            # pred_horizon est maintenant FIXE et fourni par la boucle externe
            fixed_threshold = trial.suggest_float('fixed_threshold', 0.005, 0.01)
            sequence_length = trial.suggest_int('sequence_length', 2, 120)
            # top_n_features = trial.suggest_int('top_n_features', 10, 50)
            # ecart_min = trial.suggest_float('ecart_min', 0.0, 0.2)
            
            # Paramètres du modèle
            lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
            dense_units = trial.suggest_int('dense_units', 16, 64, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            l2_reg = trial.suggest_float('l2_reg', 1e-4, 1e-1, log=True)
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.5, 5)

            # ==================== 2. PRÉPARATION DES DONNÉES (DÉPENDANTE DU TRIAL) ====================
            df_main = create_target(df_main_raw.copy(), current_pred_horizon, fixed_threshold, tag="(train)")
            
            if feature_selection_on:
                raw_feats = select_top_features_shap(df_main, top_n=top_n_features, target_col="ret_future")
                final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
                if 'SP500_historical_data_Close' not in final_feats and 'SP500_historical_data_Close' in df_main.columns:
                    final_feats.append('SP500_historical_data_Close')
            else:
                final_feats = [f for f in df_main.columns if f not in (DATE_COL, 'target', 'ret_future')]

            split_idx = int(0.8 * len(df_main))
            train_df = df_main.iloc[:split_idx]
            val_df = df_main.iloc[split_idx:]

            X_train, y_train = train_df[final_feats], train_df['target'].to_numpy()
            X_val,   y_val   = val_df[final_feats],   val_df['target'].to_numpy()
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            full_seq = sequence_length + current_pred_horizon - 1
            gen_train = TimeseriesGenerator(X_train_scaled, y_train, length=full_seq, batch_size=BATCH_SIZE)
            gen_val = TimeseriesGenerator(X_val_scaled, y_val, length=full_seq, batch_size=BATCH_SIZE)
            
            if len(gen_train) == 0 or len(gen_val) == 0:
                print("   → Skipping trial: not enough data for TimeseriesGenerator with current parameters.")
                raise optuna.exceptions.TrialPruned()
            
            # Oversampling
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

            # ==================== 3. CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE ====================
            input_shape = (full_seq, len(final_feats))
            
            model = Sequential([
                Input(shape=input_shape),
                Lambda(lambda z: z[:, :sequence_length, :], name="truncate_future"),
                LSTM(lstm_units, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)),
                Dropout(dropout_rate),
                Dense(dense_units, activation='relu'),
                Dense(N_CLASSES, activation='softmax')
            ])

            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

            loss_fn = SparseCategoricalFocalLoss(gamma=gamma, class_weight=cw)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            metrics = ['accuracy', F1Macro(N_CLASSES), BalancedAcc(N_CLASSES)]
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            early = EarlyStopping(monitor='val_f1_macro', mode='max', patience=PATIENCE, restore_best_weights=True)

            model.fit(X_train_res, y_res, validation_data=gen_val, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early], verbose=0)
            
            # ==================== 4. ÉVALUATION ====================
            proba_val = model.predict(gen_val, verbose=0)
            y_true_val = np.concatenate([gen_val[i][1] for i in range(len(gen_val))])
            proba_val = proba_val[:len(y_true_val)]

            y_pred_val = []
            for p in proba_val:
                top2 = np.sort(p)[::-1][:2]
                if top2[0] - top2[1] >= ecart_min:
                    y_pred_val.append(np.argmax(p))
                else:
                    y_pred_val.append(1) # Classe neutre par défaut
            
            score = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
            print(f"✓ Trial {trial.number+1} → f1_macro = {score:.4f}")

            with open(log_filename, "a") as f:
                f.write(f"Trial {trial.number+1:03d} | f1_macro={score:.4f} | params={trial.params}\n")
            
            return score

        # --- Lancement de l'étude Optuna pour l'horizon actuel ---
        study = optuna.create_study(direction='maximize')

        if recup_past_trials:
            # Injecter les essais passés
            optuna_dir_path = "Pipeline/optuna_results/pour_13_1_1"
            for params, score in parse_previous_trials(optuna_dir_path):
                distributions = create_fixed_distributions(params)
                trial = optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=score,
                    state=TrialState.COMPLETE
                )
                study.add_trial(trial)

            df = study.trials_dataframe()
            print(df[['number', 'value']])

            input("entrer pour continuer")


        study.optimize(objective, n_trials=N_TRIALS)
        
        best = study.best_trial.params
        best_value = study.best_value
        
        print(f"\n✨ Meilleurs hyperparamètres trouvés pour horizon={current_pred_horizon}:")
        print(best)
        print(f"   avec un F1-macro de validation de: {best_value:.4f}")

        # ==================== ENTRAÎNEMENT FINAL ET ÉVALUATION HOLDOUT POUR CET HORIZON ====================
        print("\n--- Entraînement du modèle final et évaluation sur le Holdout ---")
        
        # 1. Préparation des données avec les meilleurs hyperparamètres
        df_main_final = create_target(df_main_raw.copy(), current_pred_horizon, best['fixed_threshold'], tag="(final train)")
        df_holdout_final = create_target(df_holdout_raw.copy(), current_pred_horizon, best['fixed_threshold'], tag="(final holdout)")
        
        # --- NOUVEAU: Split Train/Val pour l'entraînement final ---
        split_idx_final = int(0.8 * len(df_main_final))
        train_df_final = df_main_final.iloc[:split_idx_final]
        val_df_final = df_main_final.iloc[split_idx_final:]
        test_df_final = df_holdout_final

        if feature_selection_on:
            raw_feats = select_top_features_shap(df_main_final, top_n=top_n_features, target_col="ret_future")
            final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
            if 'SP500_historical_data_Close' not in final_feats and 'SP500_historical_data_Close' in df_main_final.columns:
                final_feats.append('SP500_historical_data_Close')
        else:
            final_feats = [f for f in df_main_final.columns if f not in (DATE_COL, 'target', 'ret_future')]

        X_train_final, y_train_final = train_df_final[final_feats], train_df_final['target'].to_numpy()
        X_val_final, y_val_final = val_df_final[final_feats], val_df_final['target'].to_numpy()
        X_test_final, y_test_final = test_df_final[final_feats], test_df_final['target'].to_numpy()
        date_test = test_df_final[DATE_COL].values

        scaler = StandardScaler()
        X_train_scaled_final = scaler.fit_transform(X_train_final)
        X_val_scaled_final = scaler.transform(X_val_final)
        X_test_scaled_final = scaler.transform(X_test_final)
        
        full_seq = best['sequence_length'] + current_pred_horizon - 1
        gen_train_final = TimeseriesGenerator(X_train_scaled_final, y_train_final, length=full_seq, batch_size=BATCH_SIZE)
        gen_val_final = TimeseriesGenerator(X_val_scaled_final, y_val_final, length=full_seq, batch_size=BATCH_SIZE)
        gen_test_final = TimeseriesGenerator(X_test_scaled_final, y_test_final, length=full_seq, batch_size=BATCH_SIZE)

        # Oversampling sur l'ensemble d'entraînement final uniquement
        X_seq_list, y_seq_list = [], []
        for i in range(len(gen_train_final)):
            x, y = gen_train_final[i]
            X_seq_list.append(x); y_seq_list.append(y)
        X_seqs = np.concatenate(X_seq_list); y_seqs = np.concatenate(y_seq_list)
        n_samples, seq_len, n_features = X_seqs.shape
        X_flat = X_seqs.reshape(n_samples, seq_len * n_features)
        ros = RandomOverSampler(random_state=SEED)
        X_res_flat, y_res = ros.fit_resample(X_flat, y_seqs)
        X_train_res_final = X_res_flat.reshape(-1, seq_len, n_features)
        
        # 2. Construction du modèle final
        input_shape = (full_seq, len(final_feats))
        
        final_model = Sequential([
            Input(shape=input_shape),
            Lambda(lambda z: z[:, :best['sequence_length'], :], name="truncate_future"),
            LSTM(best['lstm_units'], return_sequences=False, kernel_regularizer=regularizers.l2(best['l2_reg'])),
            Dropout(best['dropout_rate']),
            Dense(best['dense_units'], activation='relu'),
            Dense(N_CLASSES, activation='softmax')
        ])

        cw_f = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)

        loss_fn = SparseCategoricalFocalLoss(gamma=best['gamma'], class_weight=cw_f)
        optimizer = tf.keras.optimizers.Adam(learning_rate=best['lr'])
        metrics = ['accuracy', F1Macro(N_CLASSES), BalancedAcc(N_CLASSES)]
        final_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        
        # --- MODIFIÉ: EarlyStopping sur l'ensemble de validation ---
        early_stop_final = EarlyStopping(
            monitor='val_f1_macro', 
            mode='max',
            patience=PATIENCE, 
            restore_best_weights=True
        )

        # --- MODIFIÉ: Entraînement avec l'ensemble de validation ---
        final_model.fit(
            X_train_res_final, y_res, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            verbose=1,
            validation_data=gen_val_final,
            callbacks=[early_stop_final]
        )

        # 3. Évaluation finale sur le Holdout (gen_test_final)
        proba_final = final_model.predict(gen_test_final)
        y_true_final = np.concatenate([gen_test_final[i][1] for i in range(len(gen_test_final))])
        proba_final = proba_final[:len(y_true_final)]

        y_pred_final = []
        for proba in proba_final:
            top2 = np.sort(proba)[::-1][:2]
            if top2[0] - top2[1] >= ecart_min:
                y_pred_final.append(np.argmax(proba))
            else:
                y_pred_final.append(1)
        y_pred_final = np.array(y_pred_final)

        # 4. Sauvegarde et rapport
        if not os.path.exists(PREDICTION_SAVE_DIR):
            os.makedirs(PREDICTION_SAVE_DIR)
            
        proba_df = pd.DataFrame(proba_final, columns=[f"proba_class_{i}" for i in range(proba_final.shape[1])])
        proba_df["y_pred"] = y_pred_final
        proba_df["y_true"] = y_true_final
        proba_df["date"] = date_test[full_seq:]
        proba_df.reset_index(drop=True, inplace=True)
        output_csv_path = os.path.join(PREDICTION_SAVE_DIR, f"predictions_optuna_run_{date_file_name}_{current_pred_horizon}_jours.csv")
        proba_df.to_csv(output_csv_path, index=False)
        print(f"\n🔹 Fichier CSV des prédictions sauvegardé : {output_csv_path}")

        # 4. Sauvegarde des résultats et rapport dans le fichier de log
        acc_f = accuracy_score(y_true_final, y_pred_final)
        bal_f = balanced_accuracy_score(y_true_final, y_pred_final)
        f1m_f = f1_score(y_true_final, y_pred_final, average='macro', zero_division=0)
        f1w_f = f1_score(y_true_final, y_pred_final, average='weighted')
        report_str = classification_report(y_true_final, y_pred_final, zero_division=0, digits=3)
        conf_matrix_str = str(confusion_matrix(y_true_final, y_pred_final))

        print("\n--- RÉSULTATS FINALS SUR HOLDOUT ---")
        print(f"   → Accuracy final : {acc_f:.4f}")
        print(f"   → Balanced Acc.  : {bal_f:.4f}")
        print(f"   → F1-macro       : {f1m_f:.4f}")
        print(f"   → F1-weighted    : {f1w_f:.4f}")
        print("\n" + report_str)
        print("Confusion Matrix:")
        print(conf_matrix_str)
        
        # Écriture des résultats finaux dans le fichier de log
        with open(log_filename, "a") as f:
            f.write("\n" + "-"*30 + "\n")
            f.write(f"RÉSUMÉ FINAL POUR PRED_HORIZON = {current_pred_horizon}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Meilleur F1-Macro (validation): {best_value:.4f}\n")
            f.write(f"Meilleurs hyperparamètres: {best}\n\n")
            f.write("--- RÉSULTATS SUR L'ENSEMBLE DE TEST (HOLDOUT) ---\n")
            f.write(f"Accuracy: {acc_f:.4f}\n")
            f.write(f"Balanced Accuracy: {bal_f:.4f}\n")
            f.write(f"F1-Macro: {f1m_f:.4f}\n")
            f.write(f"F1-Weighted: {f1w_f:.4f}\n\n")
            f.write("Rapport de classification:\n")
            f.write(report_str + "\n\n")
            f.write("Matrice de confusion:\n")
            f.write(conf_matrix_str + "\n")
            f.write("\n" + "="*80 + "\n")

    print(f"\n✅ Pipeline terminé. Tous les résultats sont dans {log_filename}")


if __name__ == "__main__":
    debug_on = False
    feature_selection_on = True
    recup_past_trials = False

    print(f"\n--- Lancement pipeline version : {MODEL_VERSION} ---")
    
    # Création des dossiers nécessaires s'ils n'existent pas
    if not os.path.exists('Pipeline/model'):
        os.makedirs('Pipeline/model')
        
    main(debug_on, feature_selection_on, recup_past_trials)