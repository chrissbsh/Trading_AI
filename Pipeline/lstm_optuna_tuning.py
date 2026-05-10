import os
import sys
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import random
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from focal_loss import SparseCategoricalFocalLoss
from pipeline.config import (
    DATA_FILE_PATH, DATE_COL, TARGET_PRICE_COL,
    HOLDOUT_START_DATE, TOP_N_FEATURES, N_CLASSES,
    PRED_HORIZON, FIXED_THRESHOLDS, STRIDE, BATCH_SIZE, CLASS_WEIGHT_BOOST
)
from pipeline.feature_selection import select_top_features_shap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

def create_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_root = os.path.join(project_root, "results")
    run_id = datetime.now().strftime("lstm_optuna_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def build_model(trial, input_shape, n_classes):
    model_type = trial.suggest_categorical("model_type", ["LSTM", "GRU", "BiLSTM"])
    units = trial.suggest_categorical("units", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == "LSTM":
        model.add(LSTM(units, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)))
    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(units // 2, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))))
        
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Optional dense layer
    use_dense = trial.suggest_categorical("use_dense", [True, False])
    if use_dense:
        dense_units = trial.suggest_categorical("dense_units", [16, 32, 64])
        model.add(Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(n_classes, activation='softmax'))
    
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
    
    return model, lr, gamma

def main():
    results_dir = create_results_dir()
    print(f"[*] Dossier de résultats Optuna : {results_dir}")
    
    # 1. Chargement et nettoyage
    df_raw = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    df_main = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    
    # 2. Target Engineering
    df_main['ret_future'] = (df_main[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df_main[TARGET_PRICE_COL]) / df_main[TARGET_PRICE_COL]
    df_main.dropna(subset=['ret_future'], inplace=True)
    lo, hi = FIXED_THRESHOLDS
    df_main["target"] = df_main["ret_future"].apply(lambda x: 0 if x < lo else (2 if x > hi else 1))
    
    # 3. Train/Val Split chronologique (80/20)
    split_idx = int(0.8 * len(df_main))
    train_df = df_main.iloc[:split_idx]
    val_df = df_main.iloc[split_idx:]
    
    # 4. Feature Selection sur le train
    print("[*] Sélection des features via SHAP...")
    raw_feats = select_top_features_shap(train_df, top_n=TOP_N_FEATURES, target_col="ret_future")
    final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
    if TARGET_PRICE_COL not in final_feats:
        final_feats.append(TARGET_PRICE_COL)
    print(f"   → Features retenues : {len(final_feats)}")
    
    # 5. Extraction X, y et normalisation
    X_train = train_df[final_feats].values
    y_train = train_df['target'].values
    X_val = val_df[final_feats].values
    y_val = val_df['target'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 6. Class weights
    cw_raw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = np.array([cw_raw[c] * CLASS_WEIGHT_BOOST.get(c, 1.0) for c in range(N_CLASSES)])
    print(f"[*] Poids de classes utilisés : {cw}")

    def objective(trial):
        seq_len = trial.suggest_categorical("sequence_length", [30, 45, 60, 90])
        
        gen_train = TimeseriesGenerator(X_train_scaled, y_train, length=seq_len, batch_size=BATCH_SIZE, stride=STRIDE)
        gen_val = TimeseriesGenerator(X_val_scaled, y_val, length=seq_len, batch_size=BATCH_SIZE, stride=STRIDE)
        
        model, lr, gamma = build_model(trial, (seq_len, len(final_feats)), N_CLASSES)
        
        loss_fn = SparseCategoricalFocalLoss(gamma=gamma, class_weight=cw)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=loss_fn, metrics=['accuracy'])
                      
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        
        try:
            model.fit(gen_train, validation_data=gen_val, epochs=50, 
                      callbacks=[early_stop, reduce_lr], verbose=0)
                      
            # Eval custom macro F1
            y_proba = model.predict(gen_val, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            y_true_val = np.concatenate([gen_val[i][1] for i in range(len(gen_val))])
            
            f1_m = f1_score(y_true_val, y_pred, average='macro', zero_division=0)
            
            # Pénaliser sévèrement le class collapse
            r0 = float((y_pred[y_true_val == 0] == 0).mean()) if (y_true_val == 0).sum() > 0 else 0
            r1 = float((y_pred[y_true_val == 1] == 1).mean()) if (y_true_val == 1).sum() > 0 else 0
            r2 = float((y_pred[y_true_val == 2] == 2).mean()) if (y_true_val == 2).sum() > 0 else 0
            
            if min(r0, r1, r2) < 0.05:
                f1_m -= 0.10 # Penalty for class collapse
                
            return f1_m
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    print("\n[*] Lancement de l'optimisation Optuna (20 trials)...")
    study.optimize(objective, n_trials=20)

    print("\n[*] Meilleurs hyperparamètres :")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"[*] Meilleur F1-Macro validé: {study.best_value:.4f}")

    # Save results
    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(results_dir, "optuna_trials.csv"), index=False)
    
if __name__ == "__main__":
    main()
