import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, save_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Bidirectional, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import pickle

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)


os.makedirs("IA_training/model/", exist_ok=True)

def label_change(x):
    if x <= -0.043:
        return 0  # forte baisse
    elif x <= -0.009:
        return 1  # faible baisse
    elif x <= 0.017:
        return 2  # stable
    elif x <= 0.041:
        return 3  # faible hausse
    else:
        return 4  # forte hausse

df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"]).sort_values("Date")
change = (df["SP500_historical_data_Close"].shift(-7) - df["SP500_historical_data_Close"]) / df["SP500_historical_data_Close"]
df["target_multi"] = change.apply(label_change)
df = df.iloc[:-7]

df.dropna(inplace=True)

exclude = {"Date", "target_multi"}
features = [c for c in df.columns if c not in exclude]

holdout = (df["Date"] >= "2023-01-03") & (df["Date"] <= "2025-04-14")
df_holdout = df.loc[holdout]
df_roll = df.loc[~holdout]

starts = []
d = df_roll["Date"].min()
last = df_roll["Date"].max() - relativedelta(years=4) - relativedelta(months=3)
while d <= last:
    starts.append(d)
    d += relativedelta(months=3)

def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def dynamic_undersample(X, y):
    classes = np.unique(y)
    min_size = min([np.sum(y == c) for c in classes])
    if min_size == 0:
        return X, y
    indices = np.concatenate([
        np.random.choice(np.where(y == c)[0], min_size, replace=False)
        for c in classes
    ])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def build_multiclass_lstm_model(input_shape, num_classes=5):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LayerNormalization())

    model.add(Bidirectional(LSTM(32)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LayerNormalization())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

results = []
timesteps = 30
best_f1 = -1
best_model = None
best_scaler = None
best_window_info = None

for start in starts:
    tr_start, tr_end = start, start + relativedelta(years=3) - pd.Timedelta(days=1)
    val_end = tr_end + relativedelta(years=1)
    te_start = val_end + pd.Timedelta(days=1)
    te_end = te_start + relativedelta(months=3) - pd.Timedelta(days=1)

    tr = df_roll[(df_roll["Date"] >= tr_start) & (df_roll["Date"] <= tr_end)]
    va = df_roll[(df_roll["Date"] > tr_end) & (df_roll["Date"] <= val_end)]
    te = df_roll[(df_roll["Date"] >= te_start) & (df_roll["Date"] <= te_end)]
    if te.empty: break

    X_tr, y_tr = tr[features].values, tr["target_multi"].values
    X_va, y_va = va[features].values, va["target_multi"].values
    X_te, y_te = te[features].values, te["target_multi"].values

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    X_tr_seq, y_tr_seq = create_sequences(X_tr_s, y_tr, timesteps)
    X_va_seq, y_va_seq = create_sequences(X_va_s, y_va, timesteps)
    X_te_seq, y_te_seq = create_sequences(X_te_s, y_te, timesteps)

    # X_tr_seq_bal, y_tr_seq_bal = dynamic_undersample(X_tr_seq, y_tr_seq)

    X_tr_seq_bal = X_tr_seq.copy()
    y_tr_seq_bal = y_tr_seq.copy()

    noise = np.random.normal(0, 0.01, X_tr_seq_bal.shape)
    X_aug = X_tr_seq_bal + noise
    y_aug = y_tr_seq_bal.copy()

    X_final = np.vstack([X_tr_seq_bal, X_aug])
    y_final = np.concatenate([y_tr_seq_bal, y_aug])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_final), y=y_final)
    class_weights = dict(zip(np.unique(y_final), class_weights))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = build_multiclass_lstm_model(input_shape=(timesteps, X_final.shape[2]))

    model.fit(
        X_final, y_final,
        epochs=100,
        batch_size=32,
        validation_data=(X_va_seq, y_va_seq),
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=0
    )

    p_va = model.predict(X_va_seq)
    y_pred_va = np.argmax(p_va, axis=1)
    current_f1 = f1_score(y_va_seq, y_pred_va, average='weighted')

    p_te = model.predict(X_te_seq)
    y_pred_te = np.argmax(p_te, axis=1)

    window_info = {
        "window_start": tr_start.date(),
        "window_test": f"{te_start.date()}→{te_end.date()}",
        "F1_test": f1_score(y_te_seq, y_pred_te, average='weighted'),
        "BalAcc_test": balanced_accuracy_score(y_te_seq, y_pred_te),
        "Precision_test": precision_score(y_te_seq, y_pred_te, average='weighted'),
        "Recall_test": recall_score(y_te_seq, y_pred_te, average='weighted')
    }
    results.append(window_info)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model = model
        best_scaler = scaler
        best_window_info = window_info

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

if best_model is not None:
    print(f"\n=== Enregistrement du meilleur modèle ===")
    print(f"Meilleur F1 score: {best_f1:.4f}")
    print(f"Fenêtre d'entraînement: {best_window_info['window_start']}")
    print(f"Fenêtre de test: {best_window_info['window_test']}")

    save_model(best_model, "IA_training/model/best_lstm_model_v10.keras")
    model_config = {
        "scaler": best_scaler,
        "features": features,
        "timesteps": timesteps,
        "performance": best_window_info
    }
    with open("IA_training/model/model_config_v10.pkl", "wb") as f:
        pickle.dump(model_config, f)

    print("Modèle et configuration sauvegardés avec succès.")
    print(f"Model performance on test set: F1 = {best_window_info['F1_test']:.4f}, Balanced Accuracy = {best_window_info['BalAcc_test']:.4f}")
else:
    print("Aucun modèle n'a pu être entraîné correctement")