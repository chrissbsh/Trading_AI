import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from tensorflow.keras.models import Sequential, save_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Bidirectional, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import pickle

os.makedirs("IA_training/model/", exist_ok=True)

def label_change(x):
    if x <= -0.05:
        return 0
    elif x <= -0.01:
        return 1
    elif x <= 0.01:
        return 2
    elif x <= 0.05:
        return 3
    else:
        return 4

df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"]).sort_values("Date")
change = (df["SP500_historical_data_Close"].shift(-7) - df["SP500_historical_data_Close"]) / df["SP500_historical_data_Close"]
df["target_multi"] = change.apply(label_change)
df = df.iloc[:-7]

df["std_21"] = df["sp500_return_1d"].rolling(21).std()
df["hv_30"]  = df["sp500_return_1d"].rolling(30).std()
df["r_sp_gold"] = df["SP500_historical_data_Close"] / df["gold_historical_data_Close"]
df["r_sp_dxy"]  = df["SP500_historical_data_Close"] / df["dollar_index_historical_data_Close"]
df["r_sp_bond"] = df["SP500_historical_data_Close"] / df["Market_yield_US_10_year_DGS10"]
vix = df["^VIX_historical_data_Close"]
df["vix_direction"] = vix.diff().fillna(0).gt(0).astype(int)
df["vix_high"] = vix.gt(vix.rolling(63).median()).astype(int)
if "PMI" in df.columns:
    df["macro_regime"] = (df["PMI"] > 50).astype(int)
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
    indices = np.concatenate([np.random.choice(np.where(y == c)[0], min_size, replace=False) for c in classes])
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

    X_tr_seq_balanced, y_tr_seq_balanced = dynamic_undersample(X_tr_seq, y_tr_seq)
    noise = np.random.normal(0, 0.01, X_tr_seq_balanced.shape)
    X_augmented = X_tr_seq_balanced + noise
    y_augmented = y_tr_seq_balanced.copy()

    X_train_final = np.vstack([X_tr_seq_balanced, X_augmented])
    y_train_final = np.concatenate([y_tr_seq_balanced, y_augmented])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = build_multiclass_lstm_model(input_shape=(timesteps, X_train_final.shape[2]))

    model.fit(
        X_train_final, y_train_final,
        epochs=100,
        batch_size=32,
        validation_data=(X_va_seq, y_va_seq),
        callbacks=[early_stop],
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

p_te = model.predict(X_te_seq)
y_pred = np.argmax(p_te, axis=1)

print("\n=== Evaluation multi-class ===")
print(classification_report(y_te_seq, y_pred, digits=4))
print("\nConfusion Matrix :")
print(confusion_matrix(y_te_seq, y_pred))

if best_model is not None:
    print(f"\n=== Enregistrement du meilleur modèle ===")
    print(f"Meilleur F1 score: {best_f1:.4f}")
    print(f"Fenêtre d'entraînement: {best_window_info['window_start']}")
    print(f"Fenêtre de test: {best_window_info['window_test']}")

    save_model(best_model, "IA_training/model/best_lstm_model_v9.keras")
    model_config = {
        "scaler": best_scaler,
        "features": features,
        "timesteps": timesteps,
        "performance": best_window_info
    }
    with open("IA_training/model/model_config_v9.pkl", "wb") as f:
        pickle.dump(model_config, f)

    print("Modèle et configuration sauvegardés avec succès.")
else:
    print("Aucun modèle n'a pu être entraîné correctement.")
