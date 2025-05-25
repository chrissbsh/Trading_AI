import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
from tensorflow.keras.models import Sequential, save_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Bidirectional, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

import pickle

os.makedirs("IA_training/model/", exist_ok=True)

# --- 1) Chargement et feature engineering ---
df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"]).sort_values("Date")
df["target_7d"] = ((df["SP500_historical_data_Close"].shift(-7) - df["SP500_historical_data_Close"]) 
                   / df["SP500_historical_data_Close"] > 0.05).astype(int)  # +5% minimum
df = df.iloc[:-7]
df.dropna(inplace=True)
exclude = {"Date", "target_7d"}
features = [c for c in df.columns if c not in exclude]

# --- Hold-out final ---
holdout = (df["Date"] >= "2023-01-03") & (df["Date"] <= "2025-04-14")
df_holdout = df.loc[holdout]
df_roll    = df.loc[~holdout]

# --- Rolling windows quarterly ---
starts = []
d = df_roll["Date"].min()
last = df_roll["Date"].max() - relativedelta(years=4) - relativedelta(months=3)
while d <= last:
    starts.append(d)
    d += relativedelta(months=3)

# Fonction pour créer les séquences
def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i-timesteps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def dynamic_undersample(X, y):
    """
    Equilibre les classes 0 et 1 par undersampling aléatoire
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    min_size = min(len(pos_indices), len(neg_indices))

    if min_size == 0:
        # Pas possible d'équilibrer
        return X, y

    pos_selected = np.random.choice(pos_indices, size=min_size, replace=False)
    neg_selected = np.random.choice(neg_indices, size=min_size, replace=False)

    selected_indices = np.concatenate([pos_selected, neg_selected])
    np.random.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]

def build_lstm_model(input_shape):
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
    model.add(Dense(1, activation='sigmoid'))  # classification binaire

    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

results = []
timesteps = 30  # Nombre de jours passés utilisés pour chaque prédiction

# Variables pour suivre le meilleur modèle
best_f1 = -1
best_model = None
best_scaler = None
best_threshold = None
best_window_info = None

for start in starts:
    tr_start, tr_end = start, start + relativedelta(years=3) - pd.Timedelta(days=1)
    val_end          = tr_end + relativedelta(years=1)
    te_start         = val_end + pd.Timedelta(days=1)
    te_end           = te_start + relativedelta(months=3) - pd.Timedelta(days=1)

    tr = df_roll[(df_roll["Date"] >= tr_start) & (df_roll["Date"] <= tr_end)]
    va = df_roll[(df_roll["Date"] >  tr_end)   & (df_roll["Date"] <= val_end)]
    te = df_roll[(df_roll["Date"] >= te_start)  & (df_roll["Date"] <= te_end)]
    if te.empty: break

    X_tr, y_tr = tr[features].values, tr["target_7d"].values
    X_va, y_va = va[features].values, va["target_7d"].values
    X_te, y_te = te[features].values, te["target_7d"].values

    # a) Mise à l'échelle des caractéristiques
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # b) Création des séquences pour LSTM
    X_tr_seq, y_tr_seq = create_sequences(X_tr_s, y_tr, timesteps)
    X_va_seq, y_va_seq = create_sequences(X_va_s, y_va, timesteps)
    X_te_seq, y_te_seq = create_sequences(X_te_s, y_te, timesteps)

    # c) Sous-échantillonnage pour équilibrer les classes dans l'entraînement
    pos_indices = np.where(y_tr_seq == 1)[0]
    neg_indices = np.where(y_tr_seq == 0)[0]
    if len(neg_indices) > len(pos_indices):
        selected_neg_indices = np.random.choice(neg_indices, size=len(pos_indices), replace=False)
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
    else:
        selected_pos_indices = np.random.choice(pos_indices, size=len(neg_indices), replace=False)
        selected_indices = np.concatenate([selected_pos_indices, neg_indices])
    # Sous-échantillonnage équilibré
    X_tr_seq_balanced, y_tr_seq_balanced = dynamic_undersample(X_tr_seq, y_tr_seq)

    noise = np.random.normal(0, 0.01, X_tr_seq_balanced.shape)
    X_augmented = X_tr_seq_balanced + noise
    y_augmented = y_tr_seq_balanced.copy()

    # Combine original + augmented
    X_train_final = np.vstack([X_tr_seq_balanced, X_augmented])
    y_train_final = np.concatenate([y_tr_seq_balanced, y_augmented])


    # d) Définition et entraînement du modèle LSTM
    model = build_lstm_model(input_shape=(timesteps, X_train_final.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    model.fit(
        X_train_final, y_train_final,
        epochs=100,
        batch_size=32,
        validation_data=(X_va_seq, y_va_seq),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # e) Prédiction sur l'ensemble de validation et optimisation du seuil
    p_va = model.predict(X_va_seq).flatten()
    ths = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_va_seq, p_va > t) for t in ths]
    t_star = ths[np.argmax(f1s)]

    # f) Prédiction sur l'ensemble de test et calcul des métriques
    p_te = model.predict(X_te_seq).flatten()
    y_pred = (p_te > t_star).astype(int)
    
    current_f1 = f1_score(y_te_seq, y_pred)
    window_info = {
        "window_start":  tr_start.date(),
        "window_test":   f"{te_start.date()}→{te_end.date()}",
        "threshold*":    t_star,
        "AUC_test":      roc_auc_score(y_te_seq, p_te),
        "F1_test":       current_f1,
        "BalAcc_test":   balanced_accuracy_score(y_te_seq, y_pred),
        "Precision_test":precision_score(y_te_seq, y_pred),
        "Recall_test":   recall_score(y_te_seq, y_pred)
    }
    results.append(window_info)
    
    # Vérifier si c'est le meilleur modèle jusqu'à présent
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model = model
        best_scaler = scaler
        best_threshold = t_star
        best_window_info = window_info

# Affichage des résultats
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# Enregistrer le meilleur modèle et ses configurations
if best_model is not None:
    print(f"\n=== Enregistrement du meilleur modèle ===")
    print(f"Meilleur F1 score: {best_f1:.4f}")
    print(f"Fenêtre d'entraînement commençant le: {best_window_info['window_start']}")
    print(f"Fenêtre de test: {best_window_info['window_test']}")
    print(f"Seuil optimal: {best_threshold:.4f}")
    
    # Sauvegarder le modèle
    save_model(best_model, "IA_training/model/best_lstm_model_v8.keras")
    
    # Sauvegarder le scaler, le seuil et les métadonnées
    model_config = {
        "scaler": best_scaler,
        "threshold": best_threshold,
        "features": features,
        "timesteps": timesteps,
        "performance": best_window_info,
    }
    
    with open("IA_training/model/model_config_v8.pkl", "wb") as f:
        pickle.dump(model_config, f)
        
    print("Modèle et configuration sauvegardés avec succès dans le dossier 'IA_training/model/'")
else:
    print("Aucun modèle n'a pu être entraîné correctement")