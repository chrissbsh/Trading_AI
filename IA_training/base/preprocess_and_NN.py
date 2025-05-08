# === 1. Prétraitement des données ===

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# 1.1 Chargement
df_feat = pd.read_csv('csv_data/consolidated_data_processed.csv')
df_sp   = pd.read_csv(
    'csv_data/stock_prices/SP500_historical_data.csv',
    parse_dates=['Date'],
    infer_datetime_format=True
)

# 1.2 Normalisation au niveau jour
df_feat['Date'] = pd.to_datetime(df_feat['Date'], errors='coerce').dt.normalize()
df_sp['Date'] = (
    pd.to_datetime(df_sp['Date'], utc=True, errors='coerce')
      .dt.tz_convert(None)
      .dt.normalize()
)

# 1.3 Fusion
df = (
    pd.merge(
        df_sp[['Date', 'Close']],
        df_feat,
        left_on='Date',
        right_on='Date',
        how='inner'
    )
    .sort_values('Date')
    .drop(columns=['Date'])
)

# on n'a plus besoin de la colonne 'date' doublon
df.drop(columns=['Date'], inplace=True)

# 1.4 Remplissage et nettoyage
df.fillna(method='ffill', inplace=True)   # forward-fill
df.dropna(inplace=True)

# 1.5 Séparation features / cible
feature_cols = [c for c in df.columns if c not in ['Date','Close']]
X_raw = df[feature_cols].values
y_raw = df[['Close']].values

# 1.6 Mise à l’échelle (Standardization)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 1.7 Création des séquences glissantes
def create_sequences(X, y, seq_len=60, horizon=5):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        Xs.append(X[i : i+seq_len])
        ys.append(y[i+seq_len : i+seq_len+horizon].flatten())
    return np.array(Xs), np.array(ys)

SEQ_LEN = 60    # nombre de jours passés
HORIZON = 5     # prévoir 5 jours (une semaine de bourse)
X, y = create_sequences(X_scaled, y_scaled, SEQ_LEN, HORIZON)

# 1.8 Split entraînement / test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train shape: {X_train.shape} → {y_train.shape}")
print(f"Test  shape: {X_test.shape} → {y_test.shape}")

# === 2. Modélisation LSTM et prédiction ===

# 2.1 Définition du modèle
model = Sequential([
    LSTM(64,  return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(HORIZON)   # une unité par jour prédit
])
model.compile(optimizer='adam', loss='mse')

# 2.2 Entraînement
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

# 2.3 Évaluation sur le set de test
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f}")

# 2.4 Prédiction pour la semaine suivante
# on prend la dernière séquence connue
last_seq = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_scaled.shape[1])
pred_scaled = model.predict(last_seq)

# remise à l’échelle du prix
pred = scaler_y.inverse_transform(pred_scaled).flatten()
print("Cours S&P 500 prévus pour les 5 prochains jours :", pred)