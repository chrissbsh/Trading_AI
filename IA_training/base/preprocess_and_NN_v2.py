import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import timedelta

# Charger les données
df_sp500 = pd.read_csv('csv_data/stock_prices/SP500_historical_data.csv')
df_indicators = pd.read_csv('csv_data/consolidated_data_processed.csv')

# Convertir les colonnes Date en datetime
df_sp500['Date'] = pd.to_datetime(df_sp500['Date'], utc=True)
df_sp500['date'] = df_sp500['Date'].dt.date

df_indicators['Date'] = pd.to_datetime(df_indicators['Date'])
df_indicators['date'] = df_indicators['Date'].dt.date

# Agréger df_indicators au niveau quotidien
df_indicators_daily = df_indicators.sort_values('Date').groupby('date').last().reset_index()

# Vérifier les colonnes et types de df_indicators_daily
print("Colonnes de df_indicators_daily:", df_indicators_daily.columns)
print("Types des colonnes de df_indicators_daily:\n", df_indicators_daily.dtypes)

# Fusionner df_sp500 avec df_indicators_daily
df_merged = pd.merge(df_sp500[['date', 'Close']], df_indicators_daily, on='date', how='inner')
df_merged = df_merged.sort_values('date')

# Convertir la colonne 'date' en datetime pour faciliter les manipulations
df_merged['date'] = pd.to_datetime(df_merged['date'])

# Créer des caractéristiques retardées pour Close
M = 30
for i in range(1, M+1):
    df_merged[f'Close_lag{i}'] = df_merged['Close'].shift(i)

# Créer les variables cibles
for i in range(1, 8):
    df_merged[f'target{i}'] = df_merged['Close'].shift(-i)

# Supprimer les lignes avec des valeurs NaN
df_merged = df_merged.dropna()

# Sélectionner les caractéristiques (uniquement numériques)
numeric_cols = [col for col in df_indicators_daily.columns if col not in ['date', 'Date'] and df_indicators_daily[col].dtype in [np.float64, np.int64]]
features = [f'Close_lag{i}' for i in range(1, M+1)] + numeric_cols
targets = [f'target{i}' for i in range(1, 8)]

X = df_merged[features]
y = df_merged[targets]

# Vérifier les types de X
print("Colonnes de X:", X.columns)
print("Types des colonnes de X:\n", X.dtypes)

# Normaliser les caractéristiques et les cibles
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Diviser en ensembles d'entraînement et de test
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Définir le modèle de réseau de neurones
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(7)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Évaluer le modèle
loss = model.evaluate(X_test, y_test)
print(f'Perte sur l\'ensemble de test : {loss}')

# Prédire pour la semaine prochaine
last_X = X_scaled[-1].reshape(1, -1)
prediction_scaled = model.predict(last_X)
prediction = scaler_y.inverse_transform(prediction_scaled)
print(f'Prix de clôture prédits pour les 7 prochains jours : {prediction}')

# --- Visualisation ---
# Extraire les 3 dernières semaines (21 jours) de données historiques
last_date = df_merged['date'].iloc[-1]
three_weeks_ago = last_date - timedelta(days=21)
historical_data = df_merged[df_merged['date'] >= three_weeks_ago][['date', 'Close']]

# Créer les dates pour les prédictions (7 jours après la dernière date)
pred_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
pred_values = prediction.flatten()

# Tracer le graphique
plt.figure(figsize=(12, 6))
# Données historiques (3 semaines)
plt.plot(historical_data['date'], historical_data['Close'], label='Cours réel (3 dernières semaines)', color='blue')
# Prédictions
plt.plot(pred_dates, pred_values, label='Cours prédits (7 jours)', color='red', linestyle='--')
plt.title('Cours réels et prédits du S&P 500')
plt.xlabel('Date')
plt.ylabel('Prix de clôture')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()