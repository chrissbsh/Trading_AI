import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError # type: ignore

import json

train_model = False  # Changer à True pour entraîner le modèle

# Chargement des données
df = pd.read_csv("consolidated_data.csv", parse_dates=["Date"], index_col="Date")

# Création de la cible et suppression des valeurs NaN
df['Target'] = df['AAPL_Close'].shift(-7)
df['Target_Binary'] = (df['AAPL_Close'].shift(-7) > df['AAPL_Close']).astype(int)
df = df.dropna()

# Affichage des dernières lignes
print("Dernières lignes des données originales :")
print(df[['AAPL_Close', 'Target']].tail())

# Découpage des données en train/test
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Préparation des jeux de données
X_train = df_train.drop(columns=['Target', 'Target_Binary'])
y_train = df_train['Target']
X_test = df_test.drop(columns=['Target', 'Target_Binary'])
y_test = df_test['Target']

# Mise à l'échelle des caractéristiques
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Mise à l'échelle de la cible
target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

if train_model:
    # Construction du modèle
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    # Compilation du modèle
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    model.summary()

    # Configuration du callback EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraînement
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Sauvegarde de l'historique
    with open("model/training_history.json", "w") as f:
        json.dump(history.history, f)

    print("Historique de l'entraînement sauvegardé avec succès !")
else:
    # Chargement du modèle sauvegardé
    model = load_model("model/trading_model.keras")
    print("Modèle chargé avec succès !")

    # Chargement de l'historique
    with open("model/training_history.json", "r") as f:
        loaded_history = json.load(f)

# Évaluation du modèle
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Sauvegarde du modèle
model.save("model/trading_model.keras")
print("Modèle sauvegardé avec succès !")

# Prédictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled)

# Résidus
residuals = y_test.values.reshape(-1, 1) - y_pred

# Calculer le biais moyen
bias = np.mean(residuals)

# Ajuster les prédictions
y_pred_adjusted = y_pred + bias

print(f"Biais moyen : {bias}")

# Derniers jours pour les visualisations
last_week_X = X_test_scaled[-7:]
last_week_y_true = y_test.values[-7:].reshape(-1, 1)
last_week_y_pred_scaled = model.predict(last_week_X)
last_week_y_pred = target_scaler.inverse_transform(last_week_y_pred_scaled)
last_week_y_pred_adjusted = last_week_y_pred + bias


# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, y_test.values, label='Vraies Valeurs')
plt.plot(df_test.index, y_pred_adjusted, label='Prédictions')
plt.legend()
plt.title("Comparaison des Vraies Valeurs et des Prédictions (avec Dates)")
plt.xlabel("Date")
plt.ylabel("Cours de Clôture")
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(last_week_y_true)), last_week_y_true, label="Vraies Valeurs", marker='o')
plt.plot(np.arange(len(last_week_y_pred_adjusted)), last_week_y_pred_adjusted, label="Prédictions", marker='x')
plt.title("Projection sur la Dernière Semaine")
plt.xlabel("Jour")
plt.ylabel("Cours de Clôture (USD)")
plt.legend()
plt.grid()
plt.show()


# Distribution des résidus
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, color='blue', alpha=0.7)
plt.title("Distribution des Résidus")
plt.xlabel("Erreur (Vraie Valeur - Prédiction)")
plt.ylabel("Fréquence")

plt.show()
