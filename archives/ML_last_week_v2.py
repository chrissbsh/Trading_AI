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
import os

# Créer le dossier model s'il n'existe pas
if not os.path.exists("model"):
    os.makedirs("model")

train_model = False  # Changer à True pour entraîner le modèle

df = pd.read_csv("consolidated_data.csv", parse_dates=["Date"], index_col="Date")

df['Target'] = df['AAPL_Close'].shift(-7)
df = df.dropna()  # Supprimer les lignes avec des NaN APRÈS le shift

print("Dernières lignes des données originales :")
print(df[['AAPL_Close', 'Target']].tail())

# Découpe en train / test
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Séparation X et y (plus besoin de Target_Binary pour la régression)
X_train = df_train.drop(columns=['Target'])
y_train = df_train['Target']
X_test = df_test.drop(columns=['Target'])
y_test = df_test['Target']

# Scalers (important : fit sur les données d'entraînement SEULEMENT)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

if train_model:
    # Construction du modèle
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, y_train_scaled,  # Utiliser les données scalées
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    with open("model/training_history.json", "w") as f:
        json.dump(history.history, f)

    model.save("model/trading_model.keras")
    print("Modèle entraîné et sauvegardé avec succès !")

else:
    try:
      model = load_model("model/trading_model.keras")
      with open("model/training_history.json", "r") as f:
        loaded_history = json.load(f)
      print("Modèle et historique chargés avec succès !")
      history = loaded_history
    except FileNotFoundError:
      print("Erreur : Fichiers du modèle non trouvés. Veuillez entraîner le modèle en mettant train_model à True.")
      exit()

# Évaluation (sur les données SCALÉES)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
print(f"Test Loss (scaled): {test_loss}")
print(f"Test MAE (scaled): {test_mae}")

# Prédictions (sur les données SCALÉES)
y_pred_scaled = model.predict(X_test_scaled)

# Inversion du scaling (TRÈS IMPORTANT)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test = target_scaler.inverse_transform(y_test_scaled)


# Calcul des résidus (sur les données à l'échelle originale)
residuals = y_test - y_pred

# Prédictions sur les 7 et 14 derniers jours (en utilisant les données scalées puis en inversant)
last_week_X_scaled = X_test_scaled[-7:]
last_week_y_true = y_test[-7:]
last_week_y_pred_scaled = model.predict(last_week_X_scaled)
last_week_y_pred = target_scaler.inverse_transform(last_week_y_pred_scaled)

last_14_X_scaled = X_test_scaled[-14:]
last_14_y_true = y_test[-14:]
last_14_y_pred_scaled = model.predict(last_14_X_scaled)
last_14_y_pred = target_scaler.inverse_transform(last_14_y_pred_scaled)


if train_model == False:
    history = loaded_history

else:
    history = history.history

# Courbe de perte
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Évolution de la Perte pendant l'Entraînement")

# Comparer les prédictions avec les vraies valeurs
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Vraies Valeurs')
plt.plot(y_pred, label='Prédictions')
plt.legend()
plt.title("Comparaison des Vraies Valeurs et des Prédictions")

# Visualisation prédictions sur 7 jours
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(last_week_y_true)), last_week_y_true, label="Vraies Valeurs", marker='o')
plt.plot(np.arange(len(last_week_y_pred)), last_week_y_pred, label="Prédictions", marker='x')
plt.title("Projection du Cours sur la Dernière Semaine")
plt.xlabel("Jour (par rapport à la dernière semaine)")
plt.ylabel("Cours de Clôture (prédiction sur 7 jours)")
plt.legend()
plt.grid()


# Visualisation sur 14 jours
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(last_14_y_true)), last_14_y_true, label="Vraies Valeurs", marker='o')
plt.plot(np.arange(len(last_14_y_pred)), last_14_y_pred, label="Prédictions", marker='x')
plt.axvline(x=6.5, color='gray', linestyle='--', label="Début de la Dernière Semaine")
plt.title("Projection du Cours avec Contexte")
plt.xlabel("Jour (14 derniers jours)")
plt.ylabel("Cours de Clôture")
plt.legend()
plt.grid()


# Visualisation des résidus
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, color='blue', alpha=0.7)
plt.title("Distribution des Résidus")
plt.xlabel("Erreur (Vraie Valeur - Prédiction)")
plt.ylabel("Fréquence")


# Afficher les graphiques
plt.show()