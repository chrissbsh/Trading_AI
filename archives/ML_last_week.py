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

# Récupérer le nom du fichier courant sans son extension
name = os.path.splitext(os.path.basename(__file__))[0]

# Afficher le nom du fichier
print(name)

# Créer un sous-dossier avec le nom du fichier
os.makedirs(f"model/{name}", exist_ok=True)

train_model = True # Changer à True pour entraîner le modèle


df = pd.read_csv("consolidated_data.csv", parse_dates=["Date"], index_col="Date")

df['Target'] = df['AAPL_Close'].shift(-7)
df['Target_Binary'] = (df['AAPL_Close'].shift(-7) > df['AAPL_Close']).astype(int)
df = df.dropna()  # Supprimer les lignes avec des NaN

print("Dernières lignes des données originales :")
print(df[['AAPL_Close', 'Target']].tail())


# Découpe d'abord le DataFrame en train / test
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# X et y sur train
X_train = df_train.drop(columns=['Target', 'Target_Binary'])
y_train = df_train['Target']

# X et y sur test
X_test = df_test.drop(columns=['Target', 'Target_Binary'])
y_test = df_test['Target']

# Fit le scaler sur X_train uniquement
feature_scaler = MinMaxScaler()
feature_scaler.fit(X_train)
X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Idem pour la cible
target_scaler = MinMaxScaler()
target_scaler.fit(y_train.values.reshape(-1,1))
y_train_scaled = target_scaler.transform(y_train.values.reshape(-1,1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1,1))


if train_model == True:
    # Construire le modèle
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Couche d'entrée
        Dropout(0.2),  # Dropout pour éviter l'overfitting
        Dense(64, activation='relu'),  # Couche cachée
        Dropout(0.2),
        Dense(1, activation='linear')  # Couche de sortie (prédiction continue pour 'Target')
    ])

    # Compiler le modèle
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])


    # Résumé du modèle
    model.summary()


    # Configurer EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # Fraction des données d'entraînement utilisée pour la validation
        epochs=100,  # Nombre d'epochs
        batch_size=32,  # Taille des batchs
        callbacks=[early_stopping],
        verbose=1
    )

    # Sauvegarder l'historique dans un fichier JSON
    with open(f"model/{name}/training_history.json", "w") as f:
        json.dump(history.history, f)

    print("Historique de l'entraînement sauvegardé avec succès !")

else :
    # Charger le modèle sauvegardé
    model = load_model(f"model/{name}/trading_model.keras")
    print("Modèle chargé avec succès !")

    # Charger l'historique depuis le fichier JSON
    with open(f"model/{name}/training_history.json", "r") as f:
        loaded_history = json.load(f)


# Évaluation sur les données de test
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Sauvegarder le modèle 
model.save(f"model/{name}/trading_model.keras")
print("Modèle sauvegardé avec succès !")




# Prédire avec les données de test
y_pred = model.predict(X_test)

y_test = np.array(y_test)

print(type(y_pred), y_pred.shape)
print(type(y_test), y_pred.shape)

print(type(X_test), X_test.shape)
print(type(y_test), y_test.shape)

# Calcul des résidus sur la même forme
residuals = y_test - y_pred  # tous deux (N, 1)

# Sélectionner les 7 derniers jours pour la prédiction
last_week_X = X_test[-7:]
last_week_y_true = y_test[-7:]  # 7 vraies valeurs (7, 1)

# Faire des prédictions pour la dernière semaine
last_week_y_pred = model.predict(last_week_X)  # (7, 1)

# Récupérer les 14 derniers jours pour une meilleure perspective
last_14_X = X_test[-14:]
last_14_y_true = y_test[-14:]  # (14, 1)

# Faire des prédictions pour les 14 derniers jours
last_14_y_pred = model.predict(last_14_X)  # (14, 1)


# ---------------------------
# Inverser la normalisation
# ---------------------------

# 1) Sur l'ensemble du test
y_pred_original_scale = target_scaler.inverse_transform(y_pred)          # (N, 1)

# 2) Sur la dernière semaine (7 jours)
last_week_y_pred_original_scale = target_scaler.inverse_transform(last_week_y_pred)    # (7, 1)
last_week_y_true_original_scale = target_scaler.inverse_transform(last_week_y_true)    # (7, 1)

# 3) Sur les 14 derniers jours
last_14_y_pred_original_scale = target_scaler.inverse_transform(last_14_y_pred)        # (14, 1)
last_14_y_true_original_scale = target_scaler.inverse_transform(last_14_y_true)        # (14, 1)

# Calcul des résidus sur la dernière semaine, après inversion
residuals_original_scale = (
    last_week_y_true_original_scale.flatten()
    - last_week_y_pred_original_scale.flatten()
)
print("Écarts (résidus) après inversion :", residuals_original_scale)



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
plt.plot(y_pred_original_scale, label='Prédictions')
plt.legend()
plt.title("Comparaison des Vraies Valeurs et des Prédictions")

# Visualisation prédictions sur 7 jours
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(last_week_y_true_original_scale)), last_week_y_true_original_scale, label="Vraies Valeurs", marker='o')
plt.plot(np.arange(len(last_week_y_pred_original_scale)), last_week_y_pred_original_scale, label="Prédictions", marker='x')
plt.title("Projection du Cours sur la Dernière Semaine")
plt.xlabel("Jour (par rapport à la dernière semaine)")
plt.ylabel("Cours de Clôture (prédiction sur 7 jours)")
plt.legend()
plt.grid()


# Visualisation sur 14 jours
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(last_14_y_true_original_scale)), last_14_y_true_original_scale, label="Vraies Valeurs", marker='o')
plt.plot(np.arange(len(last_14_y_pred_original_scale)), last_14_y_pred_original_scale, label="Prédictions", marker='x')
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