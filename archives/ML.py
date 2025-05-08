import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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


train_model = False # Changer à True pour entraîner le modèle


df = pd.read_csv("consolidated_data.csv", parse_dates=["Date"], index_col="Date")

df['Target'] = df['AAPL_Close'].shift(-7)

df['Target_Binary'] = (df['AAPL_Close'].shift(-7) > df['AAPL_Close']).astype(int)

df = df.dropna()  # Supprimer les lignes avec des NaN

features = df.drop(columns=['Target', 'Target_Binary'])  # Retirer les colonnes de la cible
# Définir la cible y comme la colonne 'Target'
y = df['Target']

print("Dernières lignes des données originales :")
print(df[['AAPL_Close', 'Target']].tail())

# Normaliser les données
# Scaler pour les caractéristiques
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(features)

# Scaler pour la cible
target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(y.values.reshape(-1, 1))

# Créer un DataFrame avec les données normalisées pour les caractéristiques
features_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

# Diviser les données
X = features_scaled
y_scaled = scaled_target  # Utiliser la cible normalisée

X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

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
    with open("model/training_history.json", "w") as f:
        json.dump(history.history, f)

    print("Historique de l'entraînement sauvegardé avec succès !")

else :
    # Charger le modèle sauvegardé
    model = load_model("model/trading_model.keras")
    print("Modèle chargé avec succès !")

    # Charger l'historique depuis le fichier JSON
    with open("model/training_history.json", "r") as f:
        loaded_history = json.load(f)


# Évaluation sur les données de test
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Sauvegarder le modèle 
model.save("model/trading_model.keras")
print("Modèle sauvegardé avec succès !")




# Prédire avec les données de test
y_pred = model.predict(X_test)

# Calcul des résidus
residuals = y_test.flatten() - y_pred.flatten()

# Sélectionner les 7 derniers jours pour la prédiction
last_week_X = X_test[-7:]  # Dernières 7 observations dans X_test
last_week_y_true = y_test[-7:]  # Vraies valeurs correspondantes

# Faire des prédictions pour la dernière semaine
last_week_y_pred = model.predict(last_week_X)

# Récupérer les 14 derniers jours pour une meilleure perspective
last_14_X = X_test[-14:]  # 7 jours avant et la dernière semaine
last_14_y_true = y_test[-14:]  # Vraies valeurs correspondantes

# Faire des prédictions pour les 14 derniers jours
last_14_y_pred = model.predict(last_14_X)




# Inverser la normalisation pour les prédictions
y_pred_original_scale = target_scaler.inverse_transform(y_pred)

# Inverser la normalisation pour les vraies valeurs (y_test)
y_test_original_scale = target_scaler.inverse_transform(y_test)

# Inverser la normalisation pour les 7 derniers jours
last_week_y_pred_original_scale = target_scaler.inverse_transform(last_week_y_pred)
last_week_y_true_original_scale = target_scaler.inverse_transform(last_week_y_true)

# Inverser la normalisation pour les 14 derniers jours
last_14_y_pred_original_scale = target_scaler.inverse_transform(last_14_y_pred)
last_14_y_true_original_scale = target_scaler.inverse_transform(last_14_y_true)

residuals_original_scale = last_week_y_true_original_scale.flatten() - last_week_y_pred_original_scale.flatten()
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
plt.plot(y_test_original_scale, label='Vraies Valeurs')
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