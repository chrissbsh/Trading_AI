import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# NOUVEAUX IMPORTS
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
# Assurez-vous que votre fichier config.py est accessible
from config import *

# --- Fonctions (inchangées) ---

def load_data():
    print("🔹 Chargement des données...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    return df

def create_target(df):
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)
    
    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        def label_target(x):
            if x < thresholds[0]: return 0
            elif x <= thresholds[1]: return 1
            else: return 2
        df["target"] = df["ret_future"].apply(label_target)
    else:
        raise ValueError("Stratégie de seuil non reconnue.")
    return df

# --- 1. Préparation des données ---
print("🔹 1. Préparation des données...")
df_raw = load_data()
df_raw = df_raw.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"])

df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)]
df_main = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE]

df_main = create_target(df_main)
df_holdout = create_target(df_holdout)

print(f"Taille de l'ensemble d'entraînement (main): {len(df_main)}")
print(f"Taille de l'ensemble de test (holdout): {len(df_holdout)}")

print("\n🔹 Distribution des classes (main):")
print(df_main["target"].value_counts(normalize=True))
print("\n🔹 Distribution des classes (holdout):")
print(df_holdout["target"].value_counts(normalize=True))

features_to_use = [col for col in df_main.columns if col not in [DATE_COL, 'target', 'ret_future']]

print(f"\n✅ {len(features_to_use)} features utilisées pour l'entraînement.")

X_train = df_main[features_to_use]
y_train = df_main['target']

X_test = df_holdout[features_to_use]
y_test = df_holdout['target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Optimisation des hyperparamètres avec GridSearchCV ---
print("\n🔹 2. Optimisation des hyperparamètres...")

# ÉTAPE 2.1 : Définir le validateur croisé pour les séries temporelles
# Cela garantit que les données de validation sont TOUJOURS postérieures aux données d'entraînement.
tscv = TimeSeriesSplit(n_splits=5)

# ÉTAPE 2.2 : Définir la grille de paramètres à tester
# C'est une grille de départ. N'hésitez pas à l'étendre, mais attention au temps de calcul.
param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],         # Ratio d'échantillonnage des observations
    'colsample_bytree': [0.8, 1.0]   # Ratio d'échantillonnage des colonnes (features)
}

# ÉTAPE 2.3 : Créer le modèle de base XGBoost
# Les paramètres seront fournis par GridSearchCV
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False # Recommandé pour les versions récentes de XGBoost
)

# ÉTAPE 2.4 : Configurer et lancer GridSearchCV
# verbose=2 pour voir la progression, n_jobs=-1 pour utiliser tous les cœurs du CPU
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,  # <-- C'est ici que la magie opère pour les séries temporelles !
    # scoring='accuracy', # ou 'f1_macro' pour mieux gérer le déséquilibre
    scoring="f1_macro", 
    n_jobs=-1,
    verbose=2
)

print("🚀 Lancement de la recherche des meilleurs hyperparamètres... (cela peut prendre du temps)")
grid_search.fit(X_train_scaled, y_train)

print("\n✅ Recherche terminée !")
print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")
print(f"Meilleur score de validation croisée (accuracy) : {grid_search.best_score_:.4f}")

# Le meilleur modèle est déjà entraîné sur l'ensemble des données de train avec les meilleurs paramètres
best_model = grid_search.best_estimator_


# --- 3. Évaluation du MEILLEUR modèle sur les données de test (holdout) ---
print("\n🔹 3. Évaluation finale sur l'ensemble de test (holdout)...")

y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred)
print(f"\nAccuracy sur le jeu de test: {accuracy:.4f}")
print(f"\F1 Score sur le jeu de test: {f1_macro:.4f}")

print("\nRapport de Classification:")
print(classification_report(y_test, y_pred, target_names=['Baisse (0)', 'Neutre (1)', 'Hausse (2)']))

print("\nMatrice de Confusion:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baisse (0)', 'Neutre (1)', 'Hausse (2)'], 
            yticklabels=['Baisse (0)', 'Neutre (1)', 'Hausse (2)'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie valeur')
plt.title('Matrice de Confusion du Meilleur Modèle')
plt.show()

# --- 4. Importance des features du MEILLEUR modèle ---
print("\n🔹 4. Importance des features du meilleur modèle...")
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(best_model, ax=ax, importance_type='gain', max_num_features=20, height=0.8)
plt.title('Top 20 des Features les plus importantes (par gain)')
plt.show()