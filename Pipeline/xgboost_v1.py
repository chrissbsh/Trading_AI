import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from feature_selection import select_top_features_pca, select_top_features_shap
from config import *
from sklearn.preprocessing import StandardScaler

def load_data():
    print("🔹 Chargement des données...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    return df

def create_target(df):
    """
    Ajoute une colonne 'ret_future' qui mesure l'évolution de TARGET_PRICE_COL comme (prix actuel - prix passé) / prix actuel.
    """
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)

    # print(df[["SP500_historical_data_Close", "ret_future"]].head(20))

    # input("Press Enter to continue...")

    # labelling target
    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        
        print(f"🔹 Seuils pour la classification : {thresholds}")

        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2

        df["target"] = df["ret_future"].apply(label_target)

    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}. Utilisez 'fixed'.")

    return df

    
# 1. Charger les données
df_raw = load_data()

df_raw = df_raw.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"])

print(f"📊 Données chargées : {len(df_raw)} lignes, {len(df_raw.columns)} colonnes")

df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)]

df_main = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE]

df_main = create_target(df_main)

df_holdout = create_target(df_holdout)

# Vérification des classes
class_counts = df_main["target"].value_counts(normalize=True)
print("🔹 Distribution des classes main:")
print(class_counts)

class_counts = df_holdout["target"].value_counts(normalize=True)
print("🔹 Distribution des classes holdout:")
print(class_counts)

final_features = df_main.columns

final_features = [col for col in final_features if col != 'target' and col != 'ret_future' and col != 'Date']

if 'SP500_historical_data_Close' not in final_features:
    final_features.append('SP500_historical_data_Close')

print(f"🔹 Nombre features sélectionnées: {len(final_features)}")

X_train = df_main[final_features]
y_train = df_main['target']

X_test = df_holdout[final_features]
y_test = df_holdout['target']

# Appliquez le scaling si nécessaire (XGBoost y est moins sensible mais c'est une bonne pratique)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- 3. Création et Entraînement du modèle XGBoost ---
print("\n🔹 3. Entraînement du modèle XGBoost...")

# Paramètres de base pour un classifieur multi-classes
# XGBoost est très performant avec ses paramètres par défaut
model = xgb.XGBClassifier(
    objective='multi:softmax',  # Spécifie le type de problème
    num_class=3,              # Nombre de classes (0, 1, 2)
    n_estimators=400,         # Nombre d'arbres (itérations)
    max_depth=4,              # Profondeur max pour éviter l'overfitting
    learning_rate=0.01,        # Taux d'apprentissage
    eval_metric='mlogloss'    # Métrique d'évaluation pour le multi-classes
)

# Entraînement du modèle
model.fit(X_train, y_train)


# --- 4. Évaluation du modèle sur les données de test ---
print("\n🔹 4. Évaluation sur l'ensemble de test...")

# Prédiction des classes
y_pred = model.predict(X_test)

# Calcul des métriques de performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy sur le jeu de test: {accuracy:.4f}")

print("\nRapport de Classification:")
# Ce rapport vous donne la précision, le rappel et le f1-score pour chaque classe
print(classification_report(y_test, y_pred, target_names=['Baisse (0)', 'Neutre (1)', 'Hausse (2)']))

# Matrice de confusion pour voir où le modèle se trompe
print("\nMatrice de Confusion:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baisse (0)', 'Neutre (1)', 'Hausse (2)'], 
            yticklabels=['Baisse (0)', 'Neutre (1)', 'Hausse (2)'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie valeur')
plt.title('Matrice de Confusion')
plt.show()


print("\n🔹 5. Importance des features...")

# importance_type = 'weight'  # or 'gain', 'cover'
importance_type = 'gain'  # 'weight', 'gain', or 'cover'

# Visualisation avec xgb.plot_importance
fig, ax = plt.subplots(figsize=(8, 6))
xgb.plot_importance(model, ax=ax, importance_type=importance_type, max_num_features=25, height=0.2)
plt.title('Top des Features les plus importantes')

# Extraction et affichage de l'importance - VERSION CORRIGÉE
# Pour XGBClassifier, il faut passer par le booster
importance_dict = model.get_booster().get_score(importance_type=importance_type)

# Créer un mapping entre les indices XGBoost (f0, f1, f2...) et les vrais noms
feature_mapping = {f'f{i}': name for i, name in enumerate(final_features)}

# Créer le DataFrame avec les vrais noms de features
importance_df = pd.DataFrame({
    'Feature': [feature_mapping.get(k, k) for k in importance_dict.keys()],
    'Importance': list(importance_dict.values())
}).sort_values(by='Importance', ascending=False)

print("\n🔹 Features importantes:")
print(importance_df.head(10))

# # Version alternative si vous voulez inclure toutes les features (même celles avec importance 0)
# print("\n🔹 Version alternative avec toutes les features:")
# all_importance_df = pd.DataFrame({
#     'Feature': final_features,
#     'Importance': [importance_dict.get(f'f{i}', 0) for i, f in enumerate(final_features)]
# }).sort_values(by='Importance', ascending=False)

# print(all_importance_df.head(10))


plt.show()