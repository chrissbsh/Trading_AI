import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
import shap

"""
Ce script propose deux méthodes de sélection de features à partir d’un DataFrame contenant des variables financières
et une méthode de visualisation des variables les plus corrélées :

1. `select_top_features_pca` : utilise l’analyse en composantes principales (PCA) pour identifier les variables 
   les plus représentatives dans un espace réduit.
2. `select_top_features_shap` : utilise un modèle LightGBM et l’analyse SHAP pour mesurer l’influence moyenne de 
   chaque variable sur la prédiction d’une variable cible.
3. `correlation_with_target` : calcule la corrélation linéaire de chaque feature avec la variable cible.

Ces fonctions sont utiles pour réduire la dimensionnalité et conserver les variables les plus pertinentes 
avant entraînement d’un modèle.
"""

# Sélection des top features via PCA
def select_top_features_pca(df, top_n, target_col="ret_future"):
    # Enlever colonnes non numériques et target
    df_clean = df.dropna().copy()
    X = df_clean.select_dtypes(include=["number"]).drop(columns=[target_col], errors='ignore')

    # Appliquer PCA
    pca = PCA(n_components=top_n)
    pca.fit(X)

    # Importance des features selon la somme des valeurs absolues des composantes principales
    feature_importance = pd.Series(
        abs(pca.components_).sum(axis=0),
        index=X.columns
    ).sort_values(ascending=False)

    return list(feature_importance.head(top_n).index)


# Sélection des top features via SHAP et LightGBM
def select_top_features_shap(df, top_n,target_col="ret_future"):
    df_clean = df.dropna().copy()
    X = df_clean.select_dtypes(include=["number"]).drop(columns=[target_col], errors='ignore')
    y = df_clean[target_col]

    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_importance = pd.Series(
        abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)

    return list(feature_importance.head(top_n).index)

# Calcul de la corrélation de chaque feature avec la target
def correlation_with_target(df, target_col="TARGET"):
    # Ensure the target column is present in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Calculate the correlation with the target column
    correlations = df.corr()[target_col].drop(target_col)

    return correlations