import pandas as pd
from sklearn.decomposition import PCA
import lightgbm as lgb
import shap

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


def correlation_with_target(df, target_col="TARGET"):
    """
    Calculate the correlation between each feature in the DataFrame and the target column.

    Parameters:
    - df: pandas DataFrame containing the data.
    - target_col: name of the target column.

    Returns:
    - A pandas Series containing the correlation of each feature with the target.
    """
    # Ensure the target column is present in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Calculate the correlation with the target column
    correlations = df.corr()[target_col].drop(target_col)

    return correlations