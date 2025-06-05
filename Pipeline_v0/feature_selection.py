import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import config
from preprocessing_v2 import compute_future_return # Assurez-vous que c'est bien utilisé ou nécessaire

def shap_feature_selection(df: pd.DataFrame,
                           target_col_name: str = 'target_for_shap', # temp target
                           price_col: str = config.TARGET_PRICE_COL,
                           horizon: int = config.PRED_HORIZON,
                           n_classes: int = config.N_CLASSES, # n_classes from config for qcut
                           top_n: int = config.TOP_N_FEATURES,
                           exclude_cols_from_X = None):
    """
    Performs feature selection using SHAP with a LightGBM model.
    """
    if exclude_cols_from_X is None:
        exclude_cols_from_X = {config.DATE_COL}

    df_fs = df.copy()
    
    df_fs['ret_future_fs'] = compute_future_return(df_fs[price_col], horizon)
    df_fs.dropna(subset=['ret_future_fs'], inplace=True)
    
    # Utiliser n_classes pour le qcut
    df_fs[target_col_name] = pd.qcut(df_fs['ret_future_fs'], q=n_classes, labels=False, duplicates='drop')
    df_fs.dropna(subset=[target_col_name], inplace=True)

    potential_features = [c for c in df_fs.columns if c not in exclude_cols_from_X and c not in ['ret_future_fs', target_col_name, price_col]]
    
    X = df_fs[potential_features].copy()
    X = X.dropna(axis=1, how='all')
    # X = X.loc[:, X.nunique() > 1] 
    
    y = df_fs[target_col_name]

    if X.empty or y.empty or len(X) != len(y) or X.shape[1] == 0:
        print("Warning: Not enough data or features for SHAP feature selection after pre-processing.")
        # Fallback: return all potential features if SHAP can't run
        return [c for c in df.columns if c not in exclude_cols_from_X and c != price_col and c not in ['ret_future_fs', target_col_name]], None

    # S'assurer que y a bien plusieurs classes si n_classes > 1
    # LGBM peut avoir besoin d'au moins 2 classes pour 'multiclass'
    if n_classes > 1 and len(np.unique(y)) < 2 :
        print(f"Warning: Target for SHAP has only {len(np.unique(y))} unique values. SHAP might not work as expected.")
        # Fallback
        return [c for c in df.columns if c not in exclude_cols_from_X and c != price_col and c not in ['ret_future_fs', target_col_name]], None


    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=config.SEED, verbose=-1)
    try:
        lgb_model.fit(X, y)
    except Exception as e:
        print(f"Error during LightGBM model fitting for SHAP: {e}")
        # Fallback
        return [c for c in df.columns if c not in exclude_cols_from_X and c != price_col and c not in ['ret_future_fs', target_col_name]], None


    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X) 
    
    # --- DÉBUT DE LA MODIFICATION ---
    if isinstance(shap_values, list):
        # Cas multiclasse où shap_values est une liste de K arrays (un par classe),
        # chaque array ayant la forme (N_samples, M_features).
        # On convertit la liste en un seul array numpy de forme (K_classes, N_samples, M_features).
        shap_values_arr = np.array(shap_values)
        # On prend la valeur absolue.
        abs_shap_values = np.abs(shap_values_arr)
        # On moyenne sur l'axe des classes (axis 0) ET sur l'axe des échantillons (axis 1).
        # Le résultat est un array 1D de longueur M_features.
        mean_abs_shap = abs_shap_values.mean(axis=(0, 1))
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Cas (moins courant pour TreeExplainer avec LGBM, mais possible) où shap_values
        # est directement un array 3D (N_samples, M_features, K_classes).
        abs_shap_values = np.abs(shap_values)
        # On moyenne sur l'axe des échantillons (axis 0) ET sur l'axe des classes (axis 2).
        # Le résultat est un array 1D de longueur M_features.
        mean_abs_shap = abs_shap_values.mean(axis=(0, 2))
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # Cas binaire ou régression où shap_values est (N_samples, M_features).
        abs_shap_values = np.abs(shap_values)
        # On moyenne sur l'axe des échantillons (axis 0).
        # Le résultat est un array 1D de longueur M_features.
        mean_abs_shap = abs_shap_values.mean(axis=0)
    else:
        raise TypeError(f"Format des valeurs SHAP non supporté : {type(shap_values)}, "
                        f"ndim: {shap_values.ndim if isinstance(shap_values, np.ndarray) else 'N/A'}")
    # --- FIN DE LA MODIFICATION ---
            
    feature_importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
    
    top_features = feature_importance.head(top_n).index.tolist()
    print(f"Selected top {len(top_features)} features via SHAP: {top_features}")
    return top_features, feature_importance