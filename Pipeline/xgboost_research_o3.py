# =============================================================================
# 0) Imports & Config
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, make_scorer)

import xgboost as xgb

# --- modules perso / constantes (fichier config.py) --------------------------
from config import *

# Si tu veux garder la sélection PCA / SHAP
# from feature_selection import select_top_features_pca, select_top_features_shap

SEED = 42  # Reproductibilité
N_JOBS = -1  # Parallélisation

# =============================================================================
# 1) Fonctions utilitaires
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    print("🔹 Chargement des données…")
    return pd.read_csv(path, parse_dates=[DATE_COL])


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule 'ret_future' et créée la classe target (0,1,2).
    """
    df = df.copy()  # évite le SettingWithCopyWarning
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)

    if THRESHOLD_STRATEGY != "fixed":
        raise ValueError("Stratégie de seuil non gérée dans ce script")

    print(f"🔹 Seuils de classification : {FIXED_THRESHOLDS}")

    lo, hi = FIXED_THRESHOLDS
    df['target'] = np.select([df.ret_future < lo, df.ret_future <= hi],
                             [0, 1], default=2)
    return df


def display_confusion(y_true, y_pred, title="Matrice de confusion"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Baisse', 'Neutre', 'Hausse'],
                yticklabels=['Baisse', 'Neutre', 'Hausse'])
    plt.title(title); plt.xlabel('Prévision'); plt.ylabel('Réalité')
    plt.show()


# =============================================================================
# 2) Chargement & split temporel
# =============================================================================
df = load_data(DATA_FILE_PATH)

# Option : drop de colonnes inutiles
cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Split temporel : hold-out = période récente
mask_holdout = df[DATE_COL].between(HOLDOUT_START_DATE, HOLDOUT_END_DATE)
df_holdout   = create_target(df[mask_holdout])
df_train_raw = create_target(df[~mask_holdout])

# Vérification des classes
print("\nDistribution classes (train) :")
print(df_train_raw['target'].value_counts(normalize=True).round(3))

print("\nDistribution classes (hold-out) :")
print(df_holdout['target'].value_counts(normalize=True).round(3))

# =============================================================================
# 3) Sélection de features (facultatif)
# =============================================================================
# Version simple : on prend toutes les colonnes numériques sauf target et dates
feature_cols = df_train_raw.select_dtypes(np.number).columns.tolist()
feature_cols = [c for c in feature_cols if c not in ('ret_future', 'target')]

print(f"✅ {len(feature_cols)} features retenues.")

X_train = df_train_raw[feature_cols].values
y_train = df_train_raw['target'].values
X_test  = df_holdout[feature_cols].values
y_test  = df_holdout['target'].values

# =============================================================================
# 4) Pipeline = scaling + XGBoost
# =============================================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),          # XGBoost n'en a pas besoin mais c'est sain
    ('clf', xgb.XGBClassifier(
        objective='multi:softprob',        # renvoie proba multi-classe
        num_class=3,
        eval_metric='mlogloss',
        tree_method='hist',                # rapide sur CPU
        random_state=SEED,
        n_jobs=N_JOBS
    ))
])

# =============================================================================
# 5) Grille d’hyper-paramètres
# =============================================================================
param_grid = {
    'clf__n_estimators'      : [200, 400, 800],
    'clf__learning_rate'     : [0.03, 0.05, 0.1],
    'clf__max_depth'         : [3, 4, 6],
    'clf__subsample'         : [0.7, 0.85, 1.0],
    'clf__colsample_bytree'  : [0.7, 0.85, 1.0],
    'clf__min_child_weight'  : [1, 5, 10],
    'clf__gamma'             : [0, 0.1, 0.3],
    'clf__reg_lambda'        : [1, 5, 10],
}

# NOTE : si la grille est trop grosse, utilise RandomizedSearchCV
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring={
        'accuracy' : 'accuracy',
        'f1_macro' : make_scorer(f1_score, average='macro')
    },
    refit='f1_macro',              # on retient le meilleur f1_macro
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    verbose=2,
    n_jobs=N_JOBS
)

# =============================================================================
# 6) Recherche & entraînement
# =============================================================================
print("\n🔍  Recherche des meilleurs hyper-paramètres…")
search.fit(X_train, y_train)

print(f"\n✅  Best params ({search.best_score_:.4f} f1_macro CV) :")
for k, v in search.best_params_.items():
    print(f"   {k} : {v}")

best_model = search.best_estimator_

# =============================================================================
# 7) Évaluation finale sur le hold-out
# =============================================================================
print("\n🏁  Évaluation sur la période hold-out :")
y_pred = best_model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
f1m  = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy : {acc:.4f}   |   F1-macro : {f1m:.4f}\n")
print(classification_report(y_test, y_pred,
                            target_names=['Baisse', 'Neutre', 'Hausse']))

display_confusion(y_test, y_pred, title="Matrice de confusion – Hold-out")

# =============================================================================
# 8) Importance des variables (par gain)
# =============================================================================
booster = best_model.named_steps['clf'].get_booster()
gain_imps = booster.get_score(importance_type='gain')

# mapping f0→nom
mapping = {f"f{i}": col for i, col in enumerate(feature_cols)}
imp_df = (pd.DataFrame({'feature': [mapping.get(k, k) for k in gain_imps],
                        'importance': list(gain_imps.values())})
          .sort_values('importance', ascending=False))

print("\n🔝  Top 20 features par 'gain' :")
print(imp_df.head(20))

plt.figure(figsize=(8,6))
sns.barplot(data=imp_df.head(20), y='feature', x='importance', palette='viridis')
plt.title("Top 20 des features (gain XGBoost)")
plt.tight_layout()
plt.show()