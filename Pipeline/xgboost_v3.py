import warnings
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Assuming ces constantes sont définies dans config.py
from config import DATA_FILE_PATH, DATE_COL, TARGET_PRICE_COL, PRED_HORIZON, FIXED_THRESHOLDS
from config import HOLDOUT_START_DATE, HOLDOUT_END_DATE

# 1. Chargement et nettoyage des données
df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
df.drop(columns=["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"],
        errors="ignore", inplace=True)

# 2. Séparation hold-out / train
mask_hold = df[DATE_COL].between(HOLDOUT_START_DATE, HOLDOUT_END_DATE)
df_hold  = df[mask_hold].copy()
df_train = df[~mask_hold].copy()

# 3. Création de la cible
def make_target(d):
    d = d.copy()
    d['ret_future'] = (d[TARGET_PRICE_COL].shift(-PRED_HORIZON) - d[TARGET_PRICE_COL]) / d[TARGET_PRICE_COL]
    d.dropna(subset=['ret_future'], inplace=True)
    lo, hi = FIXED_THRESHOLDS
    d['target'] = np.select(
        [d.ret_future < lo, d.ret_future <= hi],
        [0, 1],
        default=2
    )
    return d

df_train = make_target(df_train)
df_hold  = make_target(df_hold)

# 4. Colonnes de features
feat_cols = [c for c in df_train.select_dtypes(np.number).columns
             if c not in ('ret_future', 'target')]

# 5. Standardisation
X_train_full = df_train[feat_cols].values
y_train_full = df_train['target'].values
X_test       = df_hold[feat_cols].values
y_test       = df_hold['target'].values

scaler = StandardScaler().fit(X_train_full)
X_train_full = scaler.transform(X_train_full)
X_test       = scaler.transform(X_test)

# 6. Calcul des poids de classes sur le train complet
classes       = np.unique(y_train_full)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=classes,
                                     y=y_train_full)
weight_map    = {cls: w for cls, w in zip(classes, class_weights)}
sample_w_full = np.vectorize(weight_map.get)(y_train_full)
print("Class weights:", weight_map)

# 7. Découpe manuelle 80/20 par ordre chronologique
df_train_sorted = df_train.sort_values(DATE_COL).reset_index(drop=True)

split_idx = int(len(df_train_sorted) * 0.8)
df_tr = df_train_sorted.iloc[:split_idx]
df_val = df_train_sorted.iloc[split_idx:]

# Reconstruction des arrays
X_tr   = scaler.transform(df_tr[feat_cols].values)
y_tr   = df_tr['target'].values
sw_tr  = np.vectorize(weight_map.get)(y_tr)

X_val  = scaler.transform(df_val[feat_cols].values)
y_val  = df_val['target'].values

# 8. Création des DMatrix
dtrain = xgb.DMatrix(X_tr,  label=y_tr,  weight=sw_tr)
dvalid = xgb.DMatrix(X_val,  label=y_val)
dhold  = xgb.DMatrix(X_test, label=y_test)

# 9. Paramètres XGBoost
param_best = {
    'objective':      'multi:softprob',
    'num_class':      3,
    'eval_metric':    'mlogloss',
    'learning_rate':  0.03,
    'max_depth':      4,
    'subsample':      0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma':            0.2,
    'reg_lambda':       5,
    'tree_method':     'hist',
    'random_state':    42
}

# 10. Entraînement avec early stopping sur le set de validation
evals = [(dtrain, 'train'), (dvalid, 'validation')]

model = xgb.train(
    params=param_best,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=10
)

print(f"Meilleur itération (validation) : {model.best_iteration}")

# 11. Fonctions de métriques
def metrics(model, X, y, name="set"):
    yprob = model.predict(xgb.DMatrix(X))
    yhat  = np.argmax(yprob, axis=1)
    acc   = accuracy_score(y, yhat)
    f1    = f1_score(y, yhat, average='weighted')
    print(f"{name} -> accuracy={acc:.3f} | f1_weighted={f1:.3f}")
    return yhat

yhat_tr = metrics(model, X_tr,  y_tr,  "TRAIN")
yhat_va = metrics(model, X_val, y_val, "VALID")
yhat_te = metrics(model, X_test, y_test, "HOLD")

# 12. Rapport classification et matrice de confusion
print("\nClassification report (hold-out):")
print(classification_report(y_test, yhat_te,
                            target_names=['Baisse', 'Neutre', 'Hausse'],
                            zero_division=0))

cm = confusion_matrix(y_test, yhat_te)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Baisse', 'Neutre', 'Hausse'],
            yticklabels=['Baisse', 'Neutre', 'Hausse'])
plt.title("Confusion matrix – hold-out")
plt.show()

# 13. Importance des features
gain = model.get_score(importance_type='gain')
mapping = {f"f{i}": feat for i, feat in enumerate(feat_cols)}
imp_df = (pd.DataFrame({
              'feature': [mapping.get(k, k) for k in gain],
              'gain':    list(gain.values())
          })
          .sort_values('gain', ascending=False))
print("\nTop 15 features:\n", imp_df.head(15))
