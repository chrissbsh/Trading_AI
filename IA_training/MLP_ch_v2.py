import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.utils import compute_sample_weight

# --- 1) Chargement et feature engineering comme avant ---
df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"]).sort_values("Date")
df["target_7d"] = (df["SP500_historical_data_Close"].shift(-7) > df["SP500_historical_data_Close"]).astype(int)
df = df.iloc[:-7]
df["std_21"] = df["sp500_return_1d"].rolling(21).std()
df["hv_30"]  = df["sp500_return_1d"].rolling(30).std()
df["r_sp_gold"] = df["SP500_historical_data_Close"] / df["gold_historical_data_Close"]
df["r_sp_dxy"]  = df["SP500_historical_data_Close"] / df["dollar_index_historical_data_Close"]
df["r_sp_bond"] = df["SP500_historical_data_Close"] / df["Market_yield_US_10_year_DGS10"]
vix = df["^VIX_historical_data_Close"]
df["vix_direction"] = vix.diff().fillna(0).gt(0).astype(int)
df["vix_high"]      = vix.gt(vix.rolling(63).median()).astype(int)
if "PMI" in df.columns:
    df["macro_regime"] = (df["PMI"] > 50).astype(int)
df.dropna(inplace=True)
exclude = {"Date", "target_7d"}
features = [c for c in df.columns if c not in exclude]

# --- Hold-out final ---
holdout = (df["Date"] >= "2023-01-03") & (df["Date"] <= "2025-04-14")
df_holdout = df.loc[holdout]
df_roll    = df.loc[~holdout]

# --- Rolling windows quarterly ---
starts = []
d = df_roll["Date"].min()
last = df_roll["Date"].max() - relativedelta(years=4) - relativedelta(months=3)
while d <= last:
    starts.append(d)
    d += relativedelta(months=3)

results = []
for start in starts:
    tr_start, tr_end = start, start + relativedelta(years=3) - pd.Timedelta(days=1)
    val_end          = tr_end + relativedelta(years=1)
    te_start         = val_end + pd.Timedelta(days=1)
    te_end           = te_start + relativedelta(months=3) - pd.Timedelta(days=1)

    tr = df_roll[(df_roll["Date"] >= tr_start) & (df_roll["Date"] <= tr_end)]
    va = df_roll[(df_roll["Date"] >  tr_end)   & (df_roll["Date"] <= val_end)]
    te = df_roll[(df_roll["Date"] >= te_start)  & (df_roll["Date"] <= te_end)]
    if te.empty: break

    X_tr, y_tr = tr[features].values, tr["target_7d"].values
    X_va, y_va = va[features].values, va["target_7d"].values
    X_te, y_te = te[features].values, te["target_7d"].values

    # a) Scaling
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # b) Entraînement MLP avec gestion du déséquilibre des classes
    # Utilisation d'une approche alternative pour traiter les poids sans utiliser sample_weight
    # Solution 1: Rééquilibrage par sous-échantillonnage ou sur-échantillonnage
    
    # Compter les classes
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    
    # Calculer les indices des échantillons positifs et négatifs
    pos_indices = np.where(y_tr == 1)[0]
    neg_indices = np.where(y_tr == 0)[0]
    
    # Option 1: Rééquilibrage par sous-échantillonnage de la classe majoritaire
    if n_pos < n_neg:
        # Sous-échantillonnage de la classe négative
        selected_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])
    else:
        # Sous-échantillonnage de la classe positive
        selected_pos_indices = np.random.choice(pos_indices, size=n_neg, replace=False)
        selected_indices = np.concatenate([selected_pos_indices, neg_indices])
    
    # Utiliser les indices sélectionnés pour l'entraînement
    X_tr_balanced = X_tr_s[selected_indices]
    y_tr_balanced = y_tr[selected_indices]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        max_iter=200,
        random_state=42,
        early_stopping=False
    )
    
    # Entraînement avec données rééquilibrées
    mlp.fit(X_tr_balanced, y_tr_balanced)

    # c) Optimisation du seuil sur validation
    p_va = mlp.predict_proba(X_va_s)[:,1]
    ths = np.linspace(0.1,0.9,81)
    f1s = [f1_score(y_va, p_va>t) for t in ths]
    t_star = ths[np.argmax(f1s)]

    # d) Évaluation sur test
    p_te    = mlp.predict_proba(X_te_s)[:,1]
    y_pred  = (p_te > t_star).astype(int)
    results.append({
        "window_start":  tr_start.date(),
        "window_test":   f"{te_start.date()}→{te_end.date()}",
        "threshold*":    t_star,
        "AUC_test":      roc_auc_score(y_te, p_te),
        "F1_test":       f1_score(y_te, y_pred),
        "BalAcc_test":   balanced_accuracy_score(y_te, y_pred),
        "Precision_test":precision_score(y_te, y_pred),
        "Recall_test":   recall_score(y_te, y_pred)
    })

# Affichage
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))