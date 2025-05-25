"""
End-to-end pipeline sans fuite d’information :
– seuils recalculés dans chaque fenêtre d’entraînement  
– purged walk-forward (3 ans train → 1 an val → 3 mois test, buffer 14 jours)  
– LSTM unidirectionnel + jitter data-augmentation  
– métriques F1 + PnL/Sharpe par fenêtre
"""

# ───────────────────────── IMPORTS ────────────────────────── #
import os, random, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             precision_score, recall_score)
from scipy.stats import gmean
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, LayerNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore

import pickle

warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx("float32")

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ───────────────────────── PARAMETERS ─────────────────────── #
PRED_HORIZON      = 7            # jours
N_CLASSES         = 5
TIMESTEPS         = 30           # taille séquence
TRAIN_YEARS       = 3
VAL_YEARS         = 1
TEST_MONTHS       = 3
BUFFER_DAYS       = 14           # purge entre splits
JITTER_RATIO      = 0.05         # % du σ pour le jitter
EPOCHS            = 100
BATCH_SIZE        = 32
PATIENCE          = 10

# ───────────────────────── HELPERS ────────────────────────── #
def compute_future_return(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) - close) / close

def make_thresholds(series: pd.Series, k: int) -> np.ndarray:
    q = np.linspace(0, 1, k + 1)
    return series.quantile(q).values[1:-1]          # k-1 seuils internes

def label_from_thresholds(ret: float, th: np.ndarray) -> int:
    for i, t in enumerate(th):
        if ret <= t:
            return i
    return len(th)                                  # dernière classe

def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

def add_jitter(X, ratio):
    std = X.std(axis=(0,1), keepdims=True)
    noise = np.random.normal(0, std * ratio, X.shape)
    return X + noise

def lstm_model(input_shape, n_classes=5):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    lr_sched = CosineDecay(1e-3, decay_steps=1000)
    model.compile(Adam(lr_sched), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def pnl_metrics(y_true, y_pred, returns, horizon=PRED_HORIZON):
    """Long classes 3-4, short classes 0-1, cash 2."""
    pos = np.select([y_pred>=3, y_pred<=1], [ 1, -1], default=0)
    strat_ret = pos * returns.iloc[-len(pos):].values    # horizon-ahead returns
    cum = np.cumprod(1 + strat_ret) - 1
    pnl = cum[-1]
    if (strat_ret.std() == 0):
        sharpe = 0
    else:
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252/horizon)
    return float(pnl), float(sharpe)

# ───────────────────────── LOAD DATA ──────────────────────── #
df = (pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv",
                  parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True))

# exclure colonnes
EXCLUDE = {"Date"}
FEATURES = [c for c in df.columns if c not in EXCLUDE]

# hold-out très récent (jamais vu durant training walk)
holdout_mask = (df["Date"] >= "2023-01-03") & (df["Date"] <= "2025-04-14")
df_holdout   = df[holdout_mask].copy()
df_roll      = df[~holdout_mask].copy()

# ───────────────────────── WALK FORWARD ───────────────────── #
starts = []
d0   = df_roll["Date"].min()
last = df_roll["Date"].max() - relativedelta(years=TRAIN_YEARS+VAL_YEARS) - relativedelta(months=TEST_MONTHS)
while d0 <= last:
    starts.append(d0)
    d0 += relativedelta(months=3)

results  = []
best_f1  = -np.inf
best_set = {}

for split, start in enumerate(starts, 1):

    # ── window bounds
    tr_start = start
    tr_end   = start + relativedelta(years=TRAIN_YEARS) - pd.Timedelta(days=1)
    val_end  = tr_end + relativedelta(years=VAL_YEARS)
    test_start = val_end + pd.Timedelta(days=BUFFER_DAYS)
    test_end   = test_start + relativedelta(months=TEST_MONTHS) - pd.Timedelta(days=1)
    if test_end > df_roll["Date"].max(): break

    # ── raw splits
    tr = df_roll[(df_roll["Date"] >= tr_start) & (df_roll["Date"] <= tr_end)].copy()
    va = df_roll[(df_roll["Date"] > tr_end) & (df_roll["Date"] <= val_end)].copy()
    te = df_roll[(df_roll["Date"] >= test_start) & (df_roll["Date"] <= test_end)].copy()

    # ── compute returns & thresholds on TRAIN ONLY
    tr['ret_future'] = compute_future_return(tr["SP500_historical_data_Close"], PRED_HORIZON)
    tr.dropna(subset=['ret_future'], inplace=True)
    thres = make_thresholds(tr['ret_future'], N_CLASSES)

    # ── label whole sets with those thresholds
    for part in (tr, va, te):
        part['ret_future'] = compute_future_return(part["SP500_historical_data_Close"], PRED_HORIZON)
        part['target'] = part['ret_future'].apply(lambda r: label_from_thresholds(r, thres))
    tr, va, te = (p.iloc[:-PRED_HORIZON].dropna() for p in (tr, va, te))   # drop trailing NaNs

    # ── scale
    scaler = StandardScaler().fit(tr[FEATURES])
    X_tr = scaler.transform(tr[FEATURES]); y_tr = tr['target'].values
    X_va = scaler.transform(va[FEATURES]); y_va = va['target'].values
    X_te = scaler.transform(te[FEATURES]); y_te = te['target'].values

    # ── sequences
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, TIMESTEPS)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, TIMESTEPS)
    X_te_seq, y_te_seq = create_sequences(X_te, y_te, TIMESTEPS)

    # ── jitter augmentation
    X_aug = add_jitter(X_tr_seq, JITTER_RATIO)
    y_aug = y_tr_seq.copy()
    X_final = np.vstack([X_tr_seq, X_aug])
    y_final = np.concatenate([y_tr_seq, y_aug])

    # ── class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_final), y=y_final)
    class_weights = dict(zip(np.unique(y_final), cw))

    # ── model
    model = lstm_model(input_shape=(TIMESTEPS, X_final.shape[2]), n_classes=N_CLASSES)
    early = EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss')
    model.fit(X_final, y_final, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_va_seq, y_va_seq),
              callbacks=[early], class_weight=class_weights, verbose=0)

    # ── scores
    y_pred_va = model.predict(X_va_seq, verbose=0).argmax(axis=1)
    y_pred_te = model.predict(X_te_seq, verbose=0).argmax(axis=1)
    f1_val  = f1_score(y_va_seq, y_pred_va, average='weighted')
    f1_test = f1_score(y_te_seq, y_pred_te, average='weighted')

    pnl, sharpe = pnl_metrics(y_te_seq, y_pred_te, te['ret_future'].iloc[TIMESTEPS:])

    res = {
        "split"      : split,
        "train"      : f"{tr_start.date()}→{tr_end.date()}",
        "val"        : f"{(tr_end+pd.Timedelta(days=1)).date()}→{val_end.date()}",
        "test"       : f"{test_start.date()}→{test_end.date()}",
        "F1_val"     : f1_val,
        "F1_test"    : f1_test,
        "BalAcc_test": balanced_accuracy_score(y_te_seq, y_pred_te),
        "Precision"  : precision_score(y_te_seq, y_pred_te, average='weighted'),
        "Recall"     : recall_score(y_te_seq, y_pred_te, average='weighted'),
        "PnL_test"   : pnl,
        "Sharpe_test": sharpe
    }
    results.append(res)

    # save best by F1_val
    if f1_val > best_f1:
        best_f1  = f1_val
        best_set = {"model": model, "scaler": scaler, "thresholds": thres, "info": res}

# ──────────────────── SUMMARY & HOLD-OUT ──────────────────── #
results_df = pd.DataFrame(results)
print("\nWalk-forward summary:\n", results_df)

print("\nBest split:", best_set["info"])

# ── Evaluate on unseen hold-out 2023-2025
hd = df_holdout.copy()
hd['ret_future'] = compute_future_return(hd["SP500_historical_data_Close"], PRED_HORIZON)
hd.dropna(subset=['ret_future'], inplace=True)
hd['target'] = hd['ret_future'].apply(lambda r: label_from_thresholds(r, best_set["thresholds"]))
hd = hd.iloc[:-PRED_HORIZON].dropna()

X_hd = best_set["scaler"].transform(hd[FEATURES])
X_hd_seq, y_hd_seq = create_sequences(X_hd, hd['target'].values, TIMESTEPS)
y_hd_pred = best_set["model"].predict(X_hd_seq, verbose=0).argmax(axis=1)

print("\nHold-out metrics 2023-2025")
print("F1 :",     f1_score(y_hd_seq, y_hd_pred, average='weighted'))
print("BalAcc :", balanced_accuracy_score(y_hd_seq, y_hd_pred))

pnl_hd, sharpe_hd = pnl_metrics(y_hd_seq, y_hd_pred, hd['ret_future'].iloc[TIMESTEPS:])
print("PnL :", pnl_hd, "Sharpe :", sharpe_hd)


# ───────────────────────── POST-RUN EXPORTS ───────────────────────── #
# 1. tableau récap des splits
results_df = pd.DataFrame(results)
print("\n──────────────── Walk-forward splits summary ────────────────")
# print(results_df.to_string(index=False))

# 2. sauvegarde du meilleur modèle + méta-données
if best_set:
    print("\n=== Saving best model ===")
    print(f"Best F1_val : {best_f1:.4f}")
    print(f"Train window: {best_set['info']['train']}")
    print(f"Test window : {best_set['info']['test']}")

    os.makedirs("IA_training/model", exist_ok=True)
    MODEL_PATH  = "IA_training/model/best_lstm_model_v11.keras"
    CONFIG_PATH = "IA_training/model/model_config_v11.pkl"

    best_set["model"].save(MODEL_PATH)

    model_cfg = {
        "scaler"    : best_set["scaler"],
        "features"  : FEATURES,
        "timesteps" : TIMESTEPS,
        "thresholds": best_set["thresholds"],
        "info"      : best_set["info"]
    }
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(model_cfg, f)

else:
    raise RuntimeError("No model was trained – nothing to save.")

# ──────────────────────── INFERENCE UTILITIES ─────────────────────── #
from sklearn.metrics import (accuracy_score, confusion_matrix)  # compléments

def _seq_only(X, steps):
    """Fabrique des séquences sans étiquette (inference)."""
    return np.array([X[i-steps:i] for i in range(steps, len(X))], dtype=np.float32)

def load_and_predict(data_path: str,
                     version: str = "11",
                     date_col: str = "Date",
                     horizon: int = PRED_HORIZON):
    """
    Charge modèle + scaler, produit les prédictions
    et écrit un .csv + un résumé métriques.
    """
    model_path  = f"IA_training/model/best_lstm_model_v{version}.keras"
    config_path = f"IA_training/model/model_config_v{version}.pkl"
    out_csv     = f"prediction/predictions_v{version}.csv"
    os.makedirs("prediction", exist_ok=True)

    model  = tf.keras.models.load_model(model_path)
    config = pickle.load(open(config_path, "rb"))

    scaler     = config["scaler"]
    feats      = config["features"]
    timesteps  = config["timesteps"]
    thresholds = config["thresholds"]

    df = (pd.read_csv(data_path, parse_dates=[date_col])
            .sort_values(date_col)
            .reset_index(drop=True))

    df["ret_future"] = compute_future_return(df["SP500_historical_data_Close"], horizon)
    df["target"]     = df["ret_future"].apply(lambda r: label_from_thresholds(r, thresholds))
    df = df.iloc[:-horizon].dropna(subset=feats + ["target"])

    X      = scaler.transform(df[feats])
    X_seq  = _seq_only(X, timesteps)
    preds  = model.predict(X_seq, verbose=0).argmax(axis=1)

    eval_df = df.iloc[timesteps:].copy()
    eval_df["Prediction"] = preds
    eval_df.rename(columns={"target": "Actual"}, inplace=True)
    eval_df.to_csv(out_csv, index=False)

    # —— métriques classification
    y_true, y_pred = eval_df["Actual"].astype(int), eval_df["Prediction"].astype(int)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    cm   = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation hold-out ===")
    print(f"Accuracy : {acc:.4f} | F1 : {f1:.4f} | Precision : {prec:.4f} | Recall : {rec:.4f}")
    print("Confusion matrix :\n", cm)

    # —— export métriques
    met_path = out_csv.replace(".csv", "_metrics.txt")
    with open(met_path, "w") as f:
        f.write(f"Accuracy {acc:.4f}\nF1 {f1:.4f}\nPrecision {prec:.4f}\nRecall {rec:.4f}\n")
        f.write("\nConfusion matrix\n")
        f.write(np.array2string(cm))

    return eval_df


# ─────────────────────── BACKTEST PORTFOLIO ──────────────────────── #
def backtest_portfolio_multiclass(df: pd.DataFrame,
                                  initial_cash: float = 1_000,
                                  fee: float = 5e-4):
    """
    Back-tester simple : position longue/courte proportionnelle
    à la classe prédite. Mapping :
    0 → -80 % short, 1 → -40 % short, 2 → cash, 3 → +40 % long, 4 → +80 % long
    """
    size_map = {0: -0.8, 1: -0.4, 2: 0.0, 3: 0.4, 4: 0.8}

    cash, pos = initial_cash, 0.0
    port_vals = []

    for price, signal in zip(df["SP500_historical_data_Close"], df["Prediction"]):
        tgt_pct = size_map.get(signal, 0.0)
        tot_val = cash + pos * price
        tgt_pos_val = tot_val * tgt_pct
        diff_val = tgt_pos_val - pos * price

        if abs(diff_val) > 0:
            qty = diff_val / price

            if qty > 0:   # buy
                cost = qty * price * (1 + fee)
                qty_eff = min(qty, cash / (price * (1 + fee)))
                cash  -= qty_eff * price * (1 + fee)
                pos   += qty_eff

            else:         # sell
                qty_eff = min(-qty, pos)
                cash  += qty_eff * price * (1 - fee)
                pos   -= qty_eff

        port_vals.append(cash + pos * price)

    df_bt = df.copy()
    df_bt["PortfolioValue"] = port_vals
    df_bt["Returns"]        = df_bt["PortfolioValue"].pct_change().fillna(0)
    df_bt["CumReturn"]      = df_bt["PortfolioValue"] / initial_cash - 1

    pnl     = df_bt["PortfolioValue"].iloc[-1] - initial_cash
    pnl_pct = pnl / initial_cash
    mdd     = (df_bt["PortfolioValue"].cummax() - df_bt["PortfolioValue"])\
                .div(df_bt["PortfolioValue"].cummax()).max()
    sharpe  = (df_bt["Returns"].mean() / df_bt["Returns"].std()) * np.sqrt(252) \
                if df_bt["Returns"].std() else 0
    bench   = df_bt["SP500_historical_data_Close"].iloc[-1] / \
              df_bt["SP500_historical_data_Close"].iloc[0] - 1

    print("\n=== Backtest Portfolio Multi-Class ===")
    print(f"Final PnL          : {pnl:.2f} USD  ({pnl_pct:.2%})")
    print(f"S&P-500 benchmark  : {bench:.2%}")
    print(f"Max Drawdown       : {mdd:.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")

    return df_bt

# ──────────────────────── USAGE ──────────────────────── #
if __name__ == "__main__":
    version = input("Enter model version (default 11): ")
    df = load_and_predict("csv_data/consolidated_data/normalized_complete_data.csv", version=version)
    df_bt = backtest_portfolio_multiclass(df)
    df_bt.to_csv(f"prediction/backtest_results_v{version}.csv", index=False)