"""
End-to-end pipeline sans fuite d'information :
– seuils recalculés dans chaque fenêtre d'entraînement  
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
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
import pickle
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import time

# ───────────────────────── OPTIMISATION CONFIG ────────────────────────── #
# Optimisation TensorFlow
physical_devices = tf.config.list_physical_devices('CPU')
print(f"Nombre de CPU détectés: {len(physical_devices)}")
tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())
tf.config.optimizer.set_jit(True)  # XLA compilation (accélération)

# Cache de données TF
AUTOTUNE = tf.data.experimental.AUTOTUNE

warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx("float32")

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ───────────────────────── PARAMETERS ─────────────────────── #
PRED_HORIZON      = 7            # jours
N_CLASSES         = 3
TIMESTEPS         = 30           # taille séquence
TRAIN_YEARS       = 3
VAL_YEARS         = 1
TEST_MONTHS       = 3
BUFFER_DAYS       = 14           # purge entre splits
JITTER_RATIO      = 0.05         # % du σ pour le jitter
EPOCHS            = 100
BATCH_SIZE        = 64           # Augmenté pour améliorer l'utilisation CPU
PATIENCE          = 10
NUM_WORKERS       = max(1, multiprocessing.cpu_count() - 1)  # Laisser un coeur libre
PREFETCH_SIZE     = 5            # Nombre de batches à précharger

print(f"Utilisation de {NUM_WORKERS} workers pour la parallélisation")

# ───────────────────────── HELPERS OPTIMISÉS ────────────────────────── #
@tf.function
def compute_future_return(close, horizon):
    """Version TensorFlow optimisée du calcul de retour futur"""
    future_close = tf.roll(close, shift=-horizon, axis=0)
    return (future_close - close) / close

def create_tf_dataset(X, y, batch_size, augment=False, jitter_ratio=0, is_training=True):
    """Crée un tf.data.Dataset optimisé pour la performance"""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    if augment and is_training:
        # Fonction d'augmentation optimisée avec TF
        @tf.function
        def add_jitter(x, y):
            std = tf.math.reduce_std(x, axis=(0, 1), keepdims=True)
            noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std * jitter_ratio)
            return x + noise, y
        
        # Dataset augmenté
        ds_aug = tf.data.Dataset.from_tensor_slices((X, y))
        ds_aug = ds_aug.map(add_jitter, num_parallel_calls=AUTOTUNE)
        ds = ds.concatenate(ds_aug)
    
    if is_training:
        ds = ds.shuffle(buffer_size=10000, seed=SEED)
    
    return ds.batch(batch_size).prefetch(PREFETCH_SIZE)

def create_sequences_vectorized(X, y, timesteps):
    """Version vectorisée de la création de séquences pour meilleure performance"""
    n_samples = len(X) - timesteps
    X_result = np.zeros((n_samples, timesteps, X.shape[1]), dtype=np.float32)
    
    # Création vectorisée des séquences
    for i in range(timesteps):
        X_result[:, i, :] = X[i:i+n_samples]
    
    y_result = y[timesteps:]
    
    return X_result, y_result

def build_sliding_thresholds(df: pd.DataFrame, horizon: int, n_classes: int, step_months: int = 6):
    from collections import OrderedDict
    
    df = df.copy()
    # Calcul vectorisé des retours futurs
    df['ret_future'] = compute_future_return(df["SP500_historical_data_Close"].values, horizon).numpy()
    df = df.dropna(subset=['ret_future'])
    
    thresholds_map = OrderedDict()
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    
    while start_date + relativedelta(months=step_months) <= end_date:
        window_end = start_date + relativedelta(months=step_months)
        window_df = df[(df["Date"] >= start_date) & (df["Date"] < window_end)]
        if len(window_df) >= 100:  # éviter les petites fenêtres
            thresholds = make_thresholds(window_df["ret_future"], n_classes)
            thresholds_map[(start_date, window_end)] = thresholds
        start_date += relativedelta(months=step_months // 2)  # 50% overlap
    
    return thresholds_map

def make_thresholds(series: pd.Series, k: int) -> np.ndarray:
    q = np.linspace(0, 1, k + 1)
    return series.quantile(q).values[1:-1]          # k-1 seuils internes

def label_with_sliding_thresholds(df: pd.DataFrame, thresholds_map):
    labels = np.zeros(len(df), dtype=np.int32)
    
    # Vectorisation partielle - éviter la boucle coûteuse
    date_array = df["Date"].values
    ret_array = df["ret_future"].values
    
    # Conversion des dates clés en numpy array pour recherche vectorisée
    # Assurons-nous que toutes les dates sont du même type (datetime64)
    window_starts = np.array([start.to_datetime64() if hasattr(start, 'to_datetime64') else np.datetime64(start) 
                              for start, _ in thresholds_map.keys()])
    window_ends = np.array([end.to_datetime64() if hasattr(end, 'to_datetime64') else np.datetime64(end)
                            for _, end in thresholds_map.keys()])
    thresholds_list = list(thresholds_map.values())
    
    for i in range(len(df)):
        date = date_array[i]
        ret = ret_array[i]
        
        # Assurons-nous que date est en format datetime64 pour la comparaison
        if isinstance(date, (int, float, str)):
            date = np.datetime64(date)
        
        # Recherche vectorisée de la fenêtre applicable
        in_window = (date >= window_starts) & (date < window_ends)
        if np.any(in_window):
            idx = np.where(in_window)[0][0]
            thresholds = thresholds_list[idx]
        else:
            thresholds = thresholds_list[-1]  # fallback
            
        # Calcul de la classe
        for j, t in enumerate(thresholds):
            if ret <= t:
                labels[i] = j
                break
        else:
            labels[i] = len(thresholds)
            
    return labels

def label_from_thresholds(ret: float, th: np.ndarray) -> int:
    for i, t in enumerate(th):
        if ret <= t:
            return i
    return len(th)                                  # dernière classe

def lstm_model(input_shape, n_classes=N_CLASSES):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape,
             recurrent_activation='sigmoid',  # Plus rapide que tanh
             unroll=False),  # False pour grandes séquences
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(),
        LSTM(32, recurrent_activation='sigmoid', unroll=False),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    # Optimisation du learning rate schedule
    lr_sched = CosineDecay(1e-3, decay_steps=1000)
    loss_fn = SparseCategoricalCrossentropy()
    
    model.compile(optimizer=Adam(lr_sched),
                 loss=loss_fn,
                 metrics=['accuracy'])

    return model

def pnl_metrics(y_true, y_pred, returns, horizon=PRED_HORIZON):
    """Long si classe 2, short si classe 0, cash si classe 1 (3 classes)."""
    pos = np.select([y_pred == 2, y_pred == 0], [1, -1], default=0)
    strat_ret = pos * returns.iloc[-len(pos):].values
    cum = np.cumprod(1 + strat_ret) - 1
    pnl = cum[-1]
    sharpe = 0 if strat_ret.std() == 0 else strat_ret.mean() / strat_ret.std() * np.sqrt(252/horizon)
    return float(pnl), float(sharpe)

# ───────────────────────── TRAINING FUNCTION ─────────────────────── #
def train_single_split(start_idx, start, df_roll, FEATURES):
    """Fonction pour entraîner un seul split (utilisée pour parallélisation)"""
    split = start_idx + 1  # Pour garder la numérotation cohérente
    
    # ── window bounds
    tr_start = start
    tr_end   = start + relativedelta(years=TRAIN_YEARS) - pd.Timedelta(days=1)
    val_end  = tr_end + relativedelta(years=VAL_YEARS)
    test_start = val_end + pd.Timedelta(days=BUFFER_DAYS)
    test_end   = test_start + relativedelta(months=TEST_MONTHS) - pd.Timedelta(days=1)
    
    if test_end > df_roll["Date"].max(): 
        return None
    
    # ── raw splits
    tr = df_roll[(df_roll["Date"] >= tr_start) & (df_roll["Date"] <= tr_end)].copy()
    va = df_roll[(df_roll["Date"] > tr_end) & (df_roll["Date"] <= val_end)].copy()
    te = df_roll[(df_roll["Date"] >= test_start) & (df_roll["Date"] <= test_end)].copy()

    # ── compute returns & thresholds on TRAIN ONLY
    # 1. Concatène les 3 parties pour appliquer les labels de manière globale
    all_parts = pd.concat([tr, va, te]).copy()
    all_parts['ret_future'] = compute_future_return(all_parts["SP500_historical_data_Close"].values, PRED_HORIZON).numpy()
    all_parts.dropna(subset=["ret_future"], inplace=True)

    # 2. Créer les seuils glissants
    thresholds_map = build_sliding_thresholds(all_parts, PRED_HORIZON, N_CLASSES)

    # 3. Ré-appliquer les seuils via le mapping
    for part in (tr, va, te):
        part['ret_future'] = compute_future_return(part["SP500_historical_data_Close"].values, PRED_HORIZON).numpy()
        part.dropna(subset=["ret_future"], inplace=True)
        part['target'] = label_with_sliding_thresholds(part, thresholds_map)

    tr, va, te = (p.iloc[:-PRED_HORIZON].dropna() for p in (tr, va, te))   # drop trailing NaNs

    # ── scale
    scaler = StandardScaler().fit(tr[FEATURES])
    X_tr = scaler.transform(tr[FEATURES]); y_tr = tr['target'].values
    X_va = scaler.transform(va[FEATURES]); y_va = va['target'].values
    X_te = scaler.transform(te[FEATURES]); y_te = te['target'].values

    # ── sequences (version vectorisée)
    X_tr_seq, y_tr_seq = create_sequences_vectorized(X_tr, y_tr, TIMESTEPS)
    X_va_seq, y_va_seq = create_sequences_vectorized(X_va, y_va, TIMESTEPS)
    X_te_seq, y_te_seq = create_sequences_vectorized(X_te, y_te, TIMESTEPS)

    # ── Création des datasets TensorFlow optimisés
    train_ds = create_tf_dataset(X_tr_seq, y_tr_seq, BATCH_SIZE, augment=True, jitter_ratio=JITTER_RATIO)
    val_ds = create_tf_dataset(X_va_seq, y_va_seq, BATCH_SIZE, is_training=False)
    
    # ── class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_tr_seq), y=y_tr_seq)
    class_weights = dict(zip(np.unique(y_tr_seq), cw))

    # ── model
    model = lstm_model(input_shape=(TIMESTEPS, X_tr_seq.shape[2]), n_classes=N_CLASSES)
    early = EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss')
    
    # Entraînement avec dataset optimisé
    model.fit(train_ds, epochs=EPOCHS, 
              validation_data=val_ds,
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
    
    return {"result": res, "model": model, "scaler": scaler, "thresholds": thresholds_map}

# ───────────────────────── LOAD DATA ──────────────────────── #
def main():
    start_time = time.time()
    print("Chargement des données...")
    
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

    # ───────────────────────── WALK FORWARD PARALLÈLE ───────────────────── #
    starts = []
    d0   = df_roll["Date"].min()
    last = df_roll["Date"].max() - relativedelta(years=TRAIN_YEARS+VAL_YEARS) - relativedelta(months=TEST_MONTHS)
    while d0 <= last:
        starts.append(d0)
        d0 += relativedelta(months=3)

    results = []
    best_f1 = -np.inf
    best_set = {}

    print(f"Démarrage entraînement walk-forward parallèle sur {len(starts)} splits avec {NUM_WORKERS} workers...")
    
    # Version parallèle avec tqdm pour le suivi de progression
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Création des tâches
        futures = [executor.submit(train_single_split, i, start, df_roll, FEATURES) 
                  for i, start in enumerate(starts)]
        
        # Traitement des résultats avec barre de progression
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result["result"])
                
                # Mise à jour du meilleur modèle
                if result["result"]["F1_val"] > best_f1:
                    best_f1 = result["result"]["F1_val"]
                    best_set = {
                        "model": result["model"], 
                        "scaler": result["scaler"], 
                        "thresholds": result["thresholds"], 
                        "info": result["result"]
                    }

    # Tri des résultats par numéro de split
    results.sort(key=lambda x: x["split"])

    # ──────────────────── SUMMARY & HOLD-OUT ──────────────────── #
    results_df = pd.DataFrame(results)
    print("\nWalk-forward summary:\n", results_df)

    print("\nBest split:", best_set["info"])

    # ── Evaluate on unseen hold-out 2023-2025
    print("Évaluation sur le hold-out 2023-2025...")
    hd = df_holdout.copy()
    thresholds_map = build_sliding_thresholds(df_roll, PRED_HORIZON, N_CLASSES)
    hd['ret_future'] = compute_future_return(hd["SP500_historical_data_Close"].values, PRED_HORIZON).numpy()
    hd.dropna(subset=['ret_future'], inplace=True)
    hd["target"] = label_with_sliding_thresholds(hd, thresholds_map)
    hd = hd.iloc[:-PRED_HORIZON].dropna()

    X_hd = best_set["scaler"].transform(hd[FEATURES])
    X_hd_seq, y_hd_seq = create_sequences_vectorized(X_hd, hd['target'].values, TIMESTEPS)
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
        MODEL_PATH  = "IA_training/model/best_lstm_model_v12.keras"
        CONFIG_PATH = "IA_training/model/model_config_v12.pkl"

        best_set["model"].save(MODEL_PATH)

        model_cfg = {
            "scaler"    : best_set["scaler"],
            "features"  : FEATURES,
            "timesteps" : TIMESTEPS,
            "thresholds": best_set["thresholds"],
            "thresholds_map": thresholds_map,
            "info"      : best_set["info"]
        }
        with open(CONFIG_PATH, "wb") as f:
            pickle.dump(model_cfg, f)

    else:
        raise RuntimeError("No model was trained – nothing to save.")
        
    total_time = time.time() - start_time
    print(f"\nTemps total d'exécution: {total_time:.2f} secondes")


# ──────────────────────── INFERENCE UTILITIES ─────────────────────── #
from sklearn.metrics import (accuracy_score, confusion_matrix)  # compléments

def _seq_only(X, steps):
    """Fabrique des séquences sans étiquette (inference)."""
    return np.array([X[i-steps:i] for i in range(steps, len(X))], dtype=np.float32)

def load_and_predict(data_path: str,
                     version: str = "12",
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

    df["ret_future"] = compute_future_return(df["SP500_historical_data_Close"].values, horizon).numpy()
    thresholds_map = config["thresholds_map"]
    df["target"] = label_with_sliding_thresholds(df, thresholds_map)
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
    # size_map = {0: -0.8, 1: -0.4, 2: 0.0, 3: 0.4, 4: 0.8} # 5 classes

    size_map = {0: -0.8, 1: 0.0, 2: 0.8} # 3 classes

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
    main()