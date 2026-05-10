"""
Grid Search — CLASS_WEIGHT_BOOST Optimization
==============================================
Teste une grille de combinaisons de poids de classes (boost sur balanced)
avec les parametres target fixes (h=3, t=0.008).
Entraine LightGBM + XGBoost pour chaque combinaison et compare.
"""

import os
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score,
)

from pipeline.config import (
    DATA_FILE_PATH, DATE_COL, TARGET_PRICE_COL,
    HOLDOUT_START_DATE, HOLDOUT_END_DATE,
    TOP_N_FEATURES, N_CLASSES, PRED_HORIZON, FIXED_THRESHOLDS,
)
from pipeline.feature_selection import select_top_features_shap
from pipeline.backtest import run_backtest

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# Grille de poids
# ══════════════════════════════════════════════════════════════════════

# Poids a tester pour chaque classe (appliques PAR-DESSUS balanced)
W0_VALUES = [0.8, 1.0, 1.2, 1.5]   # classe 0 (baisse)
W1_VALUES = [0.6, 0.8, 0.9, 1.0]   # classe 1 (neutre — majoritaire)
W2_VALUES = [0.8, 1.0, 1.2, 1.5]   # classe 2 (hausse)

# Total: 4 x 4 x 4 = 64 combinaisons


def create_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)
    run_id = datetime.now().strftime("weight_grid_search_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def create_target_fixed(df, horizon, threshold):
    df = df.copy()
    df["ret_future"] = (
        df[TARGET_PRICE_COL].shift(-horizon) - df[TARGET_PRICE_COL]
    ) / df[TARGET_PRICE_COL]
    df.dropna(subset=["ret_future"], inplace=True)
    lo, hi = -threshold, threshold
    df["target"] = df["ret_future"].apply(
        lambda x: 0 if x < lo else (2 if x > hi else 1)
    )
    return df


def pad_proba(proba, model_classes, n_classes=N_CLASSES):
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    for col_idx, cls in enumerate(model_classes):
        full[:, int(cls)] = proba[:, col_idx]
    return full


def main():
    results_dir = create_results_dir()
    print(f"[*] Dossier resultats : {results_dir}")

    threshold = abs(FIXED_THRESHOLDS[0])
    horizon = PRED_HORIZON
    print(f"[*] Parametres target fixes : horizon={horizon}, seuil=+/-{threshold}")

    # Chargement
    df_raw = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    df_main = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    df_holdout = df_raw[
        (df_raw[DATE_COL] >= HOLDOUT_START_DATE) &
        (df_raw[DATE_COL] <= HOLDOUT_END_DATE)
    ].copy()

    df_main = create_target_fixed(df_main, horizon, threshold)
    df_holdout = create_target_fixed(df_holdout, horizon, threshold)
    print(f"  -> Train: {len(df_main)} | Holdout: {len(df_holdout)}")

    # Feature selection (une seule fois, partagee)
    print("[*] Selection SHAP features...")
    raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
    features = [f for f in raw_feats if f not in (DATE_COL, "target", "ret_future")]
    if TARGET_PRICE_COL not in features:
        features.append(TARGET_PRICE_COL)
    print(f"  -> {len(features)} features")

    # Prepare data
    split_idx = int(0.8 * len(df_main))
    train_df, val_df, test_df = df_main.iloc[:split_idx], df_main.iloc[split_idx:], df_holdout

    X_train, y_train = train_df[features], train_df["target"].values
    X_val, y_val = val_df[features], val_df["target"].values
    X_test, y_test = test_df[features], test_df["target"].values
    date_test = test_df[DATE_COL].values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

    # Balanced base weights
    present_classes = np.unique(y_train)
    cw_raw = compute_class_weight("balanced", classes=present_classes, y=y_train)
    base_weights = {c: float(w) for c, w in zip(present_classes, cw_raw)}
    print(f"  -> Balanced base weights: {base_weights}")

    price_series = test_df.set_index(DATE_COL)[TARGET_PRICE_COL]

    # Grid search
    grid = list(itertools.product(W0_VALUES, W1_VALUES, W2_VALUES))
    total = len(grid)
    print(f"\n[*] Grid search : {total} combinaisons de poids")
    print("=" * 70)

    all_results = []

    for idx, (w0, w1, w2) in enumerate(grid):
        boost = {0: w0, 1: w1, 2: w2}
        cw_dict = {c: base_weights[c] * boost.get(c, 1.0) for c in present_classes}
        label = f"w0={w0}_w1={w1}_w2={w2}"

        pct = (idx + 1) / total * 100
        print(f"\r  [{idx+1}/{total}] ({pct:5.1f}%) {label}", end="", flush=True)

        # ── LightGBM ──
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, class_weight=cw_dict,
            random_state=SEED, verbosity=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(period=-1)])

        lgb_proba = pad_proba(lgb_model.predict_proba(X_test), lgb_model.classes_)
        lgb_pred = lgb_model.predict(X_test)

        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_bal = balanced_accuracy_score(y_test, lgb_pred)
        lgb_f1m = f1_score(y_test, lgb_pred, average="macro", zero_division=0)
        lgb_r0 = float((lgb_pred[y_test == 0] == 0).mean()) if (y_test == 0).sum() > 0 else 0
        lgb_r1 = float((lgb_pred[y_test == 1] == 1).mean()) if (y_test == 1).sum() > 0 else 0
        lgb_r2 = float((lgb_pred[y_test == 2] == 2).mean()) if (y_test == 2).sum() > 0 else 0

        # LightGBM backtest
        sorted_p = np.sort(lgb_proba, axis=1)[:, ::-1]
        pred_df = pd.DataFrame({
            "date_prediction": pd.to_datetime(date_test),
            "y_true": y_test, "y_pred": lgb_pred,
            "proba_0": lgb_proba[:, 0], "proba_1": lgb_proba[:, 1], "proba_2": lgb_proba[:, 2],
            "top_proba": sorted_p[:, 0], "second_proba": sorted_p[:, 1],
            "confidence_gap": sorted_p[:, 0] - sorted_p[:, 1],
        })
        try:
            import pipeline.config as cfg
            orig = cfg.PRED_HORIZON
            cfg.PRED_HORIZON = horizon
            bt = run_backtest(pred_df, price_series, results_dir)
            lgb_sharpe = float(bt["sharpe_ratio"].iloc[0])
            lgb_ret = float(bt["total_return"].iloc[0])
            cfg.PRED_HORIZON = orig
        except:
            lgb_sharpe, lgb_ret = 0.0, 0.0

        # ── XGBoost ──
        sw_train = np.array([cw_dict.get(c, 1.0) for c in y_train])
        sw_val = np.array([cw_dict.get(c, 1.0) for c in y_val])
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
        dtest = xgb.DMatrix(X_test)

        xgb_booster = xgb.train(
            {"objective": "multi:softprob", "num_class": N_CLASSES,
             "eval_metric": "mlogloss", "learning_rate": 0.05,
             "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
             "seed": SEED, "verbosity": 0},
            dtrain, num_boost_round=500,
            evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=0,
        )
        xgb_proba = xgb_booster.predict(dtest).reshape(-1, N_CLASSES)
        xgb_pred = xgb_proba.argmax(axis=1)

        xgb_bal = balanced_accuracy_score(y_test, xgb_pred)
        xgb_f1m = f1_score(y_test, xgb_pred, average="macro", zero_division=0)
        xgb_r0 = float((xgb_pred[y_test == 0] == 0).mean()) if (y_test == 0).sum() > 0 else 0
        xgb_r1 = float((xgb_pred[y_test == 1] == 1).mean()) if (y_test == 1).sum() > 0 else 0
        xgb_r2 = float((xgb_pred[y_test == 2] == 2).mean()) if (y_test == 2).sum() > 0 else 0

        all_results.append({
            "w0": w0, "w1": w1, "w2": w2,
            "lgb_bal_acc": round(lgb_bal, 4), "lgb_f1_macro": round(lgb_f1m, 4),
            "lgb_recall_0": round(lgb_r0, 4), "lgb_recall_1": round(lgb_r1, 4),
            "lgb_recall_2": round(lgb_r2, 4), "lgb_sharpe": round(lgb_sharpe, 3),
            "lgb_return_pct": round(lgb_ret * 100, 2),
            "xgb_bal_acc": round(xgb_bal, 4), "xgb_f1_macro": round(xgb_f1m, 4),
            "xgb_recall_0": round(xgb_r0, 4), "xgb_recall_1": round(xgb_r1, 4),
            "xgb_recall_2": round(xgb_r2, 4),
        })

    print("\n")

    # Resultats
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(results_dir, "weight_grid_results.csv"), index=False)

    # Score composite: on veut maximiser bal_acc + f1_macro + min(recalls) > 0.10
    # Penaliser les combos ou une classe a recall < 10%
    for prefix in ["lgb", "xgb"]:
        min_recall = df_res[[f"{prefix}_recall_0", f"{prefix}_recall_1", f"{prefix}_recall_2"]].min(axis=1)
        df_res[f"{prefix}_min_recall"] = min_recall
        df_res[f"{prefix}_score"] = (
            df_res[f"{prefix}_bal_acc"] + df_res[f"{prefix}_f1_macro"]
            + 0.5 * min_recall  # bonus for balanced recalls
            - 0.5 * (min_recall < 0.05).astype(float)  # penalty if any class < 5%
        )

    # Top 5 LightGBM
    lgb_top = df_res.nlargest(5, "lgb_score")
    print("=" * 70)
    print("  TOP 5 LightGBM (par score composite)")
    print("=" * 70)
    lgb_cols = ["w0", "w1", "w2", "lgb_bal_acc", "lgb_f1_macro",
                "lgb_recall_0", "lgb_recall_1", "lgb_recall_2",
                "lgb_sharpe", "lgb_return_pct", "lgb_score"]
    print(lgb_top[lgb_cols].to_string(index=False))

    # Top 5 XGBoost
    xgb_top = df_res.nlargest(5, "xgb_score")
    print(f"\n{'='*70}")
    print("  TOP 5 XGBoost (par score composite)")
    print("=" * 70)
    xgb_cols = ["w0", "w1", "w2", "xgb_bal_acc", "xgb_f1_macro",
                "xgb_recall_0", "xgb_recall_1", "xgb_recall_2", "xgb_score"]
    print(xgb_top[xgb_cols].to_string(index=False))

    # Heatmap: score moyen par (w0, w2) pour LightGBM (aggrege sur w1)
    for prefix, model_name in [("lgb", "LightGBM"), ("xgb", "XGBoost")]:
        pivot = df_res.groupby(["w0", "w2"])[f"{prefix}_score"].mean().reset_index()
        pivot = pivot.pivot(index="w0", columns="w2", values=f"{prefix}_score")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"Score composite {model_name} (moyenne sur w1)")
        ax.set_xlabel("w2 (hausse)")
        ax.set_ylabel("w0 (baisse)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"heatmap_{prefix}_w0_w2.png"), dpi=150)
        plt.close()

    # Best overall
    best_lgb = df_res.loc[df_res["lgb_score"].idxmax()]
    best_xgb = df_res.loc[df_res["xgb_score"].idxmax()]
    print(f"\n{'='*70}")
    print(f"  MEILLEURE CONFIG LightGBM : w0={best_lgb['w0']}, w1={best_lgb['w1']}, w2={best_lgb['w2']}")
    print(f"    BalAcc={best_lgb['lgb_bal_acc']:.4f}  F1m={best_lgb['lgb_f1_macro']:.4f}  "
          f"R0={best_lgb['lgb_recall_0']:.2f} R1={best_lgb['lgb_recall_1']:.2f} R2={best_lgb['lgb_recall_2']:.2f}  "
          f"Sharpe={best_lgb['lgb_sharpe']:.2f}")
    print(f"\n  MEILLEURE CONFIG XGBoost  : w0={best_xgb['w0']}, w1={best_xgb['w1']}, w2={best_xgb['w2']}")
    print(f"    BalAcc={best_xgb['xgb_bal_acc']:.4f}  F1m={best_xgb['xgb_f1_macro']:.4f}  "
          f"R0={best_xgb['xgb_recall_0']:.2f} R1={best_xgb['xgb_recall_1']:.2f} R2={best_xgb['xgb_recall_2']:.2f}")
    print("=" * 70)

    print(f"\n[*] Resultats sauvegardes dans : {results_dir}")
    return df_res


if __name__ == "__main__":
    main()
