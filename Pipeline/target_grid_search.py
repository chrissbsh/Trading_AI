"""
Grid Search — Target Engineering Optimization
===============================================
Phase 1 : Évaluation statistique pure (sans modèle) de toutes les combinaisons
           (PRED_HORIZON × seuil) sur 3 métriques :
           - Équilibre des classes (objectif 30/40/30)
           - Persistance du signal (autocorrélation lag-1)
           - Stabilité temporelle (KL divergence sur sous-périodes)

Phase 2 : Entraînement LightGBM sur les top-3 combinaisons uniquement,
           avec évaluation holdout + backtest financier.
"""

import os
import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
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
from scipy.stats import entropy as kl_divergence

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
    TOP_N_FEATURES, N_CLASSES, CLASS_WEIGHT_BOOST,
    ECART_MIN,
)
from pipeline.feature_selection import select_top_features_shap
from pipeline.backtest import run_backtest

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# Grille de recherche
# ══════════════════════════════════════════════════════════════════════

HORIZONS   = [3, 5, 7, 10, 14]
THRESHOLDS = [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]

# Ratio cible pour l'équilibre des classes : 30% baisse / 40% neutre / 30% hausse
IDEAL_RATIO = np.array([0.30, 0.40, 0.30])

# Zone optimale pour l'autocorrélation du signal
AUTOCORR_IDEAL = 0.30  # centre de la zone [0.15, 0.50]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def create_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)
    run_id = datetime.now().strftime("target_grid_search_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def create_target_parametric(df, horizon, threshold):
    """
    Crée la colonne target avec des paramètres explicites (pas de lecture config).
    Stratégie fixe uniquement : seuils symétriques [-threshold, +threshold].
    """
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


# ══════════════════════════════════════════════════════════════════════
# Phase 1 — Métriques statistiques
# ══════════════════════════════════════════════════════════════════════

def compute_balance_score(target_series):
    """
    Distance L1 entre la distribution observée et le ratio idéal 30/40/30.
    Score = 0 → parfait.  Score = 1 → une classe a 100%.
    """
    counts = target_series.value_counts(normalize=True).sort_index()
    observed = np.array([counts.get(c, 0.0) for c in range(N_CLASSES)])
    return float(np.sum(np.abs(observed - IDEAL_RATIO)))


def compute_class_percentages(target_series):
    """Renvoie (pct_class0, pct_class1, pct_class2)."""
    counts = target_series.value_counts(normalize=True).sort_index()
    return tuple(counts.get(c, 0.0) for c in range(N_CLASSES))


def compute_persistence(target_series):
    """
    Autocorrélation du target à lag=1.
    Mesure si la classe à J est corrélée à la classe à J+1.
    - Élevée (>0.5) = signal trop persistant / redondant
    - Nulle (~0)    = bruit pur → inapprénable
    - Zone idéale   = [0.15, 0.50]
    """
    s = target_series.astype(float)
    if s.std() == 0:
        return 0.0
    return float(s.autocorr(lag=1))


def compute_temporal_stability(target_series, n_splits=4):
    """
    Découpe la série en n_splits sous-périodes et mesure la divergence KL
    entre chaque sous-période et la distribution globale.
    Renvoie (mean_kl, max_kl).
    """
    global_counts = target_series.value_counts(normalize=True).sort_index()
    global_dist = np.array([global_counts.get(c, 1e-10) for c in range(N_CLASSES)])
    global_dist = global_dist / global_dist.sum()  # normaliser

    splits = np.array_split(target_series.values, n_splits)
    kl_scores = []
    for split in splits:
        if len(split) < 10:
            continue
        split_s = pd.Series(split)
        split_counts = split_s.value_counts(normalize=True).sort_index()
        split_dist = np.array([split_counts.get(c, 1e-10) for c in range(N_CLASSES)])
        split_dist = split_dist / split_dist.sum()
        kl = float(kl_divergence(split_dist, global_dist))
        kl_scores.append(kl)

    if not kl_scores:
        return 0.0, 0.0
    return float(np.mean(kl_scores)), float(np.max(kl_scores))


def run_phase1(df_train):
    """
    Boucle sur toute la grille (horizon × seuil) et calcule les métriques statistiques.
    Retourne un DataFrame avec les résultats.
    """
    print("=" * 70)
    print("  PHASE 1 -- Evaluation statistique de la grille target")
    print("=" * 70)

    total = len(HORIZONS) * len(THRESHOLDS)
    results = []

    for i, (horizon, threshold) in enumerate(itertools.product(HORIZONS, THRESHOLDS)):
        pct = (i + 1) / total * 100
        print(f"\r  [{i+1}/{total}] ({pct:5.1f}%) horizon={horizon:2d}  seuil={threshold:.3f}", end="")

        df = create_target_parametric(df_train, horizon, threshold)
        target = df["target"]

        if len(target) < 100:
            print(f"  [!] Pas assez de samples ({len(target)}), skip")
            continue

        # Métriques
        pct_0, pct_1, pct_2 = compute_class_percentages(target)
        balance_score = compute_balance_score(target)
        autocorr = compute_persistence(target)
        persistence_score = abs(autocorr - AUTOCORR_IDEAL)
        mean_kl, max_kl = compute_temporal_stability(target)
        n_samples = len(target)

        # Nombre de classes effectives (certaines combinaisons éliminent une classe)
        n_classes_present = target.nunique()

        results.append({
            "horizon": horizon,
            "threshold": threshold,
            "n_samples": n_samples,
            "n_classes": n_classes_present,
            "pct_class0": round(pct_0, 4),
            "pct_class1": round(pct_1, 4),
            "pct_class2": round(pct_2, 4),
            "balance_score": round(balance_score, 4),
            "autocorr_lag1": round(autocorr, 4),
            "persistence_score": round(persistence_score, 4),
            "mean_kl": round(mean_kl, 4),
            "max_kl": round(max_kl, 4),
        })

    print()  # newline after progress
    df_results = pd.DataFrame(results)
    return df_results


def select_top_k(df_results, k=3):
    """
    Sélectionne les top-k combinaisons par score composite.
    Filtre d'abord les combinaisons avec 3 classes présentes.
    """
    # Filtrer : on veut 3 classes présentes
    valid = df_results[df_results["n_classes"] == 3].copy()

    if len(valid) == 0:
        print("  [!] Aucune combinaison avec 3 classes ! Relachement du filtre...")
        valid = df_results.copy()

    # Rangs (plus bas = meilleur)
    valid["rank_balance"] = valid["balance_score"].rank(method="min")
    valid["rank_persistence"] = valid["persistence_score"].rank(method="min")
    valid["rank_stability"] = valid["max_kl"].rank(method="min")
    valid["composite_rank"] = (
        valid["rank_balance"] + valid["rank_persistence"] + valid["rank_stability"]
    )
    valid = valid.sort_values("composite_rank")

    top_k = valid.head(k)
    return top_k


def plot_heatmaps(df_results, results_dir):
    """Génère des heatmaps (horizon vs seuil) pour chaque métrique."""
    metrics = [
        ("balance_score", "Equilibre des classes\n(distance au 30/40/30, bas=mieux)", "RdYlGn_r"),
        ("autocorr_lag1", "Autocorrelation lag-1\n(zone ideale 0.15-0.50)", "RdYlGn"),
        ("persistence_score", "Score persistance\n(distance a 0.30, bas=mieux)", "RdYlGn_r"),
        ("max_kl", "Stabilite temporelle (max KL)\n(bas=mieux)", "RdYlGn_r"),
        ("pct_class0", "% Classe 0 (Baisse)\n(objectif ~30%)", "coolwarm"),
        ("pct_class1", "% Classe 1 (Neutre)\n(objectif ~40%)", "coolwarm"),
        ("pct_class2", "% Classe 2 (Hausse)\n(objectif ~30%)", "coolwarm"),
    ]

    for metric, title, cmap in metrics:
        pivot = df_results.pivot(index="horizon", columns="threshold", values=metric)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Seuil (threshold)")
        ax.set_ylabel("Horizon (jours)")
        plt.tight_layout()

        safe_name = metric.replace(" ", "_")
        plt.savefig(os.path.join(results_dir, f"heatmap_{safe_name}.png"), dpi=150)
        plt.close()

    # Heatmap composite rank
    valid = df_results[df_results["n_classes"] == 3].copy()
    if len(valid) > 0:
        valid["rank_balance"] = valid["balance_score"].rank(method="min")
        valid["rank_persistence"] = valid["persistence_score"].rank(method="min")
        valid["rank_stability"] = valid["max_kl"].rank(method="min")
        valid["composite_rank"] = (
            valid["rank_balance"] + valid["rank_persistence"] + valid["rank_stability"]
        )
        pivot = valid.pivot(index="horizon", columns="threshold", values="composite_rank")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title("Score composite (rang, bas=mieux)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Seuil (threshold)")
        ax.set_ylabel("Horizon (jours)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "heatmap_composite_rank.png"), dpi=150)
        plt.close()

    print(f"  -> {len(metrics)+1} heatmaps sauvegardees dans {results_dir}")


# ══════════════════════════════════════════════════════════════════════
# Phase 2 — Entraînement LightGBM sur les top-k combinaisons
# ══════════════════════════════════════════════════════════════════════

def pad_proba(proba, model_classes, n_classes=N_CLASSES):
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    for col_idx, cls in enumerate(model_classes):
        full[:, int(cls)] = proba[:, col_idx]
    return full


def _evaluate_model(model_name, y_test, y_pred, proba, date_test,
                     model_dir, combo_name, features, model_obj=None):
    """Helper: evaluate a single model and save all artifacts. Returns metrics dict."""
    acc = accuracy_score(y_test, y_pred)
    bal = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  === HOLDOUT {model_name} -- {combo_name} ===")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Balanced Acc. : {bal:.4f}")
    print(f"  F1-Macro      : {f1m:.4f}")
    print(f"  F1-Weighted   : {f1w:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(model_dir, exist_ok=True)

    pd.DataFrame([{
        "model": model_name, "combo": combo_name,
        "accuracy": acc, "balanced_accuracy": bal,
        "f1_macro": f1m, "f1_weighted": f1w, "n_holdout": len(y_test),
    }]).to_csv(os.path.join(model_dir, "holdout_metrics.csv"), index=False)

    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(cr).transpose().to_csv(os.path.join(model_dir, "classification_report.csv"))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix -- {model_name} -- {combo_name}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh_cm = cm.max() / 2 if cm.size else 0
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            plt.text(jj, ii, format(cm[ii, jj], "d"), ha="center", va="center",
                     color="white" if cm[ii, jj] > thresh_cm else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    pred_df = pd.DataFrame({
        "date_prediction": pd.to_datetime(date_test),
        "y_true": y_test, "y_pred": y_pred,
        "proba_0": proba[:, 0], "proba_1": proba[:, 1], "proba_2": proba[:, 2],
        "top_proba": sorted_proba[:, 0], "second_proba": sorted_proba[:, 1],
        "confidence_gap": sorted_proba[:, 0] - sorted_proba[:, 1],
    })
    pred_df.to_csv(os.path.join(model_dir, "holdout_predictions.csv"), index=False)

    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": features, "importance": model_obj.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)

    return {"accuracy": acc, "balanced_accuracy": bal,
            "f1_macro": f1m, "f1_weighted": f1w, "pred_df": pred_df}


def run_phase2(df_raw, top_combinations, results_dir, feature_selection_on=True):
    """
    Pour chaque combinaison du top-k, entraine LightGBM ET XGBoost et evalue sur holdout.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2 -- Entrainement LightGBM + XGBoost")
    print("=" * 70)

    cols_to_drop = [
        "sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high",
    ]
    df_clean = df_raw.drop(columns=cols_to_drop, errors="ignore")
    comparison_rows = []

    for rank_idx, (_, combo) in enumerate(top_combinations.iterrows()):
        horizon = int(combo["horizon"])
        threshold = float(combo["threshold"])
        combo_name = f"h{horizon}_t{threshold:.3f}"

        print(f"\n  TOP-{rank_idx+1} : horizon={horizon}, seuil=+/-{threshold:.3f}")
        print(f"  Balance={combo['balance_score']:.3f}  Autocorr={combo['autocorr_lag1']:.3f}")

        combo_dir = os.path.join(results_dir, f"top{rank_idx+1}_{combo_name}")
        os.makedirs(combo_dir, exist_ok=True)

        # Split temporel
        df_main = df_clean[df_clean[DATE_COL] < HOLDOUT_START_DATE].copy()
        df_holdout = df_clean[
            (df_clean[DATE_COL] >= HOLDOUT_START_DATE) &
            (df_clean[DATE_COL] <= HOLDOUT_END_DATE)
        ].copy()
        df_main = create_target_parametric(df_main, horizon, threshold)
        df_holdout = create_target_parametric(df_holdout, horizon, threshold)
        print(f"  -> Train: {len(df_main)} | Holdout: {len(df_holdout)}")

        # Feature selection
        if feature_selection_on:
            print("  -> Selection SHAP features...")
            raw_feats = select_top_features_shap(
                df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
            features = [f for f in raw_feats if f not in (DATE_COL, "target", "ret_future")]
            if TARGET_PRICE_COL not in features:
                features.append(TARGET_PRICE_COL)
        else:
            features = [f for f in df_main.columns
                        if f not in (DATE_COL, "target", "ret_future")]
        print(f"  -> {len(features)} features retenues")

        # Split train / val
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

        # Class weights
        present_classes = np.unique(y_train)
        cw_raw = compute_class_weight("balanced", classes=present_classes, y=y_train)
        cw_dict = {c: float(w * CLASS_WEIGHT_BOOST.get(c, 1.0))
                   for c, w in zip(present_classes, cw_raw)}
        print(f"  -> Class weights: {cw_dict}")

        # ── LightGBM ──
        print("\n  [LightGBM] Entrainement...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, class_weight=cw_dict,
            random_state=SEED, verbosity=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(period=200)])
        lgb_proba = pad_proba(lgb_model.predict_proba(X_test), lgb_model.classes_)
        lgb_pred = lgb_model.predict(X_test)
        lgb_dir = os.path.join(combo_dir, "lightgbm")
        lgb_m = _evaluate_model("LightGBM", y_test, lgb_pred, lgb_proba,
                                date_test, lgb_dir, combo_name, features, lgb_model)

        # ── XGBoost ──
        print("\n  [XGBoost] Entrainement...")
        sw_train = np.array([cw_dict.get(c, 1.0) for c in y_train])
        sw_val = np.array([cw_dict.get(c, 1.0) for c in y_val])
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
        dtest = xgb.DMatrix(X_test)
        xgb_params = {
            "objective": "multi:softprob", "num_class": N_CLASSES,
            "eval_metric": "mlogloss", "learning_rate": 0.05,
            "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
            "seed": SEED, "verbosity": 0,
        }
        xgb_booster = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                                evals=[(dval, "val")], early_stopping_rounds=50,
                                verbose_eval=200)
        xgb_proba = xgb_booster.predict(dtest).reshape(-1, N_CLASSES)
        xgb_pred = xgb_proba.argmax(axis=1)
        xgb_dir = os.path.join(combo_dir, "xgboost")
        xgb_m = _evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba,
                                date_test, xgb_dir, combo_name, features, None)
        # XGBoost feature importance (gain-based)
        scores = xgb_booster.get_score(importance_type="gain")
        imp = pd.DataFrame([{"feature": f, "importance": scores.get(f, 0.0)}
                            for f in features]).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(xgb_dir, "feature_importance.csv"), index=False)

        # ── Backtest for both models ──
        import pipeline.config as cfg
        original_horizon = cfg.PRED_HORIZON
        cfg.PRED_HORIZON = horizon
        price_series = test_df.set_index(DATE_COL)[TARGET_PRICE_COL]

        for mname, mm, mdir in [("LightGBM", lgb_m, lgb_dir),
                                 ("XGBoost", xgb_m, xgb_dir)]:
            try:
                bt = run_backtest(mm["pred_df"], price_series, mdir)
                sharpe = float(bt["sharpe_ratio"].iloc[0])
                total_ret = float(bt["total_return"].iloc[0])
                max_dd = float(bt["max_drawdown"].iloc[0])
            except Exception as e:
                print(f"  [!] Backtest {mname} error: {e}")
                sharpe, total_ret, max_dd = 0.0, 0.0, 0.0
            comparison_rows.append({
                "rank": rank_idx + 1, "model": mname,
                "horizon": horizon, "threshold": threshold,
                "balance_score": combo["balance_score"],
                "autocorr_lag1": combo["autocorr_lag1"],
                "accuracy": round(mm["accuracy"], 4),
                "balanced_accuracy": round(mm["balanced_accuracy"], 4),
                "f1_macro": round(mm["f1_macro"], 4),
                "f1_weighted": round(mm["f1_weighted"], 4),
                "sharpe": round(sharpe, 3),
                "total_return_pct": round(total_ret * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
            })
        cfg.PRED_HORIZON = original_horizon

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(results_dir, "phase2_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("  COMPARAISON FINALE -- LightGBM vs XGBoost")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    return comp_df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(top_k=3, feature_selection_on=True):
    results_dir = create_results_dir()
    print(f"\n[*] Dossier resultats : {results_dir}\n")

    # ── Chargement données ──
    print(f"[*] Chargement : {DATA_FILE_PATH}")
    df_raw = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    print(f"  -> {len(df_raw)} lignes, {len(df_raw.columns)} colonnes")

    # Données train uniquement pour la Phase 1
    df_train_raw = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    print(f"  -> Train (< {HOLDOUT_START_DATE}) : {len(df_train_raw)} lignes")

    # ── Phase 1 ──
    grid_results = run_phase1(df_train_raw)
    grid_results.to_csv(os.path.join(results_dir, "phase1_grid_results.csv"), index=False)

    print(f"\n  -> {len(grid_results)} combinaisons evaluees")
    print(f"  -> Combinaisons avec 3 classes : {(grid_results['n_classes'] == 3).sum()}")

    # Heatmaps
    plot_heatmaps(grid_results, results_dir)

    # Sélection top-k
    top_k_df = select_top_k(grid_results, k=top_k)
    top_k_df.to_csv(os.path.join(results_dir, "phase1_top_k.csv"), index=False)

    print(f"\n  -- Top-{top_k} combinaisons (Phase 1) --")
    display_cols = [
        "horizon", "threshold", "pct_class0", "pct_class1", "pct_class2",
        "balance_score", "autocorr_lag1", "max_kl", "composite_rank",
    ]
    print(top_k_df[display_cols].to_string(index=False))

    # ── Phase 2 ──
    comp_df = run_phase2(df_raw, top_k_df, results_dir,
                         feature_selection_on=feature_selection_on)

    print(f"\n[*] Tous les resultats sauvegardes dans : {results_dir}")
    return grid_results, top_k_df, comp_df


if __name__ == "__main__":
    main(top_k=3, feature_selection_on=True)
