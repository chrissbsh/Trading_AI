import os
import sys
import io
import json

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score,
)

from pipeline.config import *
from pipeline.feature_selection import select_top_features_shap
from pipeline.backtest import run_backtest

XGB_VERSION = "xgb_baseline_v1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)



# ──────────────────────────────────────────────────────────────────────
# Helpers partagés avec le LSTM (dupliqués pour autonomie du script)
# ──────────────────────────────────────────────────────────────────────

def create_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)
    run_id = datetime.now().strftime(f"{XGB_VERSION}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(results_dir, run_params, best_lgb_params=None):
    """Sauvegarde la config complète du run : paramètres pipeline + hyperparamètres modèle."""
    config = {
        "run_id": os.path.basename(results_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Paramètres pipeline (depuis config.py)
        "pipeline": {
            "pred_horizon": PRED_HORIZON,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "fixed_thresholds": list(FIXED_THRESHOLDS),
            "sequence_length": SEQUENCE_LENGTH,
            "top_n_features": TOP_N_FEATURES,
            "class_weight_boost": CLASS_WEIGHT_BOOST,
            "ecart_min": ECART_MIN,
            "holdout_start": HOLDOUT_START_DATE,
            "holdout_end": HOLDOUT_END_DATE,
            "model_version": MODEL_VERSION,
        },
        # Paramètres d'exécution (flags passés à main())
        "run_flags": run_params,
        # Meilleurs hyperparamètres LightGBM trouvés par Optuna (None si tune=False)
        "lightgbm_best_params": best_lgb_params,
    }
    path = os.path.join(results_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"🔹 Config run sauvegardée : {path}")


def create_target(df, tag=""):
    print(f"🔹 Création de la cible (horizon = {PRED_HORIZON}) {tag}")
    df = df.copy()
    df["ret_future"] = (
        df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]
    ) / df[TARGET_PRICE_COL]
    df.dropna(subset=["ret_future"], inplace=True)

    if THRESHOLD_STRATEGY == "fixed":
        lo, hi = FIXED_THRESHOLDS
        df["target"] = df["ret_future"].apply(
            lambda x: 0 if x < lo else (2 if x > hi else 1)
        )
        print(f"   → Seuils fixes : [{lo}, {hi}]")
    elif THRESHOLD_STRATEGY == "adaptive":
        rolling_vol = df["ret_future"].abs().rolling(21, min_periods=5).mean()
        vol_pct = rolling_vol.rank(pct=True).fillna(0.5)
        lo_base, _ = FIXED_THRESHOLDS
        thresh = lo_base * (1 + 0.5 * (vol_pct - 0.5))
        df["target"] = df.apply(
            lambda r: 0 if r["ret_future"] < -thresh[r.name]
            else (2 if r["ret_future"] > thresh[r.name] else 1),
            axis=1,
        )
        print(f"   → Seuils adaptatifs (base {lo_base:.3f})")
    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}")

    print("   → Distribution classes :", df["target"].value_counts(normalize=True).to_dict())
    return df


def add_lag_features(df, features, lags=(1, 5, 21)):
    """
    Ajoute des features décalées (lag) pour capturer la dynamique temporelle.
    Équivalent léger de la mémoire LSTM, sans la complexité d'architecture.
    Retourne le df enrichi et la liste des colonnes features mise à jour.
    """
    df = df.copy()
    new_cols = []
    for feat in features:
        if feat not in df.columns:
            continue
        for lag in lags:
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df[feat].shift(lag)
            new_cols.append(col_name)
    df.dropna(subset=new_cols, inplace=True)
    return df, features + new_cols


def pad_proba(proba, model_classes, n_classes=N_CLASSES):
    """Expand proba from (n, len(model_classes)) to (n, n_classes), filling missing columns with 0."""
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    for col_idx, cls in enumerate(model_classes):
        full[:, int(cls)] = proba[:, col_idx]
    return full


def save_confusion_matrix_plot(cm, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_prediction_distribution_plot(y_true, y_pred, output_path):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, [np.sum(y_true == c) for c in classes], width, label="True")
    plt.bar(x + width / 2, [np.sum(y_pred == c) for c in classes], width, label="Pred")
    plt.xticks(x, classes)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("True vs Predicted class distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance(model, feature_names, output_dir, model_name):
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(output_dir, f"{model_name}_feature_importance.csv"), index=False)

        plt.figure(figsize=(10, max(6, len(imp) * 0.3)))
        plt.barh(imp["feature"][:30][::-1], imp["importance"][:30][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top features — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"), dpi=150)
        plt.close()


def calibrate_and_threshold(model, X_cal, y_cal, X_test, thresholds=None):
    """
    Calibre les probabilités (isotonic regression sur X_cal/y_cal),
    puis applique des seuils asymétriques par classe.

    thresholds : dict {classe: seuil_min} ou None (argmax classique).
    Retourne (proba_calibrée, y_pred_final).
    """
    cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal_model.fit(X_cal, y_cal)
    proba_cal = pad_proba(cal_model.predict_proba(X_test), cal_model.classes_)

    if thresholds is None:
        return proba_cal, proba_cal.argmax(axis=1)

    n = proba_cal.shape[0]
    preds = np.ones(n, dtype=int)  # défaut → classe 1 (neutre)
    for i in range(n):
        best_cls, best_p = -1, -1.0
        for cls in range(N_CLASSES):
            p = proba_cal[i, cls]
            thr = thresholds.get(cls, 0.0)
            if p >= thr and p > best_p:
                best_cls, best_p = cls, p
        if best_cls != -1:
            preds[i] = best_cls
    return proba_cal, preds


def evaluate_and_save(model_name, y_true, y_pred, proba, date_index,
                      results_dir, test_df, version,
                      y_pred_cal=None, proba_cal=None):
    """
    Évalue et sauvegarde les résultats holdout.
    Si y_pred_cal/proba_cal fournis, sauvegarde aussi les métriques calibrées
    et lance le backtest sur les prédictions calibrées.
    """
    sub = os.path.join(results_dir, model_name)
    os.makedirs(sub, exist_ok=True)

    def _metrics_row(yt, yp, tag):
        return {
            "variant": tag,
            "model_version": version,
            "accuracy": accuracy_score(yt, yp),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
            "n_holdout_samples": len(yt),
        }

    rows = [_metrics_row(y_true, y_pred, "raw")]
    if y_pred_cal is not None:
        rows.append(_metrics_row(y_true, y_pred_cal, "calibrated"))

    # Affichage console — version principale (calibrée si dispo, sinon brute)
    y_show = y_pred_cal if y_pred_cal is not None else y_pred
    tag_show = "calibrated" if y_pred_cal is not None else "raw"
    r = rows[-1]
    print(f"\n===== HOLDOUT — {model_name} [{tag_show}] =====")
    print(f"   Accuracy        : {r['accuracy']:.4f}")
    print(f"   Balanced Acc.   : {r['balanced_accuracy']:.4f}")
    print(f"   F1-Macro        : {r['f1_macro']:.4f}")
    print(f"   F1-Weighted     : {r['f1_weighted']:.4f}")
    print(classification_report(y_true, y_show, zero_division=0))
    if y_pred_cal is not None:
        r_raw = rows[0]
        print(f"   [raw]  BalAcc={r_raw['balanced_accuracy']:.4f}  F1={r_raw['f1_macro']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(sub, "holdout_metrics.csv"), index=False)

    classif_dict = classification_report(y_true, y_show, output_dict=True, zero_division=0)
    pd.DataFrame(classif_dict).transpose().to_csv(
        os.path.join(sub, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_show)
    save_confusion_matrix_plot(cm, os.path.join(sub, "confusion_matrix.png"))
    save_prediction_distribution_plot(y_true, y_show,
        os.path.join(sub, "class_distribution_true_vs_pred.png"))

    # Backtest sur la version calibrée si dispo, sinon brute
    p_use = proba_cal if proba_cal is not None else proba
    sorted_proba = np.sort(p_use, axis=1)[:, ::-1]
    pred_df = pd.DataFrame({
        "date_prediction": pd.to_datetime(date_index),
        "y_true": y_true,
        "y_pred": y_show,
        "proba_0": p_use[:, 0],
        "proba_1": p_use[:, 1],
        "proba_2": p_use[:, 2],
        "top_proba": sorted_proba[:, 0],
        "second_proba": sorted_proba[:, 1],
        "confidence_gap": sorted_proba[:, 0] - sorted_proba[:, 1],
    })
    pred_df.to_csv(os.path.join(sub, "holdout_predictions.csv"), index=False)

    price_series = test_df.set_index(DATE_COL)[TARGET_PRICE_COL]
    run_backtest(pred_df, price_series, sub)

    return r["f1_macro"]


# ──────────────────────────────────────────────────────────────────────
# Optuna objective — LightGBM, optimise F1-Macro sur TimeSeriesSplit
# ──────────────────────────────────────────────────────────────────────

def _lgb_objective(trial, X, y):
    params = {
        # Axe A : max_depth >= 5 pour éviter les arbres trop faibles (depth=3 crée un biais classe unique)
        # Axe D : plages resserrées autour des meilleurs runs connus (n_est~1300, lr~0.05-0.17, depth=5-8)
        "n_estimators":      trial.suggest_int("n_estimators", 800, 2000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 5, 9),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 80),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "random_state": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        fold_classes = np.unique(y_tr)
        fold_cw_raw = compute_class_weight("balanced", classes=fold_classes, y=y_tr)
        fold_cw = {c: float(w * CLASS_WEIGHT_BOOST.get(c, 1.0))
                   for c, w in zip(fold_classes, fold_cw_raw)}
        m = lgb.LGBMClassifier(**params, class_weight=fold_cw)
        m.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_vl)
        f1m = f1_score(y_vl, preds, average="macro", zero_division=0)
        # Axe A : pénalise si une classe n'est jamais prédite (recall = 0 sur une classe présente)
        present = np.unique(y_vl)
        recalls = [f1_score(y_vl, preds, labels=[c], average="macro", zero_division=0)
                   for c in present]
        min_recall_penalty = 0.1 * (1.0 - min(recalls))
        scores.append(f1m - min_recall_penalty)
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────────────────────────────

def run_walk_forward(df_main, features, results_dir, n_folds=5,
                     max_train_rows=1000, time_weight_halflife=252, time_weighting=False):
    """
    Walk-forward avec fenêtre glissante (rolling window) optionnelle et pondération temporelle.

    - max_train_rows : taille max de la fenêtre de train. Actif uniquement si time_weighting=True.
    - time_weighting : active la pondération exponentielle et la rolling window (axe 3).
    """
    mode = f"rolling window={max_train_rows}, halflife={time_weight_halflife}j" if time_weighting else "expanding window"
    print(f"\n🔹 Walk-forward validation ({n_folds} folds, {mode})...")
    fold_size = len(df_main) // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        train_end   = fold_size * (fold + 1)
        test_start  = train_end
        test_end    = test_start + fold_size

        # Rolling window (axe 3) : on garde seulement les max_train_rows dernières lignes
        if time_weighting:
            train_start = max(0, train_end - max_train_rows)
        else:
            train_start = 0  # expanding window classique
        df_tr = df_main.iloc[train_start:train_end]
        df_te = df_main.iloc[test_start:test_end]

        if len(df_tr) < 100 or len(df_te) < 20:
            continue

        date_min = df_te[DATE_COL].iloc[0].date() if DATE_COL in df_te.columns else "?"
        date_max = df_te[DATE_COL].iloc[-1].date() if DATE_COL in df_te.columns else "?"
        date_tr_start = df_tr[DATE_COL].iloc[0].date() if DATE_COL in df_tr.columns else "?"

        X_tr = df_tr[features]
        y_tr = df_tr["target"].values
        X_te = df_te[features]
        y_te = df_te["target"].values

        sc = StandardScaler()
        X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=features)
        X_te = pd.DataFrame(sc.transform(X_te),     columns=features)

        present_classes = np.unique(y_tr)
        cw_raw_f = compute_class_weight("balanced", classes=present_classes, y=y_tr)
        fold_cw  = {c: float(w * CLASS_WEIGHT_BOOST.get(c, 1.0))
                    for c, w in zip(present_classes, cw_raw_f)}

        if time_weighting:
            n = len(X_tr)
            decay = np.exp(np.log(0.5) / time_weight_halflife * np.arange(n - 1, -1, -1))
            time_w = decay / decay.sum() * n
            sample_w = np.array([fold_cw.get(c, 1.0) for c in y_tr]) * time_w
        else:
            sample_w = np.array([fold_cw.get(c, 1.0) for c in y_tr])

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        # val interne = derniers 20% du train pour early stopping
        val_cut = int(0.8 * len(X_tr))
        model.fit(
            X_tr.iloc[:val_cut], y_tr[:val_cut],
            sample_weight=sample_w[:val_cut],
            eval_set=[(X_tr.iloc[val_cut:], y_tr[val_cut:])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        print(f"   Fold {fold+1} train [{date_tr_start} → {df_tr[DATE_COL].iloc[-1].date()}] "
              f"({len(df_tr)} lignes) → test [{date_min} → {date_max}]")

        y_pred = model.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        bal  = balanced_accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        r2   = f1_score(y_te, y_pred, labels=[2], average="macro", zero_division=0)  # recall class 2
        r0   = f1_score(y_te, y_pred, labels=[0], average="macro", zero_division=0)  # recall class 0

        print(f"          BalAcc={bal:.3f} F1m={f1m:.3f} R0={r0:.3f} R2={r2:.3f}")

        fold_results.append({
            "fold": fold + 1,
            "test_start": str(date_min),
            "test_end": str(date_max),
            "n_train": len(df_tr),
            "n_test": len(df_te),
            "accuracy": acc,
            "balanced_accuracy": bal,
            "f1_macro": f1m,
            "f1_class0": r0,
            "f1_class2": r2,
        })

    wf_df = pd.DataFrame(fold_results)
    wf_df.to_csv(os.path.join(results_dir, "walk_forward_metrics.csv"), index=False)

    print(f"\n   ── Résumé walk-forward ──")
    print(f"   BalAcc  : {wf_df['balanced_accuracy'].mean():.3f} ± {wf_df['balanced_accuracy'].std():.3f}")
    print(f"   F1-Macro: {wf_df['f1_macro'].mean():.3f} ± {wf_df['f1_macro'].std():.3f}")
    print(f"   F1-C0   : {wf_df['f1_class0'].mean():.3f} | F1-C2: {wf_df['f1_class2'].mean():.3f}")

    # Graphique stabilité par fold
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(wf_df["fold"], wf_df["balanced_accuracy"], color="steelblue")
    axes[0].axhline(wf_df["balanced_accuracy"].mean(), color="red", linestyle="--", label="moyenne")
    axes[0].set_title("Balanced Accuracy par fold")
    axes[0].set_xlabel("Fold")
    axes[0].legend()
    axes[1].bar(wf_df["fold"], wf_df["f1_macro"], color="darkorange")
    axes[1].axhline(wf_df["f1_macro"].mean(), color="red", linestyle="--", label="moyenne")
    axes[1].set_title("F1-Macro par fold")
    axes[1].set_xlabel("Fold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "walk_forward_stability.png"), dpi=150)
    plt.close()

    return wf_df


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(feature_selection_on=True, tune=False, walk_forward=False,
         lag_features=False, time_weighting=False,
         run_xgboost=False):
    results_dir = create_results_dir()
    print(f"🔹 Dossier résultats : {results_dir}")

    run_params = {
        "feature_selection_on": feature_selection_on,
        "tune": tune,
        "walk_forward": walk_forward,
        "lag_features": lag_features,
        "time_weighting": time_weighting,
        "run_xgboost": run_xgboost,
    }
    save_run_config(results_dir, run_params)

    # 1. Chargement données
    print(f"🔹 Chargement : {DATA_FILE_PATH}")
    df_raw = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    cols_to_drop = [
        "sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high",
    ]
    df_raw.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    print(f"   → {len(df_raw)} lignes, {len(df_raw.columns)} colonnes")

    # 2. Split temporel
    df_main    = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    df_holdout = df_raw[
        (df_raw[DATE_COL] >= HOLDOUT_START_DATE) &
        (df_raw[DATE_COL] <= HOLDOUT_END_DATE)
    ].copy()

    df_main    = create_target(df_main,    tag="(train)")
    df_holdout = create_target(df_holdout, tag="(holdout)")
    print(f"   → Train : {len(df_main)} | Holdout : {len(df_holdout)}")

    # 3. Axe C : lag features ajoutées AVANT SHAP pour que SHAP puisse les évaluer
    # (uniquement si lag_features=True)
    if lag_features:
        print("🔹 Ajout des lag features sur toutes les colonnes (avant sélection SHAP)...")
        all_raw_feats = [f for f in df_main.columns
                         if f not in (DATE_COL, "target", "ret_future")]
        df_full = pd.concat([df_main, df_holdout], axis=0).sort_values(DATE_COL)
        df_full, _ = add_lag_features(df_full, all_raw_feats)
        df_main    = df_full[df_full[DATE_COL] < HOLDOUT_START_DATE].copy()
        df_holdout = df_full[
            (df_full[DATE_COL] >= HOLDOUT_START_DATE) &
            (df_full[DATE_COL] <= HOLDOUT_END_DATE)
        ].copy()
        print(f"   → {len(df_main)} lignes train, {len(df_holdout)} lignes holdout après lags")

    # 4. Feature selection (SHAP sur le pool complet — inclut les lags si activés)
    if feature_selection_on:
        print("🔹 Sélection SHAP features...")
        raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
        features = [f for f in raw_feats if f not in (DATE_COL, "target", "ret_future")]
        if TARGET_PRICE_COL not in features:
            features.append(TARGET_PRICE_COL)
    else:
        features = [f for f in df_main.columns
                    if f not in (DATE_COL, "target", "ret_future")]
    print(f"   → {len(features)} features retenues")

    # 5. Split train / val (80/20 chronologique sur df_main)
    split_idx = int(0.8 * len(df_main))
    train_df = df_main.iloc[:split_idx]
    val_df   = df_main.iloc[split_idx:]
    test_df  = df_holdout

    X_train = train_df[features]
    y_train = train_df["target"].values
    X_val   = val_df[features]
    y_val   = val_df["target"].values
    X_test  = test_df[features]
    y_test  = test_df["target"].values
    date_test = test_df[DATE_COL].values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_val   = pd.DataFrame(scaler.transform(X_val),       columns=features)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=features)

    print(f"🔹 Shapes — Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape}")

    # 5. Class weights + pondération temporelle exponentielle (axe 3)
    present_classes = np.unique(y_train)
    cw_raw = compute_class_weight("balanced", classes=present_classes, y=y_train)
    cw_dict = {c: float(w * CLASS_WEIGHT_BOOST.get(c, 1.0))
               for c, w in zip(present_classes, cw_raw)}
    print(f"🔹 Class weights : {cw_dict}")

    # Pondération temporelle exponentielle (axe 3, optionnel)
    if time_weighting:
        halflife = 252
        n_tr = len(y_train)
        decay = np.exp(np.log(0.5) / halflife * np.arange(n_tr - 1, -1, -1))
        time_w = decay / decay.sum() * n_tr
        sample_weight_train = np.array([cw_dict.get(c, 1.0) for c in y_train]) * time_w
        print(f"   → Pondération temporelle activée (halflife={halflife}j)")
    else:
        sample_weight_train = np.array([cw_dict.get(c, 1.0) for c in y_train])

    # 5b. Walk-forward validation
    if walk_forward:
        print("\n🔹 Walk-forward validation...")
        run_walk_forward(df_main, features, results_dir,
                         time_weighting=time_weighting)

    # 6. LightGBM
    print("\n🔹 Entraînement LightGBM...")
    if tune:
        print("   → Optuna tuning activé (N_TRIALS trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(
            lambda trial: _lgb_objective(trial, X_train, y_train),
            n_trials=N_TRIALS,
        )
        best_params = study.best_params
        best_params.update({"class_weight": cw_dict, "random_state": SEED, "verbosity": -1})
        print(f"   → Meilleurs params : {best_params}")
        save_run_config(results_dir, run_params, best_lgb_params=study.best_params)
        lgb_model = lgb.LGBMClassifier(**best_params)
    else:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=cw_dict,
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )

    lgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)],
    )

    lgb_proba = pad_proba(lgb_model.predict_proba(X_test), lgb_model.classes_)
    lgb_pred  = lgb_model.predict(X_test)
    save_feature_importance(lgb_model, features, results_dir, "lightgbm")

    # Calibration isotonic + seuils asymétriques (sur X_val comme ensemble de calibration)
    thresholds = DECISION_THRESHOLDS if hasattr(DECISION_THRESHOLDS, "__getitem__") else None
    print("🔹 Calibration isotonic des probabilités...")
    lgb_proba_cal, lgb_pred_cal = calibrate_and_threshold(
        lgb_model, X_val, y_val, X_test, thresholds=thresholds
    )

    lgb_f1 = evaluate_and_save("lightgbm", y_test, lgb_pred, lgb_proba,
                                date_test, results_dir, test_df, XGB_VERSION,
                                y_pred_cal=lgb_pred_cal, proba_cal=lgb_proba_cal)

    # 7. XGBoost (optionnel — désactivé par défaut, run_xgboost=False)
    xgb_f1 = None
    if run_xgboost:
        print("\n🔹 Entraînement XGBoost (référence)...")
        sample_weight      = sample_weight_train
        eval_sample_weight = np.array([cw_dict.get(c, 1.0) for c in y_val])

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        dval   = xgb.DMatrix(X_val,   label=y_val,   weight=eval_sample_weight)
        dtest  = xgb.DMatrix(X_test)

        xgb_params = {
            "objective":        "multi:softprob",
            "num_class":        N_CLASSES,
            "eval_metric":      "mlogloss",
            "learning_rate":    0.05,
            "max_depth":        6,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "seed":             SEED,
            "verbosity":        0,
        }
        xgb_booster = xgb.train(
            xgb_params, dtrain, num_boost_round=1000,
            evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=100,
        )
        xgb_proba = xgb_booster.predict(dtest).reshape(-1, N_CLASSES)
        xgb_pred  = xgb_proba.argmax(axis=1)

        scores_imp = xgb_booster.get_score(importance_type="gain")
        imp = pd.DataFrame([
            {"feature": f, "importance": scores_imp.get(f, 0.0)} for f in features
        ]).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(results_dir, "xgboost_feature_importance.csv"), index=False)
        plt.figure(figsize=(10, max(6, len(imp) * 0.3)))
        plt.barh(imp["feature"][:30][::-1], imp["importance"][:30][::-1])
        plt.xlabel("Importance (gain)")
        plt.title("Top features — xgboost")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "xgboost_feature_importance.png"), dpi=150)
        plt.close()

        xgb_f1 = evaluate_and_save("xgboost", y_test, xgb_pred, xgb_proba,
                                    date_test, results_dir, test_df, XGB_VERSION)
    else:
        print("\n🔹 XGBoost désactivé (run_xgboost=False)")

    # 8. Résumé
    print("\n===== RÉSUMÉ =====")
    print(f"   LightGBM F1-Macro : {lgb_f1:.4f}")
    if xgb_f1 is not None:
        print(f"   XGBoost  F1-Macro : {xgb_f1:.4f}")
    best_f1 = max(lgb_f1, xgb_f1) if xgb_f1 is not None else lgb_f1
    if best_f1 > 0.40:
        print("   ✅ Signal confirmé — RL ou Transformer envisageable")
    elif best_f1 > 0.30:
        print("   ⚠️  Signal faible — continuer l'amélioration des features")
    else:
        print("   ❌ Signal absent — revoir la collecte de données")

    print(f"\n🔹 Résultats sauvegardés : {results_dir}")


if __name__ == "__main__":
    main(feature_selection_on=True, tune=True, walk_forward=True,
         lag_features=True, time_weighting=True, run_xgboost=False)