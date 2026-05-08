import os
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


def create_target(df, tag=""):
    print(f"🔹 Création de la cible (horizon = {PRED_HORIZON}) {tag}")
    df = df.copy()
    df["ret_future"] = (
        df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]
    ) / df[TARGET_PRICE_COL]
    df.dropna(subset=["ret_future"], inplace=True)

    lo, hi = FIXED_THRESHOLDS
    df["target"] = df["ret_future"].apply(
        lambda x: 0 if x < lo else (2 if x > hi else 1)
    )
    print("   → Distribution classes :", df["target"].value_counts(normalize=True).to_dict())
    return df


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


def evaluate_and_save(model_name, y_true, y_pred, proba, date_index,
                      results_dir, test_df, version):
    acc  = accuracy_score(y_true, y_pred)
    bal  = balanced_accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n===== HOLDOUT — {model_name} =====")
    print(f"   Accuracy        : {acc:.4f}")
    print(f"   Balanced Acc.   : {bal:.4f}")
    print(f"   F1-Macro        : {f1m:.4f}")
    print(f"   F1-Weighted     : {f1w:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    sub = os.path.join(results_dir, model_name)
    os.makedirs(sub, exist_ok=True)

    pd.DataFrame([{
        "model_version": version,
        "ecart_min": "n/a",
        "accuracy": acc,
        "balanced_accuracy": bal,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "n_holdout_samples": len(y_true),
    }]).to_csv(os.path.join(sub, "holdout_metrics.csv"), index=False)

    classif_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(classif_dict).transpose().to_csv(
        os.path.join(sub, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix_plot(cm, os.path.join(sub, "confusion_matrix.png"))
    save_prediction_distribution_plot(y_true, y_pred,
        os.path.join(sub, "class_distribution_true_vs_pred.png"))

    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    pred_df = pd.DataFrame({
        "date_prediction": pd.to_datetime(date_index),
        "y_true": y_true,
        "y_pred": y_pred,
        "proba_0": proba[:, 0],
        "proba_1": proba[:, 1],
        "proba_2": proba[:, 2],
        "top_proba": sorted_proba[:, 0],
        "second_proba": sorted_proba[:, 1],
        "confidence_gap": sorted_proba[:, 0] - sorted_proba[:, 1],
    })
    pred_df.to_csv(os.path.join(sub, "holdout_predictions.csv"), index=False)

    price_series = test_df.set_index(DATE_COL)[TARGET_PRICE_COL]
    run_backtest(pred_df, price_series, sub)

    return f1m


# ──────────────────────────────────────────────────────────────────────
# Optuna objective — LightGBM, optimise F1-Macro sur TimeSeriesSplit
# ──────────────────────────────────────────────────────────────────────

def _lgb_objective(trial, X, y, cw_dict):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": cw_dict,
        "random_state": SEED,
        "verbosity": -1,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
        preds = m.predict(X_vl)
        scores.append(f1_score(y_vl, preds, average="macro", zero_division=0))
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(feature_selection_on=True, tune=False):
    results_dir = create_results_dir()
    print(f"🔹 Dossier résultats : {results_dir}")

    # 1. Chargement données
    print(f"🔹 Chargement : {DATA_FILE_PATH}")
    df_raw = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
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

    # 3. Feature selection
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

    # 4. Split train / val (80/20 chronologique sur df_main)
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

    # 5. Class weights
    cw_raw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_train)
    cw_dict = {c: float(cw_raw[c] * CLASS_WEIGHT_BOOST.get(c, 1.0)) for c in range(N_CLASSES)}
    print(f"🔹 Class weights : {cw_dict}")

    # 6. LightGBM
    print("\n🔹 Entraînement LightGBM...")
    if tune:
        print("   → Optuna tuning activé (N_TRIALS trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(
            lambda trial: _lgb_objective(trial, X_train, y_train, cw_dict),
            n_trials=N_TRIALS,
        )
        best_params = study.best_params
        best_params.update({"class_weight": cw_dict, "random_state": SEED, "verbosity": -1})
        print(f"   → Meilleurs params : {best_params}")
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
            verbosity=-1,
        )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)],
    )

    lgb_proba = lgb_model.predict_proba(X_test)
    lgb_pred  = lgb_model.predict(X_test)
    save_feature_importance(lgb_model, features, results_dir, "lightgbm")
    lgb_f1 = evaluate_and_save("lightgbm", y_test, lgb_pred, lgb_proba,
                                date_test, results_dir, test_df, XGB_VERSION)

    # 7. XGBoost
    print("\n🔹 Entraînement XGBoost...")
    # Remplace class_weight dict par sample_weight pour XGBoost
    sample_weight = np.array([cw_dict[c] for c in y_train])
    eval_sample_weight = np.array([cw_dict[c] for c in y_val])

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=SEED,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[eval_sample_weight],
        verbose=100,
    )

    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_pred  = xgb_model.predict(X_test)
    save_feature_importance(xgb_model, features, results_dir, "xgboost")
    xgb_f1 = evaluate_and_save("xgboost", y_test, xgb_pred, xgb_proba,
                                date_test, results_dir, test_df, XGB_VERSION)

    # 8. Résumé comparatif
    print("\n===== COMPARAISON =====")
    print(f"   LightGBM F1-Macro : {lgb_f1:.4f}")
    print(f"   XGBoost  F1-Macro : {xgb_f1:.4f}")
    print(f"   LSTM     F1-Macro : 0.2800  (référence run 22:19)")
    winner = "LightGBM" if lgb_f1 >= xgb_f1 else "XGBoost"
    print(f"   → Meilleur modèle gradient boosting : {winner}")
    if max(lgb_f1, xgb_f1) > 0.40:
        print("   ✅ Signal confirmé — le RL ou un Transformer peut être envisagé")
    elif max(lgb_f1, xgb_f1) > 0.30:
        print("   ⚠️  Signal faible — améliorer le feature engineering avant le RL")
    else:
        print("   ❌ Signal absent — revoir la collecte de données")

    print(f"\n🔹 Résultats sauvegardés : {results_dir}")


if __name__ == "__main__":
    main(feature_selection_on=True, tune=True)
