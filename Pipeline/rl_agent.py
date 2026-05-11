"""
rl_agent.py — Agent RL (PPO) entraîné sur signal LightGBM pour trading S&P500.

Workflow :
  1. generate_oof_predictions() → probas LightGBM out-of-fold sur le train (2006–2022)
  2. TradingEnv (gymnasium) → observation = [proba_0/1/2, position, ret_1/5/21j, vol_21j]
  3. train_rl_agent() → PPO stable-baselines3, entraîné sur les probas OOF
  4. evaluate_rl_agent() → backtest sur le holdout LightGBM (2023–2025)

Usage :
    python -m pipeline.rl_agent
    python -m pipeline.rl_agent --timesteps 300000 --folds 5
"""

import os
import sys
import io
import argparse

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

import optuna
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from pipeline.config import (
    DATA_FILE_PATH, DATE_COL, TARGET_PRICE_COL,
    PRED_HORIZON, N_CLASSES, TOP_N_FEATURES,
    FIXED_THRESHOLDS, THRESHOLD_STRATEGY,
    CLASS_WEIGHT_BOOST, ECART_MIN, HOLDOUT_START_DATE,
)
from pipeline.feature_selection import select_top_features_shap

SEED = 42
np.random.seed(SEED)

# Hyperparamètres LightGBM du meilleur run (xgb_baseline_v1_20260511_154608)
BEST_LGB_PARAMS = {
    "n_estimators":      1003,
    "learning_rate":     0.134,
    "max_depth":         9,
    "num_leaves":        36,
    "min_child_samples": 16,
    "subsample":         0.753,
    "colsample_bytree":  0.510,
    "reg_alpha":         1.438,
    "reg_lambda":        0.000251,
    "random_state":      SEED,
    "deterministic":     True,
    "force_col_wise":    True,
    "n_jobs":            1,
    "verbosity":         -1,
}


# ──────────────────────────────────────────────────────────────────────
# 1. Génération des probabilités OOF (out-of-fold) sur le train
# ──────────────────────────────────────────────────────────────────────

def _create_target(df):
    df = df.copy()
    df["ret_future"] = (
        df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]
    ) / df[TARGET_PRICE_COL]
    df.dropna(subset=["ret_future"], inplace=True)
    lo, hi = FIXED_THRESHOLDS
    df["target"] = df["ret_future"].apply(
        lambda x: 0 if x < lo else (2 if x > hi else 1)
    )
    return df


def generate_oof_predictions(data_csv: str, output_dir: str, n_folds: int = 5) -> str:
    """
    Génère des probabilités LightGBM out-of-fold sur la période train (< HOLDOUT_START_DATE).
    Sauvegarde oof_predictions.csv dans output_dir.
    Retourne le chemin du fichier généré.
    """
    print(f"\n🔹 Génération des probabilités OOF ({n_folds} folds)...")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "oof_predictions.csv")

    df_raw = pd.read_csv(data_csv, parse_dates=[DATE_COL])
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    df_train = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    df_train = _create_target(df_train)
    print(f"   → {len(df_train)} lignes train disponibles")

    # Sélection SHAP sur tout le train (une seule fois)
    print("   → Sélection SHAP features...")
    raw_feats = select_top_features_shap(df_train, top_n=TOP_N_FEATURES, target_col="ret_future")
    features = [f for f in raw_feats if f not in (DATE_COL, "target", "ret_future")]
    if TARGET_PRICE_COL not in features:
        features.append(TARGET_PRICE_COL)
    print(f"   → {len(features)} features retenues")

    tscv = TimeSeriesSplit(n_splits=n_folds)
    X_all = df_train[features]
    y_all = df_train["target"].values
    dates_all = df_train[DATE_COL].values

    oof_rows = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr = X_all.iloc[train_idx]
        y_tr = y_all[train_idx]
        X_vl = X_all.iloc[val_idx]
        y_vl = y_all[val_idx]
        dates_vl = dates_all[val_idx]

        if len(X_tr) < 100 or len(X_vl) < 20:
            continue

        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=features)
        X_vl_sc = pd.DataFrame(scaler.transform(X_vl), columns=features)

        present = np.unique(y_tr)
        cw_raw = compute_class_weight("balanced", classes=present, y=y_tr)
        cw = {c: float(w * CLASS_WEIGHT_BOOST.get(c, 1.0)) for c, w in zip(present, cw_raw)}

        # Split interne 80/20 pour early stopping
        cut = int(0.8 * len(X_tr_sc))
        model = lgb.LGBMClassifier(**BEST_LGB_PARAMS, class_weight=cw)
        model.fit(
            X_tr_sc.iloc[:cut], y_tr[:cut],
            eval_set=[(X_tr_sc.iloc[cut:], y_tr[cut:])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        # Calibration isotonic sur la partie val interne du fold
        cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal_model.fit(X_tr_sc.iloc[cut:], y_tr[cut:])

        proba = cal_model.predict_proba(X_vl_sc)
        # Pad si modèle n'a vu que 2 classes
        if proba.shape[1] < N_CLASSES:
            full = np.zeros((proba.shape[0], N_CLASSES))
            for ci, cls in enumerate(cal_model.classes_):
                full[:, int(cls)] = proba[:, ci]
            proba = full

        y_pred_vl = proba.argmax(axis=1)
        f1m = f1_score(y_vl, y_pred_vl, average="macro", zero_division=0)
        print(f"   Fold {fold+1}/{n_folds} : {len(X_tr)} train → {len(X_vl)} val  |  F1={f1m:.3f}")

        sorted_p = np.sort(proba, axis=1)[:, ::-1]
        for i, idx in enumerate(val_idx):
            oof_rows.append({
                "date_prediction": pd.Timestamp(dates_vl[i]),
                "y_true":          int(y_vl[i]),
                "y_pred":          int(y_pred_vl[i]),
                "proba_0":         proba[i, 0],
                "proba_1":         proba[i, 1],
                "proba_2":         proba[i, 2],
                "top_proba":       sorted_p[i, 0],
                "second_proba":    sorted_p[i, 1],
                "confidence_gap":  sorted_p[i, 0] - sorted_p[i, 1],
            })

    oof_df = pd.DataFrame(oof_rows).sort_values("date_prediction").reset_index(drop=True)
    oof_df.to_csv(out_path, index=False)
    print(f"   → {len(oof_df)} lignes OOF sauvegardées : {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# 2. Chargement des données
# ──────────────────────────────────────────────────────────────────────

def load_predictions(predictions_csv: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv, parse_dates=["date_prediction"])
    return df.sort_values("date_prediction").reset_index(drop=True)


def load_price_series(data_csv: str) -> pd.Series:
    df = pd.read_csv(data_csv, parse_dates=[DATE_COL])
    return df.set_index(DATE_COL)[TARGET_PRICE_COL].sort_index()


# ──────────────────────────────────────────────────────────────────────
# 3. Environnement Gymnasium
# ──────────────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Environnement Gymnasium pour trading S&P500 guidé par signal LightGBM.

    Pas de temps = PRED_HORIZON jours : l'agent décide UNE FOIS par fenêtre de trade.
    Cela élimine structurellement le double-comptage des positions chevauchantes.

    Observation (8 dimensions) :
        [proba_0, proba_1, proba_2,   ← probas LightGBM calibrées
         position_norm,               ← position courante normalisée (-1/0/1)
         ret_1j, ret_5j, ret_21j,     ← rendements passés SP500
         vol_21j]                     ← volatilité rolling 21j

    Action space : Discrete(3) → {0: short, 1: flat, 2: long}

    Reward : rendement du trade sur PRED_HORIZON jours (sans chevauchement possible).
    """

    metadata = {"render_modes": []}

    def __init__(self, predictions_df: pd.DataFrame, price_series: pd.Series,
                 pred_horizon: int = PRED_HORIZON,
                 transaction_cost: float = 0.0,
                 confidence_bonus: float = 0.3,
                 confidence_threshold: float = 0.05):
        super().__init__()
        # Sous-échantillonne les prédictions toutes les PRED_HORIZON lignes
        # pour qu'un pas de temps = une fenêtre de trade non-chevauchante.
        raw = predictions_df.reset_index(drop=True)
        self.preds                = raw.iloc[::pred_horizon].reset_index(drop=True)
        self.prices               = price_series
        self.horizon              = pred_horizon
        self.tc                   = transaction_cost
        self.confidence_bonus     = confidence_bonus
        self.confidence_threshold = confidence_threshold

        # Pré-calcul des rendements et volatilité SP500 pour chaque date de prédiction
        self._precompute_market_features()

        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 8, dtype=np.float32),
            high=np.array([1.0] * 8, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._step_idx    = 0
        self._position    = 1  # neutre au départ
        self._last_action = 1

    def _precompute_market_features(self):
        """Calcule ret_1j, ret_5j, ret_21j, vol_21j pour chaque date de prédiction."""
        dates = pd.to_datetime(self.preds["date_prediction"])
        ret_1, ret_5, ret_21, vol_21 = [], [], [], []

        for d in dates:
            p_now = self.prices.asof(d)
            def _past_ret(n):
                past_d = d - pd.offsets.BusinessDay(n)
                p_past = self.prices.asof(past_d)
                if pd.isna(p_now) or pd.isna(p_past) or p_past == 0:
                    return 0.0
                return float((p_now - p_past) / p_past)

            r1  = _past_ret(1)
            r5  = _past_ret(5)
            r21 = _past_ret(21)

            # Volatilité rolling 21j via rendements journaliers
            end_d   = d
            start_d = d - pd.offsets.BusinessDay(21)
            slice_  = self.prices.loc[start_d:end_d]
            daily_r = slice_.pct_change().dropna()
            v21 = float(daily_r.std()) if len(daily_r) >= 5 else 0.02

            ret_1.append(r1)
            ret_5.append(r5)
            ret_21.append(r21)
            vol_21.append(v21)

        self.preds = self.preds.copy()
        self.preds["_ret1"]  = ret_1
        self.preds["_ret5"]  = ret_5
        self.preds["_ret21"] = ret_21
        self.preds["_vol21"] = vol_21

    def _get_obs(self) -> np.ndarray:
        row = self.preds.iloc[self._step_idx]
        pos_norm = (self._position - 1.0) / 1.0  # -1→-1, 1→0, 2→1 mapped to [-1,0,1]
        # Normalise les rendements (clip ±10%)
        r1  = np.clip(row["_ret1"],  -0.10, 0.10) / 0.10
        r5  = np.clip(row["_ret5"],  -0.10, 0.10) / 0.10
        r21 = np.clip(row["_ret21"], -0.20, 0.20) / 0.20
        vol = np.clip(row["_vol21"], 0.0,   0.05) / 0.05
        return np.array([
            float(row["proba_0"]),
            float(row["proba_1"]),
            float(row["proba_2"]),
            float(pos_norm),
            r1, r5, r21, vol,
        ], dtype=np.float32)

    def _compute_trade_return(self, action: int) -> float:
        """Calcule le rendement du trade sur PRED_HORIZON jours."""
        if action == 1:
            return 0.0
        row = self.preds.iloc[self._step_idx]
        entry_date = pd.Timestamp(row["date_prediction"])
        exit_date  = entry_date + pd.offsets.BusinessDay(self.horizon)
        entry_p = self.prices.asof(entry_date)
        exit_p  = self.prices.asof(exit_date)
        if pd.isna(entry_p) or pd.isna(exit_p) or entry_p == 0:
            return 0.0
        raw_ret = float((exit_p - entry_p) / entry_p)
        return raw_ret if action == 2 else -raw_ret

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx    = 0
        self._position    = 1
        self._last_action = 1
        return self._get_obs(), {}

    def step(self, action: int):
        trade_ret = self._compute_trade_return(action)
        row = self.preds.iloc[self._step_idx]

        # Reward = rendement brut du trade (un pas = une fenêtre PRED_HORIZON, pas de chevauchement)
        reward = trade_ret

        # Bonus si l'agent trade et que le signal est confiant
        if action != 1:
            gap = float(row.get("confidence_gap", 0.0))
            if gap >= self.confidence_threshold:
                reward += self.confidence_bonus * gap

        # Pénalité de changement de position (optionnelle, 0 par défaut)
        if self.tc > 0 and action != self._last_action:
            reward -= self.tc

        self._position    = action
        self._last_action = action
        self._step_idx   += 1

        terminated = self._step_idx >= len(self.preds)
        truncated  = False
        obs = self._get_obs() if not terminated else np.zeros(8, dtype=np.float32)
        info = {"trade_return": trade_ret, "date": str(row["date_prediction"])}
        return obs, float(reward), terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# 4. Entraînement PPO
# ──────────────────────────────────────────────────────────────────────

def train_rl_agent(env: TradingEnv, total_timesteps: int = 200_000,
                   output_dir: str = "results/rl_ppo/") -> PPO:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🔹 Entraînement PPO ({total_timesteps:,} timesteps)...")

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        seed=SEED,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    save_path = os.path.join(output_dir, "ppo_trading")
    model.save(save_path)
    print(f"   → Modèle sauvegardé : {save_path}.zip")
    return model


# ──────────────────────────────────────────────────────────────────────
# 5. Évaluation et backtest
# ──────────────────────────────────────────────────────────────────────

def evaluate_rl_agent(model: PPO, env: TradingEnv, output_dir: str) -> pd.DataFrame:
    """
    Rejoue l'agent de façon déterministe et calcule les métriques de backtest.

    Correction chevauchement : une seule position ouverte à la fois.
    Quand un trade est ouvert (action != 1), le prochain trade ne peut démarrer
    qu'après PRED_HORIZON jours — les signaux intermédiaires sont ignorés.
    Le P&L de chaque trade est enregistré à la date d'ENTRÉE (pas d'exit).
    """
    print("\n🔹 Évaluation de l'agent RL...")
    os.makedirs(output_dir, exist_ok=True)

    # Collecte de toutes les décisions de l'agent (sans filtrage)
    obs, _ = env.reset()
    raw_records = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        raw_records.append({
            "date_prediction": info["date"],
            "action_rl":       int(action),
            "trade_return":    info["trade_return"],
        })
        if terminated or truncated:
            break

    raw_df = pd.DataFrame(raw_records)
    raw_df["date_prediction"] = pd.to_datetime(raw_df["date_prediction"])

    # Fusion avec les prédictions LightGBM (env.preds est déjà sous-échantillonné)
    preds = env.preds[["date_prediction", "y_true", "y_pred"]].copy()
    raw_df = raw_df.merge(preds, on="date_prediction", how="left")
    raw_df.rename(columns={"y_pred": "y_pred_lgbm"}, inplace=True)

    # Chaque ligne = une fenêtre de PRED_HORIZON jours non-chevauchante.
    # Pas de cooldown nécessaire : le sous-échantillonnage dans TradingEnv garantit
    # qu'il n'y a qu'un seul trade actif à la fois.
    raw_df["action_executed"]       = raw_df["action_rl"]
    raw_df["trade_return_executed"] = raw_df["trade_return"]

    # Rendement cumulé — chaque trade représente PRED_HORIZON jours de capital investi
    cumulative = 1.0
    cum_curve = []
    for r in raw_df["trade_return_executed"]:
        cumulative *= (1.0 + r)
        cum_curve.append(cumulative)
    raw_df["cumulative_return"] = cum_curve

    res_df = raw_df.copy()

    n_trades  = int((res_df["action_executed"] != 1).sum())
    total_ret = float(res_df["cumulative_return"].iloc[-1] - 1.0)
    active    = res_df[res_df["action_executed"] != 1]["trade_return_executed"]
    win_rate  = float((active > 0).mean()) if len(active) else 0.0
    # Sharpe annualisé : chaque pas = PRED_HORIZON jours de trading
    periods_per_year = 252 / PRED_HORIZON
    r_arr   = res_df["trade_return_executed"].values
    sharpe  = float(r_arr.mean() / r_arr.std() * np.sqrt(periods_per_year)) if r_arr.std() > 0 else 0.0
    rolling_mx = res_df["cumulative_return"].cummax()
    max_dd     = float(((res_df["cumulative_return"] - rolling_mx) / rolling_mx).min())

    # Buy-and-hold SP500 sur la même période (prix aux mêmes dates d'entrée de trade)
    dates = pd.to_datetime(res_df["date_prediction"])
    sp500 = env.prices
    first_p = sp500.asof(dates.iloc[0])
    last_exit = dates.iloc[-1] + pd.offsets.BusinessDay(PRED_HORIZON)
    last_p = sp500.asof(last_exit)
    bh_ret = float((last_p - first_p) / first_p) if not pd.isna(last_p) and first_p > 0 else 0.0
    # Courbe B&H aux mêmes points que le bot pour le graphique
    bh_curve = pd.Series(
        [sp500.asof(d) / first_p if not pd.isna(sp500.asof(d)) else np.nan for d in dates],
        index=dates,
    )
    bh_r      = bh_curve.pct_change().fillna(0.0)
    bh_sharpe = float(bh_r.mean() / bh_r.std() * np.sqrt(periods_per_year)) if bh_r.std() > 0 else 0.0

    print("\n===== BACKTEST RL — PPO =====")
    print(f"   Nombre de trades (hors flat) : {n_trades}")
    print(f"   Rendement bot                : {total_ret*100:.2f}%")
    print(f"   Rendement buy-and-hold       : {bh_ret*100:.2f}%")
    print(f"   Win rate                     : {win_rate*100:.1f}%")
    print(f"   Sharpe bot                   : {sharpe:.2f}")
    print(f"   Sharpe buy-and-hold          : {bh_sharpe:.2f}")
    print(f"   Max Drawdown                 : {max_dd*100:.2f}%")

    # Colonnes à sauvegarder : on garde action_rl (décision agent) + action_executed (réel)
    out_cols = ["date_prediction", "action_rl", "action_executed",
                "trade_return_executed", "cumulative_return", "y_true", "y_pred_lgbm"]
    res_df[out_cols].to_csv(os.path.join(output_dir, "backtest_results.csv"), index=False)
    pd.DataFrame([{
        "n_trades":     n_trades,
        "total_return": total_ret,
        "bh_return":    bh_ret,
        "win_rate":     win_rate,
        "sharpe_ratio": sharpe,
        "bh_sharpe":    bh_sharpe,
        "max_drawdown": max_dd,
    }]).to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)

    # ── Graphique 1 : signaux exécutés long/short sur le cours SP500 ──
    fig, ax = plt.subplots(figsize=(14, 5))
    # Courbe SP500 complète sur la période du backtest
    period_sp500 = sp500.loc[dates.iloc[0]:dates.iloc[-1] + pd.offsets.BusinessDay(PRED_HORIZON)]
    ax.plot(period_sp500.index, period_sp500.values,
            color="steelblue", linewidth=1, label="SP500", zorder=1)

    longs  = res_df[res_df["action_executed"] == 2]
    shorts = res_df[res_df["action_executed"] == 0]
    long_dates   = pd.to_datetime(longs["date_prediction"])
    short_dates  = pd.to_datetime(shorts["date_prediction"])
    long_prices  = [sp500.asof(d) for d in long_dates]
    short_prices = [sp500.asof(d) for d in short_dates]

    ax.scatter(long_dates,  long_prices,  marker="^", color="limegreen",
               s=80, zorder=3, label=f"Long ({len(longs)})")
    ax.scatter(short_dates, short_prices, marker="v", color="tomato",
               s=80, zorder=3, label=f"Short ({len(shorts)})")

    ax.set_title("Signaux RL executés — Long / Short sur SP500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix SP500")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signals_on_sp500.png"), dpi=150)
    plt.close()

    # ── Graphique 2 : bot vs buy-and-hold (rendement cumulé) ─────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, res_df["cumulative_return"].values,
            color="darkorange", linewidth=1.5, label=f"Agent RL  {total_ret*100:+.1f}%")
    ax.plot(dates, bh_curve.values,
            color="steelblue", linewidth=1.5, linestyle="--",
            label=f"Buy & Hold  {bh_ret*100:+.1f}%")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Agent RL vs Buy-and-Hold SP500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement cumulé")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"), dpi=150)
    plt.close()

    return res_df


# ──────────────────────────────────────────────────────────────────────
# 6. Optuna — tuning hyperparamètres PPO
# ──────────────────────────────────────────────────────────────────────

def _evaluate_sharpe(model: PPO, env: TradingEnv) -> tuple[float, int]:
    """Évalue un modèle PPO et retourne (sharpe, n_trades).
    Le sous-échantillonnage dans TradingEnv garantit qu'il n'y a pas de chevauchement."""
    obs, _ = env.reset()
    returns, n_trades = [], 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        ret = info["trade_return"]
        returns.append(ret)
        if int(action) != 1:
            n_trades += 1
        if terminated or truncated:
            break
    r = np.array(returns)
    periods_per_year = 252 / PRED_HORIZON
    sharpe = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else 0.0
    return sharpe, n_trades


def tune_rl_agent(oof_df: pd.DataFrame, holdout_df: pd.DataFrame,
                  price_series: pd.Series, output_dir: str,
                  n_trials: int = 30, timesteps_per_trial: int = 100_000,
                  final_timesteps: int = 300_000) -> dict:
    """
    Optimise les hyperparamètres PPO + paramètres de l'environnement via Optuna.
    Entraîne le meilleur modèle sur final_timesteps et sauvegarde les résultats.
    """
    os.makedirs(output_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparamètres PPO
        lr          = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        gamma       = trial.suggest_float("gamma", 0.90, 0.999)
        ent_coef    = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
        n_steps     = trial.suggest_categorical("n_steps", [256, 512, 1024])
        gae_lambda  = trial.suggest_float("gae_lambda", 0.90, 0.99)
        # Paramètres de l'environnement
        conf_bonus  = trial.suggest_float("confidence_bonus", 0.1, 0.8)

        env = TradingEnv(oof_df, price_series, confidence_bonus=conf_bonus)
        model = PPO(
            "MlpPolicy", env,
            verbose=0, seed=SEED,
            learning_rate=lr, n_steps=n_steps,
            batch_size=min(64, n_steps),
            n_epochs=10, gamma=gamma,
            gae_lambda=gae_lambda, ent_coef=ent_coef,
            clip_range=0.2,
        )
        model.learn(total_timesteps=timesteps_per_trial, progress_bar=False)

        eval_env = TradingEnv(holdout_df, price_series, confidence_bonus=conf_bonus)
        sharpe, n_trades = _evaluate_sharpe(model, eval_env)

        # Pénalise les agents trop passifs (< 50 trades sur 565 jours)
        if n_trades < 50:
            return -10.0
        return sharpe

    print(f"\n🔹 Optuna PPO — {n_trials} trials × {timesteps_per_trial:,} steps...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"\n   → Meilleurs params Optuna : {best}")
    print(f"   → Sharpe trial : {study.best_value:.3f}")

    # Sauvegarde des meilleurs paramètres
    import json
    params_path = os.path.join(output_dir, "optuna_best_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": best, "best_sharpe_trial": study.best_value}, f, indent=2)
    print(f"   → Params sauvegardés : {params_path}")

    # Réentraînement final avec les meilleurs hyperparamètres
    print(f"\n🔹 Réentraînement final ({final_timesteps:,} steps)...")
    conf_bonus = best.pop("confidence_bonus", 0.3)
    final_env = TradingEnv(oof_df, price_series, confidence_bonus=conf_bonus)
    final_model = PPO(
        "MlpPolicy", final_env,
        verbose=1, seed=SEED,
        batch_size=min(64, best.get("n_steps", 512)),
        n_epochs=10, clip_range=0.2,
        **best,
    )
    final_model.learn(total_timesteps=final_timesteps, progress_bar=False)
    final_model.save(os.path.join(output_dir, "ppo_trading_tuned"))

    # Évaluation finale sur le holdout
    eval_env = TradingEnv(holdout_df, price_series, confidence_bonus=conf_bonus)
    evaluate_rl_agent(final_model, eval_env, output_dir)

    return best


# ──────────────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────────────

def main(predictions_csv: str, data_csv: str,
         output_dir: str = "results/rl_ppo/",
         total_timesteps: int = 200_000,
         n_folds: int = 5,
         skip_oof: bool = False,
         tune: bool = False,
         n_trials: int = 30):
    """
    Pipeline complet RL :
      1. Génère les probas OOF sur le train LightGBM (sauf si skip_oof=True)
      2. Entraîne PPO sur OOF (ou lance Optuna si tune=True)
      3. Évalue PPO sur le holdout LightGBM

    Args:
        predictions_csv : probas holdout LightGBM (pour évaluation)
        data_csv        : données brutes (pour OOF + prix)
        output_dir      : dossier de sauvegarde
        total_timesteps : durée entraînement PPO
        n_folds         : nombre de folds TimeSeriesSplit pour OOF
        skip_oof        : réutilise oof_predictions.csv existant
        tune            : active le tuning Optuna (n_trials trials)
        n_trials        : nombre de trials Optuna
    """
    os.makedirs(output_dir, exist_ok=True)
    price_series = load_price_series(data_csv)

    # ── Étape 1 : probas OOF pour entraînement RL ──────────────────
    oof_csv = os.path.join(output_dir, "oof_predictions.csv")
    if skip_oof and os.path.exists(oof_csv):
        print(f"🔹 OOF existant chargé : {oof_csv}")
    else:
        oof_csv = generate_oof_predictions(data_csv, output_dir, n_folds=n_folds)

    oof_df = load_predictions(oof_csv)
    holdout_df = load_predictions(predictions_csv)
    print(f"   → {len(oof_df)} lignes OOF pour entraînement RL")
    print(f"   → {len(holdout_df)} lignes holdout pour évaluation RL")

    if tune:
        # ── Tuning Optuna ───────────────────────────────────────────
        tune_rl_agent(
            oof_df=oof_df,
            holdout_df=holdout_df,
            price_series=price_series,
            output_dir=output_dir,
            n_trials=n_trials,
            timesteps_per_trial=100_000,
            final_timesteps=total_timesteps,
        )
    else:
        # ── Entraînement PPO standard ───────────────────────────────
        train_env = TradingEnv(oof_df, price_series)
        print("🔹 Vérification de l'environnement...")
        check_env(train_env, warn=True)
        model = train_rl_agent(train_env, total_timesteps=total_timesteps, output_dir=output_dir)

        eval_env = TradingEnv(holdout_df, price_series)
        evaluate_rl_agent(model, eval_env, output_dir)

    print(f"\n🔹 Résultats RL sauvegardés : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent RL PPO sur signal LightGBM")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Timesteps d'entraînement PPO (ou réentraînement final si --tune)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Nombre de folds TimeSeriesSplit pour génération OOF")
    parser.add_argument("--skip-oof", action="store_true",
                        help="Réutilise oof_predictions.csv existant (skip génération)")
    parser.add_argument("--tune", action="store_true",
                        help="Lance Optuna pour tuner les hyperparamètres PPO")
    parser.add_argument("--trials", type=int, default=30,
                        help="Nombre de trials Optuna (avec --tune)")
    parser.add_argument("--output-dir", type=str, default="results/rl_ppo/",
                        help="Dossier de sauvegarde des résultats")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BEST_RUN_PREDS = os.path.join(
        project_root,
        "results", "xgb_baseline_v1_20260511_154608",
        "lightgbm", "holdout_predictions.csv"
    )
    DATA = os.path.join(project_root, DATA_FILE_PATH)
    OUTPUT = os.path.join(project_root, args.output_dir)

    main(
        predictions_csv=BEST_RUN_PREDS,
        data_csv=DATA,
        output_dir=OUTPUT,
        total_timesteps=args.timesteps,
        n_folds=args.folds,
        skip_oof=args.skip_oof,
        tune=args.tune,
        n_trials=args.trials,
    )
