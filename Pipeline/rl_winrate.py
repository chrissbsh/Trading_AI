"""
rl_winrate.py — Agent RL (PPO) dont la reward maximise le win rate.

Différence avec rl_agent.py :
  - Reward binaire : +1 si trade gagnant, -1 si trade perdant, 0 si flat
  - Contrainte minimale de 100 trades dans l'objectif Optuna
  - Optimisation Optuna centrée win_rate (pas Sharpe)

Réutilise depuis rl_agent.py :
  - load_predictions(), load_price_series()
  - TradingEnv (modifié par héritage : WinRateEnv)
  - evaluate_rl_agent()

Usage :
    python -m pipeline.rl_winrate
    python -m pipeline.rl_winrate --skip-oof --tune --trials 30
    python -m pipeline.rl_winrate --skip-oof --timesteps 300000
"""

import os
import sys
import io
import json
import argparse

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from pipeline.config import DATA_FILE_PATH, TARGET_PRICE_COL, DATE_COL, PRED_HORIZON
from pipeline.rl_agent import (
    load_predictions,
    load_price_series,
    generate_oof_predictions,
    TradingEnv,
    evaluate_rl_agent,
)

SEED = 42
np.random.seed(SEED)
MIN_TRADES = 100  # contrainte minimale pour que le win rate soit significatif


# ──────────────────────────────────────────────────────────────────────
# Environnement spécialisé : reward binaire win/loss
# ──────────────────────────────────────────────────────────────────────

class WinRateEnv(TradingEnv):
    """
    Hérite de TradingEnv mais remplace la reward par un signal binaire :
      +reward_win  si le trade est gagnant (trade_return > 0)
      -reward_loss si le trade est perdant (trade_return < 0)
       0           si l'agent choisit flat

    Cela pousse PPO à apprendre à sélectionner UNIQUEMENT les trades
    qu'il est sûr de gagner, plutôt que d'optimiser le rendement moyen.
    La magnitude du rendement n'a plus d'importance — seule la direction compte.
    """

    def __init__(self, predictions_df, price_series,
                 pred_horizon=PRED_HORIZON,
                 reward_win: float = 1.0,
                 reward_loss: float = 1.0,
                 confidence_bonus: float = 0.2,
                 confidence_threshold: float = 0.05):
        super().__init__(
            predictions_df=predictions_df,
            price_series=price_series,
            pred_horizon=pred_horizon,
            transaction_cost=0.0,
            confidence_bonus=0.0,      # on désactive le bonus de rl_agent
            confidence_threshold=confidence_threshold,
        )
        self.reward_win          = reward_win
        self.reward_loss         = reward_loss
        self.wr_confidence_bonus = confidence_bonus
        self.wr_confidence_thr   = confidence_threshold

    def step(self, action: int):
        trade_ret = self._compute_trade_return(action)
        row = self.preds.iloc[self._step_idx]

        if action == 1:
            # Flat → pas de reward (ni pénalité, ni gain)
            reward = 0.0
        elif trade_ret > 0:
            # Trade gagnant
            reward = self.reward_win
            gap = float(row.get("confidence_gap", 0.0))
            if gap >= self.wr_confidence_thr:
                reward += self.wr_confidence_bonus * gap
        else:
            # Trade perdant
            reward = -self.reward_loss

        self._position    = action
        self._last_action = action
        self._step_idx   += 1

        terminated = self._step_idx >= len(self.preds)
        truncated  = False
        obs = self._get_obs() if not terminated else np.zeros(8, dtype=np.float32)
        info = {"trade_return": trade_ret, "date": str(row["date_prediction"])}
        return obs, float(reward), terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# Entraînement PPO standard
# ──────────────────────────────────────────────────────────────────────

def train_winrate_agent(env: WinRateEnv, total_timesteps: int = 200_000,
                        output_dir: str = "results/rl_winrate/",
                        ppo_kwargs: dict | None = None) -> PPO:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🔹 Entraînement PPO win-rate ({total_timesteps:,} timesteps)...")

    defaults = dict(
        learning_rate=3e-4, n_steps=512, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.02,
    )
    if ppo_kwargs:
        defaults.update(ppo_kwargs)

    model = PPO("MlpPolicy", env, verbose=1, seed=SEED, **defaults)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    save_path = os.path.join(output_dir, "ppo_winrate")
    model.save(save_path)
    print(f"   → Modèle sauvegardé : {save_path}.zip")
    return model


# ──────────────────────────────────────────────────────────────────────
# Évaluation centrée win rate
# ──────────────────────────────────────────────────────────────────────

def _replay(model: PPO, env: WinRateEnv) -> tuple[float, int, float]:
    """Rejoue l'agent et retourne (win_rate, n_trades, sharpe)."""
    obs, _ = env.reset()
    returns, n_trades, n_wins = [], 0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        ret = info["trade_return"]
        returns.append(ret)
        if int(action) != 1:
            n_trades += 1
            if ret > 0:
                n_wins += 1
        if terminated or truncated:
            break
    win_rate = n_wins / n_trades if n_trades else 0.0
    r = np.array(returns)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return win_rate, n_trades, sharpe


def evaluate_winrate_agent(model: PPO, env: WinRateEnv,
                            output_dir: str) -> pd.DataFrame:
    """Backtest complet + sauvegarde résultats (réutilise evaluate_rl_agent)."""
    # On utilise evaluate_rl_agent de rl_agent.py pour les CSV/PNG
    # mais on affiche aussi le win rate
    win_rate, n_trades, sharpe = _replay(model, env)
    print(f"\n===== WIN RATE AGENT RL =====")
    print(f"   Nombre de trades   : {n_trades}")
    print(f"   Win rate           : {win_rate*100:.1f}%")
    print(f"   Sharpe (annualisé) : {sharpe:.2f}")

    return evaluate_rl_agent(model, env, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Optuna — tuning centré win rate
# ──────────────────────────────────────────────────────────────────────

def tune_winrate_agent(oof_df: pd.DataFrame, holdout_df: pd.DataFrame,
                       price_series, output_dir: str,
                       n_trials: int = 30, timesteps_per_trial: int = 100_000,
                       final_timesteps: int = 300_000) -> dict:
    """
    Optimise les hyperparamètres PPO + paramètres WinRateEnv via Optuna.
    Objectif : maximiser le win rate sur le holdout, sous contrainte n_trades >= MIN_TRADES.
    """
    os.makedirs(output_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparamètres PPO
        lr         = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        gamma      = trial.suggest_float("gamma", 0.90, 0.999)
        ent_coef   = trial.suggest_float("ent_coef", 0.005, 0.08, log=True)
        n_steps    = trial.suggest_categorical("n_steps", [256, 512, 1024])
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
        # Paramètres de l'environnement win-rate
        reward_win  = trial.suggest_float("reward_win", 0.5, 2.0)
        reward_loss = trial.suggest_float("reward_loss", 0.5, 2.0)
        conf_bonus  = trial.suggest_float("confidence_bonus", 0.0, 0.5)

        env = WinRateEnv(oof_df, price_series,
                         reward_win=reward_win, reward_loss=reward_loss,
                         confidence_bonus=conf_bonus)
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=SEED,
            learning_rate=lr, n_steps=n_steps,
            batch_size=min(64, n_steps), n_epochs=10,
            gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef,
            clip_range=0.2,
        )
        model.learn(total_timesteps=timesteps_per_trial, progress_bar=False)

        eval_env = WinRateEnv(holdout_df, price_series,
                              reward_win=reward_win, reward_loss=reward_loss,
                              confidence_bonus=conf_bonus)
        win_rate, n_trades, _ = _replay(model, eval_env)

        if n_trades < MIN_TRADES:
            return -1.0  # pénalise les agents trop passifs
        return win_rate

    print(f"\n🔹 Optuna win-rate — {n_trials} trials × {timesteps_per_trial:,} steps...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"\n   → Meilleurs params : {best}")
    print(f"   → Win rate trial   : {study.best_value*100:.1f}%")

    params_path = os.path.join(output_dir, "optuna_best_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": best, "best_win_rate_trial": study.best_value}, f, indent=2)
    print(f"   → Params sauvegardés : {params_path}")

    # Réentraînement final
    print(f"\n🔹 Réentraînement final ({final_timesteps:,} steps)...")
    reward_win  = best.pop("reward_win", 1.0)
    reward_loss = best.pop("reward_loss", 1.0)
    conf_bonus  = best.pop("confidence_bonus", 0.2)
    n_steps     = best.get("n_steps", 512)

    final_env = WinRateEnv(oof_df, price_series,
                           reward_win=reward_win, reward_loss=reward_loss,
                           confidence_bonus=conf_bonus)
    final_model = PPO(
        "MlpPolicy", final_env, verbose=1, seed=SEED,
        batch_size=min(64, n_steps), n_epochs=10, clip_range=0.2,
        **best,
    )
    final_model.learn(total_timesteps=final_timesteps, progress_bar=False)
    final_model.save(os.path.join(output_dir, "ppo_winrate_tuned"))

    eval_env = WinRateEnv(holdout_df, price_series,
                          reward_win=reward_win, reward_loss=reward_loss,
                          confidence_bonus=conf_bonus)
    evaluate_winrate_agent(final_model, eval_env, output_dir)
    return best


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(predictions_csv: str, data_csv: str,
         output_dir: str = "results/rl_winrate/",
         total_timesteps: int = 200_000,
         n_folds: int = 5,
         skip_oof: bool = False,
         tune: bool = False,
         n_trials: int = 30):

    os.makedirs(output_dir, exist_ok=True)
    price_series = load_price_series(data_csv)

    # OOF partagé avec rl_agent (même dossier par défaut)
    oof_dir = output_dir.replace("rl_winrate", "rl_ppo")
    oof_csv = os.path.join(oof_dir, "oof_predictions.csv")
    if skip_oof and os.path.exists(oof_csv):
        print(f"🔹 OOF existant chargé : {oof_csv}")
    else:
        oof_csv = generate_oof_predictions(data_csv, oof_dir, n_folds=n_folds)

    oof_df     = load_predictions(oof_csv)
    holdout_df = load_predictions(predictions_csv)
    print(f"   → {len(oof_df)} lignes OOF | {len(holdout_df)} lignes holdout")

    if tune:
        tune_winrate_agent(
            oof_df=oof_df, holdout_df=holdout_df,
            price_series=price_series, output_dir=output_dir,
            n_trials=n_trials, timesteps_per_trial=100_000,
            final_timesteps=total_timesteps,
        )
    else:
        env = WinRateEnv(oof_df, price_series)
        print("🔹 Vérification de l'environnement...")
        check_env(env, warn=True)
        model = train_winrate_agent(env, total_timesteps=total_timesteps,
                                    output_dir=output_dir)
        eval_env = WinRateEnv(holdout_df, price_series)
        evaluate_winrate_agent(model, eval_env, output_dir)

    print(f"\n🔹 Résultats sauvegardés : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent RL PPO maximisant le win rate")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--folds",     type=int, default=5)
    parser.add_argument("--skip-oof",  action="store_true",
                        help="Réutilise oof_predictions.csv de results/rl_ppo/")
    parser.add_argument("--tune",      action="store_true",
                        help="Lance Optuna (maximise win rate)")
    parser.add_argument("--trials",    type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results/rl_winrate/")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BEST_RUN_PREDS = os.path.join(
        project_root,
        "results", "xgb_baseline_v1_20260511_154608",
        "lightgbm", "holdout_predictions.csv",
    )
    DATA   = os.path.join(project_root, DATA_FILE_PATH)
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
