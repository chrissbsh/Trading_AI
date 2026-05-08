import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pipeline.config import *

def run_backtest(pred_df, price_series, output_dir):
    """
    Backtest simple long/short basé sur les prédictions du modèle.
    Signal : classe 2 → long, classe 0 → short, classe 1 → flat.
    Entrée le jour de prédiction, sortie PRED_HORIZON jours plus tard.
    Hypothèses : pas de coût de transaction, position unitaire.
    """
    df = pred_df.copy()
    df["date_prediction"] = pd.to_datetime(df["date_prediction"])

    price = price_series.copy()
    price.index = pd.to_datetime(price.index)

    returns = []
    for _, row in df.iterrows():
        entry_date = row["date_prediction"]
        signal = row["y_pred"]
        if signal == 1:
            returns.append(0.0)
            continue
        try:
            entry_price = price.asof(entry_date)
            exit_date = entry_date + pd.offsets.BusinessDay(PRED_HORIZON)
            exit_price = price.asof(exit_date)
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                returns.append(0.0)
                continue
            ret = (exit_price - entry_price) / entry_price
            returns.append(ret if signal == 2 else -ret)
        except Exception:
            returns.append(0.0)

    df["trade_return"] = returns
    df["cumulative_return"] = (1 + df["trade_return"]).cumprod()

    n_trades = (df["y_pred"] != 1).sum()
    total_return = df["cumulative_return"].iloc[-1] - 1 if len(df) else 0.0
    win_rate = (df.loc[df["y_pred"] != 1, "trade_return"] > 0).mean() if n_trades else 0.0
    daily_ret = df["trade_return"]
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    rolling_max = df["cumulative_return"].cummax()
    drawdown = ((df["cumulative_return"] - rolling_max) / rolling_max).min()

    print("\n===== BACKTEST FINANCIER =====")
    print(f"   Nombre de trades (hors neutre) : {n_trades}")
    print(f"   Rendement cumulé               : {total_return*100:.2f}%")
    print(f"   Win rate                        : {win_rate*100:.1f}%")
    print(f"   Sharpe ratio (annualisé)        : {sharpe:.2f}")
    print(f"   Max Drawdown                    : {drawdown*100:.2f}%")

    backtest_df = df[["date_prediction", "y_true", "y_pred", "trade_return", "cumulative_return"]]
    backtest_df.to_csv(os.path.join(output_dir, "backtest_results.csv"), index=False)

    backtest_metrics = pd.DataFrame([{
        "n_trades": int(n_trades),
        "total_return": total_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
    }])
    backtest_metrics.to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(df["date_prediction"], df["cumulative_return"], label="Stratégie modèle")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Backtest — Rendement cumulé")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulé")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_equity_curve.png"), dpi=150)
    plt.close()

    return backtest_metrics