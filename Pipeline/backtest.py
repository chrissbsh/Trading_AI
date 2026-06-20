import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pipeline.config import *

def run_backtest(pred_df, price_series, output_dir):
    """
    Backtest long/short basé sur les prédictions du modèle.
    Signal : classe 2 -> long, classe 0 -> short, classe 1 -> flat.
    Cooldown PRED_HORIZON : une seule position ouverte à la fois —
    après chaque trade, les signaux suivants sont ignorés pendant
    PRED_HORIZON jours pour éviter le double-comptage des positions chevauchantes.
    """
    df = pred_df.copy()
    df["date_prediction"] = pd.to_datetime(df["date_prediction"])

    price = price_series.copy()
    price.index = pd.to_datetime(price.index)

    # Filtre de confiance : si le gap top1-top2 < ECART_MIN -> flat
    if "confidence_gap" in df.columns:
        df["signal_filtered"] = df.apply(
            lambda r: r["y_pred"] if r["confidence_gap"] >= ECART_MIN else 1, axis=1
        )
    else:
        df["signal_filtered"] = df["y_pred"]

    n_filtered = (df["signal_filtered"] != df["y_pred"]).sum()
    if n_filtered > 0:
        print(f"   -> {n_filtered} trades filtrés par ECART_MIN={ECART_MIN:.2f}")

    # Calcule le rendement de chaque signal (sans cooldown d'abord)
    raw_returns = []
    for _, row in df.iterrows():
        entry_date = row["date_prediction"]
        signal = row["signal_filtered"]
        if signal == 1:
            raw_returns.append(0.0)
            continue
        try:
            entry_price = price.asof(entry_date)
            exit_date = entry_date + pd.offsets.BusinessDay(PRED_HORIZON)
            exit_price = price.asof(exit_date)
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                raw_returns.append(0.0)
                continue
            ret = (exit_price - entry_price) / entry_price
            raw_returns.append(ret if signal == 2 else -ret)
        except Exception:
            raw_returns.append(0.0)

    df["_raw_return"] = raw_returns

    # Applique le cooldown : une position à la fois, bloque PRED_HORIZON jours après chaque trade
    executed_signal = []
    executed_return = []
    cooldown = 0
    for _, row in df.iterrows():
        if cooldown > 0:
            executed_signal.append(1)
            executed_return.append(0.0)
            cooldown -= 1
        elif row["signal_filtered"] != 1:
            executed_signal.append(row["signal_filtered"])
            executed_return.append(row["_raw_return"])
            cooldown = PRED_HORIZON - 1
        else:
            executed_signal.append(1)
            executed_return.append(0.0)

    df["signal_executed"] = executed_signal
    df["trade_return"] = executed_return
    df["cumulative_return"] = (1 + df["trade_return"]).cumprod()

    n_trades = int((df["signal_executed"] != 1).sum())
    total_return = float(df["cumulative_return"].iloc[-1] - 1) if len(df) else 0.0
    active = df.loc[df["signal_executed"] != 1, "trade_return"]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    periods_per_year = 252 / PRED_HORIZON
    r_arr = df["trade_return"].values
    sharpe = float(r_arr.mean() / r_arr.std() * np.sqrt(periods_per_year)) if r_arr.std() > 0 else 0.0
    rolling_max = df["cumulative_return"].cummax()
    drawdown = float(((df["cumulative_return"] - rolling_max) / rolling_max).min())

    # Buy-and-hold SP500 sur la même période
    dates = df["date_prediction"]
    first_p = price.asof(dates.iloc[0])
    last_exit = dates.iloc[-1] + pd.offsets.BusinessDay(PRED_HORIZON)
    last_p = price.asof(last_exit)
    bh_ret = float((last_p - first_p) / first_p) if not pd.isna(last_p) and first_p > 0 else 0.0
    bh_curve = pd.Series([price.asof(d) / first_p for d in dates], index=dates)
    bh_r = bh_curve.pct_change().fillna(0.0)
    bh_sharpe = float(bh_r.mean() / bh_r.std() * np.sqrt(252)) if bh_r.std() > 0 else 0.0

    print("\n===== BACKTEST FINANCIER =====")
    print(f"   Nombre de trades (hors neutre) : {n_trades}")
    print(f"   Rendement bot                  : {total_return*100:.2f}%")
    print(f"   Rendement buy-and-hold         : {bh_ret*100:.2f}%")
    print(f"   Win rate                       : {win_rate*100:.1f}%")
    print(f"   Sharpe bot                     : {sharpe:.2f}")
    print(f"   Sharpe buy-and-hold            : {bh_sharpe:.2f}")
    print(f"   Max Drawdown                   : {drawdown*100:.2f}%")

    backtest_df = df[["date_prediction", "y_true", "y_pred", "signal_filtered",
                       "signal_executed", "trade_return", "cumulative_return"]]
    backtest_df.to_csv(os.path.join(output_dir, "backtest_results.csv"), index=False)

    backtest_metrics = pd.DataFrame([{
        "n_trades":     n_trades,
        "total_return": total_return,
        "bh_return":    bh_ret,
        "win_rate":     win_rate,
        "sharpe_ratio": sharpe,
        "bh_sharpe":    bh_sharpe,
        "max_drawdown": drawdown,
    }])
    backtest_metrics.to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)

    # Graphique equity curve bot vs buy-and-hold
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, df["cumulative_return"].values, color="darkorange", linewidth=1.5,
            label=f"Stratégie modèle  {total_return*100:+.1f}%")
    ax.plot(dates, bh_curve.values, color="steelblue", linewidth=1.5, linestyle="--",
            label=f"Buy & Hold SP500  {bh_ret*100:+.1f}%")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Backtest — Rendement cumulé")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement cumulé")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_equity_curve.png"), dpi=150)
    plt.close()

    return backtest_metrics