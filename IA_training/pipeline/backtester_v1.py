import numpy as np
import pandas as pd

def backtest(df, threshold, initial_cash=10000, fee=0.0005):
    cash = initial_cash
    position = 0
    values = []

    for price, prob in zip(df["SP500_historical_data_Close"], df["proba"]):
        signal = 1 if prob > threshold else 0
        if signal == 1 and position == 0:
            position = cash * 0.95 / price
            cash -= position * price * (1 + fee)
        elif signal == 0 and position > 0:
            cash += position * price * (1 - fee)
            position = 0
        values.append(cash + position * price)

    df["portfolio"] = values
    df["returns"] = df["portfolio"].pct_change()
    df["cum_returns"] = df["portfolio"] / initial_cash
    return df
