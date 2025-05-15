import pandas as pd

def load_and_prepare_data(path, prediction_horizon=7):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df["target"] = ((df["SP500_historical_data_Close"].shift(-prediction_horizon) - df["SP500_historical_data_Close"]) /
                    df["SP500_historical_data_Close"] > 0.05).astype(int)
    df = df.dropna()
    return df