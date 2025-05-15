import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def create_sequences(X, timesteps):
    X_seq = []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i])
    return np.array(X_seq)

def label_change(x):
    if x <= -0.043:
        return 0  # forte baisse
    elif x <= -0.009:
        return 1  # faible baisse
    elif x <= 0.017:
        return 2  # stable
    elif x <= 0.041:
        return 3  # faible hausse
    else:
        return 4  # forte hausse

def load_and_predict(data_path, date_col='Date', prediction_horizon=7, version="5_5"):
    model_path = f"IA_training/model/best_lstm_model_v{version}.keras"
    config_path = f"IA_training/model/model_config_v{version}.pkl"
    output_path = f"prediction/predictions_v{version}.csv"

    model = load_model(model_path)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    scaler = config["scaler"]
    features = config["features"]
    timesteps = config["timesteps"]

    df = pd.read_csv(data_path, parse_dates=[date_col]).sort_values(date_col)

    change = (df["SP500_historical_data_Close"].shift(-prediction_horizon) - df["SP500_historical_data_Close"]) / df["SP500_historical_data_Close"]
    df["target_multi"] = change.apply(label_change)
    df = df.dropna(subset=features + ["target_multi"])

    X = df[features].values
    X_scaled = scaler.transform(X)
    X_seq = create_sequences(X_scaled, timesteps)

    predictions_proba = model.predict(X_seq)
    predictions = np.argmax(predictions_proba, axis=1)

    actual_values = df["target_multi"].iloc[timesteps:].values
    results_df = pd.DataFrame({
        "Date": df.iloc[timesteps:][date_col].values,
        "Prediction": predictions,
        "Actual": actual_values,
        "SP500_historical_data_Close": df.iloc[timesteps:]["SP500_historical_data_Close"].values
    })

    results_df.to_csv(output_path, index=False)
    print(f"Prédictions sauvegardées : {output_path}")

    actual_values = df["target_multi"].iloc[timesteps:].values
    y_true = actual_values.astype(int)
    y_pred = predictions.astype(int)

    # Evaluation
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Évaluation Multi-Class ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")

    results_df = pd.DataFrame({
        "Date": df.iloc[timesteps:][date_col].values,
        "Prediction": predictions,
        "Actual": actual_values
    })
    results_df["SP500_historical_data_Close"] = df.iloc[timesteps:]["SP500_historical_data_Close"].values
    results_df.to_csv(output_path, index=False)
    print(f"\nPrédictions sauvegardées : {output_path}")
    print(results_df.tail(5))

    # Ecriture fichier metrics summary
    metrics_output_path = output_path.replace(f"_v{version}.csv", f"_metrics_summary_v{version}.txt")
    print(f"\nMetrics sauvegardées dans : {metrics_output_path}")

    with open(metrics_output_path, "w") as f:
        f.write("=== Resultats du modele (multi-class) ===\n")
        f.write(f"Total predictions : {len(y_true)}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 Score (weighted) : {f1:.4f}\n")
        f.write(f"Precision (weighted) : {prec:.4f}\n")
        f.write(f"Recall (weighted) : {rec:.4f}\n")
        f.write("\n=== Confusion Matrix ===\n")
        f.write(np.array2string(cm))

    print("\n=== Confusion Matrix ===")
    print(cm)

    return results_df


def backtest_portfolio_multiclass(df, initial_cash=1000, transaction_fee=0.0005):
    """
    Backtest multi-class : sizing dynamique
    """
    df = df.copy()
    prices = df["SP500_historical_data_Close"].values
    signals = df["Prediction"].values

    cash = initial_cash
    position = 0
    portfolio_values = []

    size_mapping_pct = {0: -0.8, 1: -0.4, 2: 0, 3: 0.4, 4: 0.8}

    for price, signal in zip(prices, signals):
        target_pct = size_mapping_pct.get(signal, 0)
        total_value = cash + position * price
        target_position_value = total_value * target_pct
        current_position_value = position * price
        difference = target_position_value - current_position_value

        if abs(difference) > 0:  # Si on doit ajuster
            qty_change = difference / price

            # Achat
            if qty_change > 0:
                cost = qty_change * price * (1 + transaction_fee)
                if cost <= cash:
                    cash -= cost
                    position += qty_change
                else:
                    # Acheter tout le cash dispo
                    max_qty = cash / (price * (1 + transaction_fee))
                    cash -= max_qty * price * (1 + transaction_fee)
                    position += max_qty

            # Vente
            elif qty_change < 0:
                qty_to_sell = min(abs(qty_change), position)
                revenue = qty_to_sell * price * (1 - transaction_fee)
                cash += revenue
                position -= qty_to_sell

        portfolio_values.append(cash + position * price)


    df["PortfolioValue"] = portfolio_values
    df["Returns"] = df["PortfolioValue"].pct_change()
    df["CumulativeReturn"] = df["PortfolioValue"] / initial_cash

    pnl = df["PortfolioValue"].iloc[-1] - initial_cash
    pnl_pct = pnl / initial_cash
    max_dd = ((df["PortfolioValue"].cummax() - df["PortfolioValue"]) / df["PortfolioValue"].cummax()).max()
    sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252) if df["Returns"].std() != 0 else 0
    sp500_returns = df["SP500_historical_data_Close"].iloc[-1] / df["SP500_historical_data_Close"].iloc[0] - 1

    print("\n=== Backtest Portfolio Multi-Class ===")
    print(f"PnL final : {pnl:.2f} USD")
    print(f"PnL % : {pnl_pct:.2%}")
    print(f"SP500 PnL % : {sp500_returns:.2%}")
    print(f"Max Drawdown : {max_dd:.2%}")
    print(f"Sharpe Ratio : {sharpe:.2f}")

    return df

if __name__ == "__main__":
    version = input("Entrez le numéro de version (ex: 1, 2, 3...) : ")
    data_path = "csv_data/consolidated_data/normalized_complete_data.csv"
    result_df = load_and_predict(data_path=data_path, version=version)
    result_df_bt = backtest_portfolio_multiclass(result_df)
    result_df_bt.to_csv(f"prediction/backtest_results_v{version}.csv", index=False)