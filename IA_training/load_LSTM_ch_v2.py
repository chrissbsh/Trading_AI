import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def create_sequences(X, timesteps):
    X_seq = []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i])
    return np.array(X_seq)

def load_and_predict(
    data_path,
    date_col='Date',
    prediction_horizon=7,
    version="1",
    model_path=None,
    config_path=None,
    output_path=None
):
    # Utilisation du numéro de version pour les chemins de fichiers
    if model_path is None:
        model_path = f"IA_training/model/best_lstm_model_v{version}.keras"
    if config_path is None:
        config_path = f"IA_training/model/model_config_v{version}.pkl"
    if output_path is None:
        output_path = f"prediction/predictions_v{version}.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config introuvable : {config_path}")

    model = load_model(model_path)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    scaler = config["scaler"]
    threshold = config["threshold"]
    features = config["features"]
    timesteps = config["timesteps"]

    print(f"Chargement des données depuis {data_path}...")
    df = pd.read_csv(data_path, parse_dates=[date_col]).sort_values(date_col)

    # Ajout des colonnes manquantes
    if "std_21" not in df.columns:
        df["std_21"] = df["sp500_return_1d"].rolling(21).std()
    if "hv_30" not in df.columns:
        df["hv_30"] = df["sp500_return_1d"].rolling(30).std()
    if "r_sp_gold" not in df.columns:
        df["r_sp_gold"] = df["SP500_historical_data_Close"] / df["gold_historical_data_Close"]
    if "r_sp_dxy" not in df.columns:
        df["r_sp_dxy"] = df["SP500_historical_data_Close"] / df["dollar_index_historical_data_Close"]
    if "r_sp_bond" not in df.columns:
        df["r_sp_bond"] = df["SP500_historical_data_Close"] / df["Market_yield_US_10_year_DGS10"]
    if "vix_direction" not in df.columns:
        vix = df["^VIX_historical_data_Close"]
        df["vix_direction"] = vix.diff().fillna(0).gt(0).astype(int)
    if "vix_high" not in df.columns:
        vix = df["^VIX_historical_data_Close"]
        df["vix_high"] = vix.gt(vix.rolling(63).median()).astype(int)
    if "PMI" in df.columns and "macro_regime" not in df.columns:
        df["macro_regime"] = (df["PMI"] > 50).astype(int)

    # Recalcul du target_7d = obligatoire
    df["target_7d"] = (
        df["SP500_historical_data_Close"].shift(-prediction_horizon) > df["SP500_historical_data_Close"]
    ).astype(int)

    # Vérification des colonnes
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Colonnes manquantes : {missing_features}")

    df = df.dropna(subset=features)
    assert set(df["target_7d"].dropna().unique()).issubset({0, 1}), "target_7d contient autre chose que 0/1"

    X = df[features].values
    X_scaled = scaler.transform(X)
    X_seq = create_sequences(X_scaled, timesteps)

    if len(X_seq) == 0:
        print(f"Pas assez de données (minimum {timesteps} points nécessaires)")
        return pd.DataFrame()

    print(f"Prédiction sur {len(X_seq)} séquences...")
    predictions_proba = model.predict(X_seq).flatten()
    predictions = (predictions_proba > threshold).astype(int)

    actual_values = df["target_7d"].iloc[timesteps:].values
    valid_indices = ~np.isnan(actual_values)

    y_true = actual_values[valid_indices].astype(int)
    y_pred = predictions[valid_indices]

    print("\n=== Évaluation ===")
    try:
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    except ValueError:
        print("Multiclass détecté → utilisation de weighted average")
        print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")

    results_df = pd.DataFrame({
        "Date": df.iloc[timesteps:][date_col].values,
        "Probability": predictions_proba,
        "Prediction": predictions,
        "Actual": actual_values
    })

    results_df["SP500_historical_data_Close"] = df.iloc[timesteps:]["SP500_historical_data_Close"].values

    results_df.to_csv(output_path, index=False)
    print(f"\nPrédictions sauvegardées : {output_path}")
    print(results_df.tail(5))

    # Utilisation du numéro de version pour le fichier metrics
    metrics_output_path = output_path.replace(f"_v{version}.csv", f"_metrics_summary_v{version}.txt")

    # Calcul des métriques
    total_predictions = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    incorrect_predictions = total_predictions - correct_predictions

    acc = accuracy_score(y_true, y_pred)
    try:
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
    except ValueError:
        f1 = f1_score(y_true, y_pred, average='weighted')
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')

    # Affichage console
    print("\n=== Résultats détaillés ===")
    print(f"Total prédictions : {total_predictions}")
    print(f"Bonnes prédictions : {correct_predictions}")
    print(f"Mauvaises prédictions : {incorrect_predictions}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall : {rec:.4f}")

    # Ajout de la confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n=== Confusion Matrix ===")
    print(f"True Negatives (prévu 0 / vrai 0) : {tn}")
    print(f"False Positives (prévu 1 / vrai 0) : {fp}")
    print(f"False Negatives (prévu 0 / vrai 1) : {fn}")
    print(f"True Positives (prévu 1 / vrai 1) : {tp}")

    # Sauvegarde dans fichier texte
    with open(metrics_output_path, "w") as f:
        f.write("\n=== Resultats du modele ===\n")
        f.write(f"Total predictions : {total_predictions}\n")
        f.write(f"Bonnes predictions : {correct_predictions}\n")
        f.write(f"Mauvaises predictions : {incorrect_predictions}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall : {rec:.4f}\n")

        f.write("\n=== Confusion Matrix ===\n")
        f.write(f"True Negatives (prevu 0 / vrai 0) : {tn}\n")
        f.write(f"False Positives (prevu 1 / vrai 0) : {fp}\n")
        f.write(f"False Negatives (prevu 0 / vrai 1) : {fn}\n")
        f.write(f"True Positives (prevu 1 / vrai 1) : {tp}\n")

    print(f"\nMetrics sauvegardées dans : {metrics_output_path}")

    return results_df


def backtest_portfolio(df, initial_cash=10000, transaction_fee=0.0005, position_size=1.0):
    """
    Backtest simple basé sur les signaux de prediction.
    Achat si signal = 1, vente si signal = 0.
    """
    df = df.copy()
    if "SP500_historical_data_Close" not in df.columns:
        raise ValueError("La colonne SP500_historical_data_Close est manquante dans le DataFrame.")

    df["Price"] = df["SP500_historical_data_Close"]

    cash = initial_cash
    position = 0
    portfolio_values = []

    for i in range(len(df)):
        price = df.iloc[i]["Price"]
        signal = df.iloc[i]["Prediction"]

        # Acheter
        if signal == 1 and position == 0:
            qty = (cash * position_size) / price
            cash -= qty * price * (1 + transaction_fee)
            position = qty


        # Vendre
        elif signal == 0 and position > 0:
            revenue = position * price * (1 - transaction_fee)
            cash += revenue
            position = 0

        # Valeur totale portefeuille
        total_value = cash + position * price
        portfolio_values.append(total_value)

    df["PortfolioValue"] = portfolio_values
    df["Returns"] = df["PortfolioValue"].pct_change()
    df["CumulativeReturn"] = df["PortfolioValue"] / initial_cash

    # Calculs métriques
    pnl = df["PortfolioValue"].iloc[-1] - initial_cash
    pnl_pct = pnl / initial_cash
    max_drawdown = ((df["PortfolioValue"].cummax() - df["PortfolioValue"]) / df["PortfolioValue"].cummax()).max()
    sharpe_ratio = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252) if df["Returns"].std() != 0 else 0

    # metriques sur sp500
    sp500_returns = df["SP500_historical_data_Close"].iloc[-1] / df["SP500_historical_data_Close"].iloc[0]

    print("\n=== Backtest Portfolio ===")
    print(f"PnL final : {pnl:.2f} USD")
    print(f"PnL % : {pnl_pct:.2%}")
    print(f"SP500 PnL % : {sp500_returns:.2%}")
    print(f"Max Drawdown : {max_drawdown:.2%}")
    print(f"Sharpe Ratio : {sharpe_ratio:.2f}")

    return df



if __name__ == "__main__":
    version = input("Entrez le numéro de version (ex: 1, 2, 3...) : ")
    data_path = "csv_data/consolidated_data/normalized_complete_data.csv"
    date_col = "Date"
    prediction_horizon = 7

    result_df = load_and_predict(
        data_path=data_path,
        date_col=date_col,
        prediction_horizon=prediction_horizon,
        version=version
    )

    result_df_bt = backtest_portfolio(result_df)
    result_df_bt.to_csv(f"prediction/backtest_results_v{version}.csv", index=False)