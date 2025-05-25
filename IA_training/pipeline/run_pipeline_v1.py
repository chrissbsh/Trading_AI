import sys
sys.path.append('.')  # Ajoute le dossier courant au path

from IA_training.pipeline.data_preprocessing import load_and_prepare_data
from IA_training.pipeline.train_model_v1 import train
from IA_training.pipeline.backtester_v1 import backtest
import numpy as np
import pandas as pd
import pickle

data_path = "csv_data/consolidated_data/normalized_complete_data.csv"
prediction_horizon = 7
df = load_and_prepare_data(data_path, prediction_horizon=prediction_horizon)
features = [col for col in df.columns if col not in ["Date", "target"]]

model, scaler = train(df, features)

X_scaled = scaler.transform(df[features].values)
X_seq = np.array([X_scaled[i-30:i] for i in range(30, len(X_scaled))])
probas = model.predict(X_seq).flatten()

df_results = df.iloc[30:].copy()
df_results["proba"] = probas

best_sharpe = -np.inf
best_threshold = 0.5
for threshold in np.arange(0.5, 0.9, 0.01):
    result = backtest(df_results.copy(), threshold)
    sharpe = (result["returns"].mean() / result["returns"].std()) * np.sqrt(252) if result["returns"].std() != 0 else 0
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_threshold = threshold

final_result = backtest(df_results.copy(), best_threshold)
final_result.to_csv("prediction/backtest_results.csv", index=False)

model.save("IA_training/model/best_model.keras")
with open("IA_training/model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("IA_training/model/config.pkl", "wb") as f:
    pickle.dump({"threshold": best_threshold, "features": features}, f)

print(f"Best threshold = {best_threshold}, Sharpe = {best_sharpe:.4f}")
