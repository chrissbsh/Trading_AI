# evaluate_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import joblib

# --- 1. Définition du MLPBaseline (même architecture que pour l'entraînement) ---
class MLPBaseline(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128,  64),    nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64,    1)   # logits
        )
    def forward(self, x):
        return self.net(x)

# --- 2. Chargement du modèle et du scaler ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPBaseline(dim_in=64).to(device)
model.load_state_dict(torch.load("IA_training/MLP_model/best_mlp.pth", map_location=device, weights_only=True))
model.eval()

# Assurez-vous d'avoir sauvegardé votre StandardScaler après entraînement :
# joblib.dump(scaler, "scaler.pkl")
scaler = joblib.load("IA_training/MLP_model/scaler.pkl")

# --- 3. Préparation des données ---
df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"])
df["target_7d"] = (df["Close"].shift(-7) > df["Close"]).astype(int)
df = df.iloc[:-7]  # retirer les dernières lignes

# --- 4. Fonction d'évaluation sur une période donnée ---
def evaluate_period(start_date, end_date):
    # Filtrer la période
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    sub = df.loc[mask]
    X = sub.drop(columns=["Date", "Close", "target_7d"])
    y = sub["target_7d"]

    # 1) Transformez X en DataFrame réindexé sur feature_names_in_
    X = X.reindex(columns=scaler.feature_names_in_)
    # 2) Maintenant X a les bons noms, on peut transformer sans warning
    X_scaled = scaler.transform(X)

    # Prédictions
    with torch.no_grad():
        X_t = torch.from_numpy(X_scaled).float().to(device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Binarisation
    preds = (probs > 0.5).astype(int)

    # Calcul des métriques
    results = {
        "AUC": roc_auc_score(y, probs),
        "PR_AUC": average_precision_score(y, probs),
        "F1": f1_score(y, preds),
        "Balanced_Acc": balanced_accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
    }

    # Affichage
    print(f"=== Période {start_date} → {end_date} ===")
    for k, v in results.items():
        print(f"{k:12s}: {v:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    print()

    return results

# --- 5. Exemple d'utilisation ---
if __name__ == "__main__":
    periods = [
        ("2018-01-01", "2018-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2023-01-03", "2025-04-14"),
    ]
    for start, end in periods:
        evaluate_period(start, end)