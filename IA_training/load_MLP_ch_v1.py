import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# Charger les données
df = pd.read_csv('csv_data/consolidated_data/normalized_complete_data.csv', parse_dates=['Date'])

# Génération de la target à +7 jours
df['target_7d'] = (df['Close'].shift(-7) > df['Close']).astype(int)
df = df.iloc[:-7]  # Supprime les 7 dernières lignes pour éviter les labels bruités

# Sélection des features et de la target
feat_cols = df.columns.difference(['Date', 'Close', 'target_7d'])
X = df[feat_cols]
y = df['target_7d']

# Fonction pour évaluer le modèle sur une période donnée
def evaluate_model_on_period(start_date, end_date, model, scaler, feat_cols, device):
    test = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    X_test = test[feat_cols]
    y_test = test['target_7d']

    X_test = scaler.transform(X_test)

    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        pred_labels = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, pred_labels)
    recall = recall_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    conf_matrix = confusion_matrix(y_test, pred_labels)

    print(f"Test AUC for period {start_date} to {end_date}: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Redéfinir la classe du modèle
class MLPBaseline(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128,  64),    nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64,    1)
        )
    def forward(self, x):
        return self.net(x)

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPBaseline(dim_in=X.shape[1]).to(device)
model.load_state_dict(torch.load("best_mlp.pth", map_location=device, weights_only=True))
model.eval()

# Charger le scaler
scaler = StandardScaler()
scaler.fit(df[(df['Date'] >= '2005-02-25') & (df['Date'] <= '2019-12-31')][feat_cols])

# Évaluer le modèle sur différentes périodes
evaluate_model_on_period('2020-01-02', '2020-12-31', model, scaler, feat_cols, device)
evaluate_model_on_period('2021-01-01', '2021-12-31', model, scaler, feat_cols, device)
evaluate_model_on_period('2022-01-01', '2022-12-31', model, scaler, feat_cols, device)
evaluate_model_on_period('2023-01-01', '2023-12-31', model, scaler, feat_cols, device)
evaluate_model_on_period('2024-01-01', '2024-12-31', model, scaler, feat_cols, device)
evaluate_model_on_period('2024-04-14', '2025-04-14', model, scaler, feat_cols, device)