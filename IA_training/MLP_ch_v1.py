import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Fix seed for reproducibility
# np.random.seed(42)

# Load data
df = pd.read_csv('csv_data/consolidated_data/normalized_complete_data.csv', parse_dates=['Date'])

# 1. Génération de la target à +7 jours
df['target_7d'] = (df['Close'].shift(-7) > df['Close']).astype(int)
df = df.iloc[:-7]  # Supprime les 7 dernières lignes pour éviter les labels bruités

# 2. Sélection des features et de la target
feat_cols = df.columns.difference(['Date', 'Close', 'target_7d'])
X = df[feat_cols]
y = df['target_7d']

# 3. Découpage temporel
train = df[(df['Date'] >= '2005-02-25') & (df['Date'] <= '2019-12-31')]
val   = df[(df['Date'] >= '2020-01-02') & (df['Date'] <= '2022-12-30')]
test  = df[(df['Date'] >= '2023-01-03') & (df['Date'] <= '2025-04-14')]

X_train, y_train = train[feat_cols], train['target_7d']
X_val,   y_val   = val[feat_cols],   val['target_7d']
X_test,  y_test  = test[feat_cols],  test['target_7d']

# 4. Ré-normalisation (stats calculées uniquement sur le train)
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, "IA_training/MLP_model/scaler.pkl")
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 5. Vérification des dimensions et déséquilibre
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Distribution train target:", y_train.value_counts(normalize=True).to_dict())


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

# 2. Préparez les tenseurs et le DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_loader(X, y, bs, shuffle=False):
    tx = torch.from_numpy(X).float().to(device)
    ty = torch.from_numpy(y.values if hasattr(y, 'values') else y).float().unsqueeze(1).to(device)
    return DataLoader(TensorDataset(tx, ty), batch_size=bs, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, bs=64, shuffle=True)
val_loader   = to_loader(X_val,   y_val,   bs=128)
test_loader  = to_loader(X_test,  y_test,  bs=128)

# 3. Définition du MLP baseline
class MLPBaseline(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128,  64),    nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64,    1)  # logits
        )
    def forward(self, x):
        return self.net(x)

model = MLPBaseline(X_train.shape[1]).to(device)

# 4. Critère et optimiseur
pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = neg / pos
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight).to(device))
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='max')

# 5. Entraînement avec Early Stopping sur AUC validation
best_auc, patience, counter = 0.0, 10, 0
val_auc_list = []
for epoch in range(1, 101):
    # train
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    auc = roc_auc_score(targets, preds)
    val_auc_list.append(auc)
    scheduler.step(auc)
    print(f"Epoch {epoch:03d} – Val AUC: {auc:.4f}")
    if auc > best_auc + 1e-3:
        best_auc, counter = auc, 0
        torch.save(model.state_dict(), "IA_training/MLP_model/best_mlp.pth")
    else:
        counter += 1
    if counter >= patience:
        print("→ Early stopping")
        break

# 6. Évaluation finale sur le test set
model.load_state_dict(torch.load("IA_training/MLP_model/best_mlp.pth"))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(torch.sigmoid(model(xb)).cpu())
        targets.append(yb.cpu())
preds = torch.cat(preds).numpy()
targets = torch.cat(targets).numpy()
test_auc = roc_auc_score(targets, preds)
print(f"Test AUC: {test_auc:.4f}")

# --- 2) Tracé de la courbe ---
import matplotlib.pyplot as plt

epochs = range(1, len(val_auc_list) + 1)
plt.plot(epochs, val_auc_list)
plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.title("Courbe de Validation AUC")
plt.show()

# import torch

# 1) Redéfinir la classe du modèle (même code que lors de l'entraînement)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPBaseline(dim_in=64).to(device)
model.load_state_dict(torch.load("IA_training/MLP_model/best_mlp.pth", map_location=device))
model.eval()

# 2) Préparer les données de test
X_test_t = torch.from_numpy(X_test).float().to(device)

# 3) Générer les probabilités
with torch.no_grad():
    logits = model(X_test_t)
    probs  = torch.sigmoid(logits).cpu().numpy().flatten()


# 4) Afficher un extrait
print("Premières probabilités de hausse :", probs[:10])
# Pour obtenir des labels 0/1 à 0.5 :
pred_labels = (probs > 0.5).astype(int)
print("Premiers labels prédits :", pred_labels[:10])
print("Labels réels           :", y_test.values[:10])