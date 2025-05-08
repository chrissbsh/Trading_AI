import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# --- Définition du modèle ---
class CNNTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, num_heads=4, num_layers=2, output_len=21):
        super(CNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model * seq_len, output_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        out = self.fc_out(x)
        return out

# --- Chargement et préparation des données ---
df = pd.read_csv('csv_data/consolidated_data/normalized_consolidated_data.csv', parse_dates=['Date'])
df = df.sort_values('Date')

features = df.drop(columns=['Date']).values

def create_sequences(data, seq_length=60, pred_length=21):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0]  # 0 = colonne cible
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

seq_len = 60
pred_len = 21
X, y = create_sequences(features, seq_len, pred_len)

# --- Chargement du modèle ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformer(input_dim=X.shape[2], seq_len=seq_len).to(device)
model.load_state_dict(torch.load('IA_training/model/best_model.pth', weights_only=True, map_location=device))
model.eval()

# --- Faire des prédictions ---
def predict_samples(model, X, y, num_samples=5):
    """
    Visualise 'num_samples' prédictions aléatoires.
    """
    import random
    indices = random.sample(range(len(X)), num_samples)

    for idx in indices:
        x_input = X[idx].unsqueeze(0).to(device)
        y_true = y[idx].cpu().numpy()

        with torch.no_grad():
            y_pred = model(x_input).cpu().numpy().flatten()

        print("y_true: ", y_true)
        print("y_pred: ", y_pred)

        # --- Plot ---
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(y_true)), y_true, label='Réel', marker='o')
        plt.plot(range(len(y_pred)), y_pred, label='Prédit', marker='x')
        plt.title(f"Prédiction du S&P500 sur {len(y_true)} jours (échantillon #{idx})")
        plt.xlabel("Jour")
        plt.ylabel("Cours Normalisé")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# --- Appel de la fonction pour visualiser 5 prédictions ---
predict_samples(model, X, y, num_samples=5)