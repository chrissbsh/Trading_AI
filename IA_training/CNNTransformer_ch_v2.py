import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

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
csv_file_path = 'csv_data/consolidated_data/normalized_complete_data.csv'
df = pd.read_csv(csv_file_path, parse_dates=['Date'])
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

# --- Split train / validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=32)

# --- Entraînement avec validation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformer(input_dim=X.shape[2], seq_len=seq_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 50
best_val_loss = float('inf')
patience, patience_counter = 10, 0  # Early stopping

save_dir = 'IA_training/model'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    # --- Entraînement ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # --- Early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        num_file = os.path.join(save_dir, 'model_num.txt')
        if os.path.exists(num_file):
            with open(num_file, 'r') as f:
                num = int(f.read().strip())
        else:
            num = 0
        num += 1
        with open(num_file, 'w') as f:
            f.write(str(num))
        file_name = f'best_model_{num}.pth'
        torch.save(model.state_dict(), os.path.join(save_dir, file_name))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break