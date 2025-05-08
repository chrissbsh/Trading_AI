import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class CNNTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, num_heads=4, num_layers=2, output_len=7):
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
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.cnn(x)         # (batch_size, d_model, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model)
        x = x + self.pos_embedding  # Add positional encoding
        x = self.transformer(x)     # (batch_size, seq_len, d_model)
        x = x.flatten(start_dim=1)  # (batch_size, seq_len * d_model)
        out = self.fc_out(x)        # (batch_size, output_len)
        return out
    
# Chargement des données
df = pd.read_csv('csv_data/consolidated_data/prepared_data.csv', parse_dates=['date'])

# Tri et nettoyage
df = df.sort_values('date')
features = df.drop(columns=['date']).values

# Normalisation si ce n’est pas déjà fait
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Préparation des séquences (fenêtre glissante)
def create_sequences(data, seq_length=60, pred_length=7):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0]  # Target: 1ère colonne (ex: Close)
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

seq_len = 60
pred_len = 7
X, y = create_sequences(features_scaled, seq_len, pred_len)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# DataLoader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNTransformer(input_dim=X.shape[2], seq_len=seq_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        print("Sample prediction:", outputs[0].cpu().numpy())
        print("True values:", yb[0].cpu().numpy())
        break