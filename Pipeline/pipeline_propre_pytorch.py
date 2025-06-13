import pandas as pd
import numpy as np
from config import *
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from feature_selection import select_top_features_pca, select_top_features_shap

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ==================== REPRODUCTIBILITÉ ====================

# Fixer toutes les graines aléatoires
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration PyTorch pour la reproductibilité
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔹 Utilisation du device: {device}")

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

# ==================== CLASSES ET FONCTIONS ====================

class TimeSeriesDataset(Dataset):
    """Dataset personnalisé pour les séries temporelles"""
    def __init__(self, X, y, sequence_length, pred_horizon):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sequence_length = sequence_length
        self.pred_horizon = pred_horizon
        self.full_seq = sequence_length + pred_horizon
        
    def __len__(self):
        return len(self.X) - self.full_seq + 1
    
    def __getitem__(self, idx):
        # Prendre la séquence complète mais ne garder que sequence_length pour l'input
        x_seq = self.X[idx:idx + self.full_seq]  # Séquence complète
        y_target = self.y[idx + self.full_seq - 1]  # Target à la fin
        return x_seq, y_target

class LSTMModel(nn.Module):
    """Modèle LSTM pour la classification de séries temporelles"""
    def __init__(self, input_size, sequence_length, n_classes, hidden_size=64, dropout_rate=0.3, l2_reg=0.01):
        super(LSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Couches
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        
        # Pour la régularisation L2
        self.l2_reg = l2_reg
        
    def forward(self, x):
        # Tronquer les séquences pour ne garder que sequence_length pas
        x = x[:, :self.sequence_length, :]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Prendre seulement la dernière sortie
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        x = self.dropout(lstm_out)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def get_l2_loss(self):
        """Calcul de la perte L2 pour la régularisation"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss

class FocalLoss(nn.Module):
    """Implémentation de Focal Loss pour PyTorch"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early Stopping pour PyTorch"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def _is_better(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def calculate_metrics(y_true, y_pred):
    """Calcul des métriques personnalisées"""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    return acc, bal_acc, f1_macro, f1_weighted

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraînement d'une époque"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) + model.get_l2_loss()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc, bal_acc, f1_macro, f1_weighted = calculate_metrics(targets, predictions)
    
    return avg_loss, acc, bal_acc, f1_macro, f1_weighted

def validate_epoch(model, dataloader, criterion, device, ecart_min=0.05):
    """Validation d'une époque"""
    model.eval()
    total_loss = 0.0
    predictions = []
    predictions_confident = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y) + model.get_l2_loss()
            total_loss += loss.item()
            
            # Prédictions avec seuil de confiance
            probs = F.softmax(outputs, dim=1)
            for prob in probs:
                sorted_prob, _ = torch.sort(prob, descending=True)
                if sorted_prob[0] - sorted_prob[1] >= ecart_min:
                    predictions_confident.append(torch.argmax(prob).item())
                else:
                    predictions_confident.append(1)  # Classe neutre par défaut
            
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc, bal_acc, f1_macro, f1_weighted = calculate_metrics(targets, predictions)
    acc_conf, bal_acc_conf, f1_macro_conf, f1_weighted_conf = calculate_metrics(targets, predictions_confident)
    
    return (avg_loss, acc, bal_acc, f1_macro, f1_weighted, 
            acc_conf, bal_acc_conf, f1_macro_conf, f1_weighted_conf, predictions_confident)

# ==================== FONCTIONS PRINCIPALES ====================

def load_data(debug_on):
    print("🔹 1) Chargement des données depuis :", DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    print("   → Aperçu des 5 premières lignes :")
    print(df.head())

    if debug_on:
        input("   [PAUSE] Vérifiez le DataFrame chargé puis appuyez sur Entrée...")

    return df

def create_target(df, tag="", debug_on=False):
    print(f"🔹 2) Création de la cible (horizon = {PRED_HORIZON}) {tag}")
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    print("   → Avant suppression des NaN, nombre de lignes :", len(df))
    df.dropna(subset=['ret_future'], inplace=True)
    print("   → Après suppression, nombre de lignes :", len(df))
    print("   → Aperçu ret_future :")
    print(df[[DATE_COL, TARGET_PRICE_COL, 'ret_future']].head(10))

    if debug_on:    
        input("   [PAUSE] Vérifiez ret_future puis Entrée...")

    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        print(f"   → Seuils utilisés pour labellisation : {thresholds}")
        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2
        df["target"] = df["ret_future"].apply(label_target)
        print("   → Répartition des classes après labellisation :")
        print(df["target"].value_counts(normalize=True))
    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}")

    if debug_on:
        input("   [PAUSE] Vérifiez la colonne 'target' puis Entrée...")

    return df

def main(cross_validation, debug_on, feature_selection_on):
    # -- Chargement et split --
    df_raw = load_data(debug_on)
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    print(f"🔹 Colonnes restantes : {df_raw.columns.tolist()}")
    print(f"🔹 Total lignes après nettoyage : {len(df_raw)}")

    if debug_on:
        input("   [PAUSE] Vérifiez le nettoyage initial puis Entrée...")

    df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
    df_main    = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    print(f"🔹 Lignes pour entraînement (avant target) : {len(df_main)}")
    print(f"🔹 Lignes pour holdout (avant target)    : {len(df_holdout)}")

    if debug_on:
        input("   [PAUSE] Vérifiez les splits temporels puis Entrée...")

    df_main    = create_target(df_main, tag="(train)", debug_on=debug_on)
    df_holdout = create_target(df_holdout, tag="(holdout)", debug_on=debug_on)

    # -- Distribution des classes --
    print("🔹 Distribution classes TRAIN :")
    print(df_main["target"].value_counts(normalize=True))
    print("🔹 Distribution classes HOLDOUT :")
    print(df_holdout["target"].value_counts(normalize=True))
    if debug_on:
        input("   [PAUSE] Vérifiez distributions puis Entrée...")

    # -- Cross-validation ou pipeline complet --
    if cross_validation:
        print("🔹 Mode Validation Croisée activé")
        tscv = TimeSeriesSplit(n_splits=5)
        fold_stats = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_main)):
            print(f"\n========== FOLD {fold+1} ==========")
            df_train = df_main.iloc[train_idx].copy()
            df_val   = df_main.iloc[val_idx].copy()
            date_train = df_train[DATE_COL].values
            date_val   = df_val[DATE_COL].values
            print(f"👉 Train: {len(df_train)}, Val: {len(df_val)}")
            if debug_on:
                input("   [PAUSE] Vérifiez les indexes de split puis Entrée...")

            if feature_selection_on:
                # Sélection de features sur df_train uniquement
                print("🔹 Sélection features SHAP...")
                raw_features = select_top_features_shap(df_train, top_n=TOP_N_FEATURES, target_col="ret_future")
                features = [f for f in raw_features if f not in ('target', 'ret_future')]
                if 'SP500_historical_data_Close' not in features:
                    features.append('SP500_historical_data_Close')
                print(f"   → features finales utilisées : {features}")

                if debug_on:
                    input("   [PAUSE] Vérifiez features puis Entrée...")
            else:
                features = [f for f in df_train.columns if f not in ('target', 'ret_future')]

            # Préparation X/y
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train[features])
            X_val   = scaler.transform(df_val[features])
            y_train = df_train['target'].to_numpy()
            y_val   = df_val['target'].to_numpy()
            print("🔹 Shape X_train, X_val, y_train, y_val :", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            
            if debug_on: 
                input("   [PAUSE] Vérifiez les shapes puis Entrée...")

            # Datasets et DataLoaders
            train_dataset = TimeSeriesDataset(X_train, y_train, SEQUENCE_LENGTH, PRED_HORIZON)
            val_dataset = TimeSeriesDataset(X_val, y_val, SEQUENCE_LENGTH, PRED_HORIZON)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print("🔹 Nombre de batches train/val :", len(train_loader), len(val_loader))

            # Class weights
            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            cw_tensor = torch.FloatTensor(cw).to(device)
            print("🔹 Class weights pli :", cw)
            
            if debug_on:
                input("   [PAUSE] Vérifiez les datasets puis Entrée...")

            # Modèle
            model = LSTMModel(len(features), SEQUENCE_LENGTH, N_CLASSES).to(device)
            criterion = FocalLoss(alpha=cw_tensor, gamma=2.0)
            optimizer = optim.Adam(model.parameters())
            early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
            
            print("🔹 Modèle créé. Architecture :")
            print(model)

            if debug_on:
                input("   [PAUSE] Vérifiez l'architecture puis Entrée...")

            # Entraînement
            best_f1_macro = 0.0
            for epoch in range(EPOCHS):
                train_loss, train_acc, train_bal_acc, train_f1_macro, train_f1_weighted = train_epoch(
                    model, train_loader, criterion, optimizer, device)
                
                val_results = validate_epoch(model, val_loader, criterion, device, ECART_MIN)
                (val_loss, val_acc, val_bal_acc, val_f1_macro, val_f1_weighted, 
                 val_acc_conf, val_bal_acc_conf, val_f1_macro_conf, val_f1_weighted_conf, _) = val_results
                
                # if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val F1-macro: {val_f1_macro:.4f}")
                
                if early_stopping(val_f1_macro, model):
                    print(f"Early stopping à l'époque {epoch}")
                    break
            
            if debug_on:    
                input("   [PAUSE] Entraînement terminé. ⇨ Entrée pour prédictions...")

            # Évaluation finale du fold
            final_val_results = validate_epoch(model, val_loader, criterion, device, ECART_MIN)
            (_, val_acc, val_bal_acc, val_f1_macro, val_f1_weighted, 
             val_acc_conf, val_bal_acc_conf, val_f1_macro_conf, val_f1_weighted_conf, y_pred) = final_val_results
            
            # Exemples de dates + prédictions
            print(f"🔹 Quelques exemples de fenêtres et dates (Fold {fold+1}):")
            full_seq = SEQUENCE_LENGTH + PRED_HORIZON
            for j in range(min(3, len(y_pred))):
                start_date      = pd.to_datetime(date_val[j])
                end_input_date  = pd.to_datetime(date_val[j + SEQUENCE_LENGTH - 1])
                pred_date       = pd.to_datetime(date_val[j + full_seq - 1])
                print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}")
            
            if debug_on:    
                input("   [PAUSE] Vérifiez dates des fenêtres CV puis Entrée...")

            print(f"   → Fold {fold+1} | Acc: {val_acc:.4f} | BalAcc: {val_bal_acc:.4f} | F1-macro: {val_f1_macro:.4f} | F1-weighted: {val_f1_weighted:.4f}")
            fold_stats.append((val_acc, val_bal_acc, val_f1_macro, val_f1_weighted))

        # Résumé CV
        accs, bals, f1s, f1w = zip(*fold_stats)
        print("\n===== RÉSULTATS CV =====")
        print(f"Acc Moyenne: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"BalAcc Moyenne: {np.mean(bals):.4f} ± {np.std(bals):.4f}")
        print(f"F1-macro Moyenne: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"F1-weighted Moyenne: {np.mean(f1w):.4f} ± {np.std(f1w):.4f}")
        
        input("   [PAUSE] Fin validation croisée. Entrée pour pipeline final...")

    # -- Pipeline final sur holdout --
    if feature_selection_on:
        print("\n🔹 Sélection finale des features sur tout df_main")
        raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
        final_feats = [f for f in raw_feats if f not in ('target', 'ret_future')]
        if 'SP500_historical_data_Close' not in final_feats:
            final_feats.append('SP500_historical_data_Close')
        print("   → Features finales :", final_feats)
    else:
        final_feats = [f for f in df_main.columns if f not in ('target', 'ret_future')]
    
    if debug_on:
        input("   [PAUSE] Vérifiez les features finales puis Entrée...")

    # Préparation X/y finales
    final_scaler = StandardScaler()
    X_tr = final_scaler.fit_transform(df_main[final_feats])
    X_te = final_scaler.transform(df_holdout[final_feats])
    y_tr = df_main['target'].to_numpy()
    y_te = df_holdout['target'].to_numpy()
    date_te = df_holdout[DATE_COL].values
    print("🔹 Shapes finales X_tr, X_te, y_tr, y_te :", X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
        
    if debug_on:    
        input("   [PAUSE] Vérifiez les données finales puis Entrée...")

    # Datasets et DataLoaders finaux
    final_train_dataset = TimeSeriesDataset(X_tr, y_tr, SEQUENCE_LENGTH, PRED_HORIZON)
    final_test_dataset = TimeSeriesDataset(X_te, y_te, SEQUENCE_LENGTH, PRED_HORIZON)
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("🔹 Batches finaux train/holdout :", len(final_train_loader), len(final_test_loader))

    # Class weights finaux
    cw_f = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_f_tensor = torch.FloatTensor(cw_f).to(device)
    print("🔹 Class weights final :", cw_f)
        
    if debug_on:    
        input("   [PAUSE] Vérifiez les datasets finaux puis Entrée...")

    # Entraînement final
    final_model = LSTMModel(len(final_feats), SEQUENCE_LENGTH, N_CLASSES).to(device)
    final_criterion = FocalLoss(alpha=cw_f_tensor, gamma=2.0)
    final_optimizer = optim.Adam(final_model.parameters())
    final_early_stopping = EarlyStopping(patience=PATIENCE, mode='max')

    print("🔹 Entraînement final...")
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_bal_acc, train_f1_macro, train_f1_weighted = train_epoch(
            final_model, final_train_loader, final_criterion, final_optimizer, device)
        
        val_results = validate_epoch(final_model, final_test_loader, final_criterion, device, ECART_MIN)
        (val_loss, val_acc, val_bal_acc, val_f1_macro, val_f1_weighted, 
         val_acc_conf, val_bal_acc_conf, val_f1_macro_conf, val_f1_weighted_conf, _) = val_results
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val F1-macro: {val_f1_macro:.4f}")
        
        if final_early_stopping(val_f1_macro, final_model):
            print(f"Early stopping à l'époque {epoch}")
            break
            
    if debug_on:    
        input("   [PAUSE] Entraînement final OK. Entrée pour évaluation...")

    # Évaluation holdout
    print("\n--- Évaluation HOLDOUT ---")
    final_results = validate_epoch(final_model, final_test_loader, final_criterion, device, ECART_MIN)
    (_, acc_f, bal_f, f1m_f, f1w_f, 
     acc_f_conf, bal_f_conf, f1m_f_conf, f1w_f_conf, y_pred_final) = final_results

    # Exemples de dates + prédictions (Hold-Out)
    print("🔹 Quelques exemples de fenêtres et dates (Hold-Out):")
    full_seq = SEQUENCE_LENGTH + PRED_HORIZON
    for j in range(min(3, len(y_pred_final))):
        start_date     = pd.to_datetime(date_te[j])
        end_input_date = pd.to_datetime(date_te[j + SEQUENCE_LENGTH - 1])
        pred_date      = pd.to_datetime(date_te[j + full_seq - 1])
        print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}")
        
    if debug_on:    
        input("   [PAUSE] Vérifiez dates des fenêtres HoldOut puis Entrée...")

    # Scores finaux
    print(f"\n🔹 RÉSULTATS FINAUX (avec seuil de confiance {ECART_MIN}):")
    print(f"   → Accuracy final : {acc_f_conf:.4f}")
    print(f"   → Balanced Acc. : {bal_f_conf:.4f}")
    print(f"   → F1-macro      : {f1m_f_conf:.4f}")
    print(f"   → F1-weighted   : {f1w_f_conf:.4f}")

    # Récupérer les vraies targets pour la comparaison
    y_true_final = []
    for batch_x, batch_y in final_test_loader:
        y_true_final.extend(batch_y.numpy())
    y_true_final = np.array(y_true_final)

    print(f"\n🔹 Distribution des prédictions vs réalité:")
    for classe in [0, 1, 2]:
        pred_count = (np.array(y_pred_final) == classe).sum()
        true_count = (y_true_final == classe).sum()
        print(f"   → Classe {classe}: prédite={pred_count}, réelle={true_count}")

    if debug_on:
        input("   [PAUSE] Vérifiez les résultats puis Entrée...")

    # Sauvegarde du modèle
    model_path = f"Pipeline/model/model_v{MODEL_VERSION}_pytorch.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': {
            'input_size': len(final_feats),
            'sequence_length': SEQUENCE_LENGTH,
            'n_classes': N_CLASSES
        },
        'scaler': final_scaler,
        'features': final_feats
    }, model_path)
    print(f"🔹 Modèle sauvegardé: {model_path}")

if __name__ == "__main__":
    cross_validation = False  # False pour pipeline complet sans CV
    debug_on = False
    feature_selection_on = True # True avec feature selection, False sans
    ECART_MIN = 0.05

    print(f"\n--- Lancement pipeline PyTorch version : {MODEL_VERSION} ---")
    model_path = f"Pipeline/model/model_v{MODEL_VERSION}_pytorch.pth"
    if os.path.exists(model_path):
        print("⚠️  Le modèle existe déjà :", model_path)
        input("   Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")

    main(cross_validation, debug_on, feature_selection_on)