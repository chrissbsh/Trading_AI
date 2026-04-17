import pandas as pd
import numpy as np

"""
Ce fichier a pour but d'explorer différentes combinaisons de paramètres `pred_horizon` (horizon de prédiction) et `threshold` (seuils de variation) dans le cadre d’une classification multi-classes du mouvement du S&P500.

Il permet de :
- Calculer les rendements à horizon glissant (`pred_horizon` jours),
- Appliquer une transformation en classes basée sur des seuils personnalisés (`threshold`),
- Étudier la distribution des classes générées et les corrélations avec les autres variables du jeu de données.

Ce script est principalement utilisé comme outil d’analyse préliminaire pour tester rapidement l’impact des paramètres sur la qualité des cibles (`target_multi`) avant l'entraînement d’un modèle.
"""


# Charger le fichier
file_path = 'csv_data/consolidated_data/normalized_complete_data.csv'
df = pd.read_csv("csv_data/consolidated_data/normalized_complete_data.csv", parse_dates=["Date"]).sort_values("Date")

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

change = (df["SP500_historical_data_Close"].shift(-7) - df["SP500_historical_data_Close"]) / df["SP500_historical_data_Close"]
df["target_multi"] = change.apply(label_change)
df = df.iloc[:-7]
# Calculer les returns à 7 jours
df['sp500_return_7d'] = (df['SP500_historical_data_Close'].shift(-7) - df['SP500_historical_data_Close']) / df['SP500_historical_data_Close']
df = df.iloc[:-7]  # retirer les dernières lignes sans return

# Afficher les quantiles
quantiles = df['sp500_return_7d'].quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
print(quantiles)


# Obtenir les informations pertinentes
info = {}

# Dates si colonne date existe
date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
if date_columns:
    date_col = date_columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    num_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else None
    info['date_column'] = date_col
    info['start_date'] = min_date
    info['end_date'] = max_date
    info['number_of_days'] = num_days
else:
    info['date_column'] = None
    info['start_date'] = None
    info['end_date'] = None
    info['number_of_days'] = None

# Autres informations
info['number_of_columns'] = df.shape[1]
info['number_of_rows'] = df.shape[0]
info['column_names'] = df.columns.tolist()
info['missing_values_per_column'] = df.isnull().sum().to_dict()
info['data_types'] = df.dtypes.astype(str).to_dict()
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
info['constant_columns'] = constant_cols

outliers = {}
for col in df.select_dtypes(include=np.number).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers[col] = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)].shape[0]
info['outliers_per_column'] = outliers


# Ajout : comptage des classes si 'target_multi' existe
if 'target_multi' in df.columns:
    class_counts = df['target_multi'].value_counts().sort_index().to_dict()
    info['target_multi_class_distribution'] = class_counts
else:
    info['target_multi_class_distribution'] = None

# Affichage
print(df.head())           # aperçu des premières lignes
print(df.info())           # résumé du DataFrame
print(df.describe())       # statistiques descriptives
print(info['target_multi_class_distribution'])   # informations supplémentaires


# Étapes supplémentaires : créer les targets multi-classes + stats + heatmap + corrélation avec target

# 1. Calcul du return à horizon 7 jours
df['sp500_return_7d'] = (df['SP500_historical_data_Close'].shift(-7) - df['SP500_historical_data_Close']) / df['SP500_historical_data_Close']

# 2. Création de la target multi-classes avec les anciens seuils pour commencer
def label_change(x):
    if x <= -0.043:
        return 0
    elif x <= -0.009:
        return 1
    elif x <= 0.017:
        return 2
    elif x <= 0.041:
        return 3
    else:
        return 4

df["target_multi"] = df["sp500_return_7d"].apply(label_change)
df = df.iloc[:-7]  # supprimer les dernières lignes sans target

# 3. Statistiques de base sur le return
quantiles = df['sp500_return_7d'].quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
class_distribution = df['target_multi'].value_counts().sort_index()

# 4. Corrélation avec la target
correlation_matrix = df.corr(numeric_only=True)
top_corr_features = correlation_matrix["sp500_return_7d"].abs().sort_values(ascending=False).head(20)

# 5. Visualisation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap globale
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title("Heatmap complète des corrélations")
heatmap_path = "/mnt/data/full_correlation_heatmap.png"
plt.tight_layout()
plt.savefig(heatmap_path)
plt.close()

# Corrélations avec la target uniquement
plt.figure(figsize=(8, 6))
top_corr = df[top_corr_features.index].corrwith(df['sp500_return_7d']).sort_values()
top_corr.plot(kind='barh', title="Corrélation avec sp500_return_7d")
corr_target_path = "/mnt/data/correlation_with_target.png"
plt.tight_layout()
plt.savefig(corr_target_path)
plt.close()