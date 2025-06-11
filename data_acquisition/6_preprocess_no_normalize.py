import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

directory = 'csv_data/consolidated_data/'

# Charger le CSV
df = pd.read_csv(directory + 'complete_data.csv')
print(f"Shape initial : {df.shape}")

# Conversion date + tri
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Suppression colonnes constantes
before = df.shape[1]
df = df.loc[:, df.nunique(dropna=True) > 1]
after = df.shape[1]
print(f"Colonnes constantes supprimées : {before - after}")

# Copie pour normalisation
df_normalized = df.copy()

binary_cols = ['vix_direction', 'vix_high', 'macro_regime']

# # Normaliser toutes colonnes numériques sauf Date et SP500 Close
# for column in df.columns:
#     if column in ['Date', 'SP500_historical_data_Close'] + binary_cols:
#         continue
#     if pd.api.types.is_numeric_dtype(df[column]):
#         df[column] = df[column].replace([np.inf, -np.inf], np.nan)
#         median_value = df[column].median()
#         df_normalized[column] = df[column].fillna(median_value)
#         df_normalized[column] = StandardScaler().fit_transform(df_normalized[[column]])
#     else:
#         df_normalized[column] = df[column]

# Supprimer colonnes full zéro
before = df_normalized.shape[1]
df_normalized = df_normalized.loc[:, (df_normalized != 0).any(axis=0)]
after = df_normalized.shape[1]
print(f"Colonnes full zéro supprimées : {before - after}")

# afficher nombre valeurs manquantes par colonne
missing_counts = df_normalized.isnull().sum()
print("Nombre de valeurs manquantes par colonne :")
print(missing_counts[missing_counts > 0])

# supprimer les lignes avec des valeurs manquantes
df_normalized = df_normalized.dropna()

# Réorganisation colonnes : Date + SP500 + sp500_* + reste
cols = list(df_normalized.columns)
sp500_cols = [col for col in cols if col.startswith('sp500_')]
other_cols = [col for col in cols if col not in ['Date', 'SP500_historical_data_Close'] + sp500_cols]
new_order = ['Date', 'SP500_historical_data_Close'] + sp500_cols + other_cols
df_normalized = df_normalized[new_order]

# Sauvegarde
df_normalized.to_csv(directory + 'final_complete_data.csv', index=False)
print(f"Shape final : {df_normalized.shape}")
print("Fichier normalisé sauvegardé : final_complete_data.csv")
print(f"Colonnes finales : {len(df_normalized.columns)} features")