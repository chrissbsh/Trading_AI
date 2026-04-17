import pandas as pd

"""
Ce script effectue un nettoyage final et une normalisation du fichier `complete_data.csv` 
pour produire un dataset prêt à être utilisé en apprentissage automatique (`final_complete_data.csv`).

Étapes réalisées :
- Chargement et tri des données par date.
- Suppression des colonnes constantes (valeur unique) et des colonnes contenant uniquement des zéros.
- Affichage et suppression des lignes contenant des valeurs manquantes.
- Réorganisation logique des colonnes : d’abord `Date`, puis `SP500_historical_data_Close`, ensuite toutes les features `sp500_*`, puis les autres.
- Export du fichier nettoyé dans `final_complete_data.csv`.

Ce fichier constitue la version finale consolidée, filtrée et structurée du dataset, utilisable directement pour la modélisation.
"""

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

# Supprimer colonnes full zéro
before = df_normalized.shape[1]
df_normalized = df_normalized.loc[:, (df_normalized != 0).any(axis=0)]
after = df_normalized.shape[1]
print(f"Colonnes full zéro supprimées : {before - after}")

# Afficher nombre valeurs manquantes par colonne
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