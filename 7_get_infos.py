import pandas as pd
import numpy as np

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