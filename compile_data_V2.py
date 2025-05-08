import pandas as pd
import numpy as np
from functools import reduce

# Fonctions pour calculs spécifiques (les mêmes que dans les versions précédentes)
def calculate_fibonacci(df, period):
    if 'High' in df.columns and 'Low' in df.columns:
        high = df['High'].rolling(window=period).max()
        low = df['Low'].rolling(window=period).min()
        df['Fibo_0.236'] = high - (high - low) * 0.236
        df['Fibo_0.382'] = high - (high - low) * 0.382
        df['Fibo_0.5'] = high - (high - low) * 0.5
        df['Fibo_0.618'] = high - (high - low) * 0.618
        df['Fibo_1.0'] = low
    return df

def calculate_ichimoku(df):
    if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
        df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou_Span'] = df['Close'].shift(-26)
    return df

def calculate_atr(df, period):
    if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_interest_rate_variations(df):
    if 'Value' in df.columns:
        df['Rate_Change'] = df['Value'].diff()
        df['Rate_SMA_20'] = df['Value'].rolling(window=20).mean()
    return df

def calculate_cpi_ppi_variations(df):
    if 'Value' in df.columns:
        df['Monthly_Change'] = df['Value'].diff()
        df['SMA_3'] = df['Value'].rolling(window=3).mean()
    return df

# Fonction principale pour traiter chaque fichier
def process_file(file_path, file_type, period_fibo=20, period_atr=14):
    # Traitement des fichiers de stocks
    if file_type == 'stocks':
        df = pd.read_csv(file_path)

        # Convertir la colonne Date
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)  # Supprimer le fuseau horaire
        df.set_index('Date', inplace=True)

        # Calculs des indicateurs techniques
        df = calculate_fibonacci(df, period=period_fibo)
        df = calculate_ichimoku(df)
        df = calculate_atr(df, period=period_atr)

    # Traitement des fichiers sans titre (CPI, PPI, etc.)
    elif file_type == 'indicator':
        df = pd.read_csv(file_path, header=None, names=['Date', 'Value'])

        # Convertir la colonne Date
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)  # Supprimer les informations de fuseau horaire
        df.set_index('Date', inplace=True)

        # Calculs spécifiques pour les indicateurs
        if "ppi" in file_path or "cpi" in file_path:
            df = calculate_cpi_ppi_variations(df)
        else:  # Taux d'intérêt ou autre
            df = calculate_interest_rate_variations(df)

    else:
        raise ValueError(f"Type de fichier inconnu pour : {file_path}")

    return df

# Définir les fichiers et leurs types
files_info = {
    "AAPL_historical_data.csv": "stocks",
    "interest_rates.csv": "indicator",
    "VIX_historical_data.csv": "stocks",
    "PPI.csv": "indicator",
    "CPI.csv": "indicator",
    "MSFT_historical_data.csv": "stocks",
    "SP500_historical_data.csv": "stocks",
    "NASDAQ_historical_data.csv": "stocks",
    "GOOGL_historical_data.csv": "stocks"
}

# Fonction pour renommer les colonnes avec un préfixe unique
def add_prefix_to_columns(df, prefix):
    return df.rename(columns=lambda x: f"{prefix}_{x}" if x != 'Date' else x)

# Traiter chaque fichier et renommer les colonnes
dataframes = {}
for file_path, file_type in files_info.items():
    name = file_path.split("/")[-1].replace(".csv", "")  # Extraire un nom simple
    df = process_file(file_path, file_type)
    dataframes[name] = add_prefix_to_columns(df, prefix=name.replace("_historical_data", ""))

# Fusionner tous les DataFrames sur les dates
combined_df = reduce(
    lambda left, right: pd.merge(left, right, on='Date', how='outer'), 
    dataframes.values()
)

# Remplir les NaN après fusion si nécessaire
combined_df.ffill(inplace=True)

# Sauvegarder dans un fichier CSV
combined_df.to_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/consolidated_data.csv")
print("Toutes les données ont été fusionnées et sauvegardées dans 'consolidated_data.csv'")
