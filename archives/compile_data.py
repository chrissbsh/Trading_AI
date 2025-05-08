import pandas as pd
import numpy as np
from functools import reduce

# Fonctions pour calculs spécifiques
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
    if 'Rate' in df.columns:
        df['Rate_Change'] = df['Rate'].diff()
        df['Rate_SMA_20'] = df['Rate'].rolling(window=20).mean()
    return df

def calculate_cpi_ppi_variations(df):
    if 'Value' in df.columns:
        df['Monthly_Change'] = df['Value'].diff()
        df['SMA_3'] = df['Value'].rolling(window=3).mean()
    return df


import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
STOCK_FILES = ['AAPL_historical_data.csv', 'GOOGL_historical_data.csv', 'MSFT_historical_data.csv', 'NASDAQ_historical_data.csv', 'SP500_historical_data.csv', 'VIX_historical_data.csv']  # Liste des fichiers CSV pour les actions
INDICATOR_FILES = ['interest_rates.csv', 'cpi_data.csv', 'ppi_data.csv']  # Liste des fichiers CSV pour les indicateurs économiques
OUTPUT_FILE = 'ml_dataset.csv'  # Nom du fichier de sortie pour ML
TARGET_COLUMN = 'Close'  # La colonne cible pour le ML (par exemple, prix de clôture des actions)

# === FONCTIONS POUR CALCULS SPÉCIFIQUES ===

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
    if 'Rate' in df.columns:
        df['Rate_Change'] = df['Rate'].diff()
        df['Rate_SMA_20'] = df['Rate'].rolling(window=20).mean()
    return df

def calculate_cpi_ppi_variations(df):
    if 'Value' in df.columns:
        df['Monthly_Change'] = df['Value'].diff()
        df['SMA_3'] = df['Value'].rolling(window=3).mean()
    return df

# === FONCTIONS UTILITAIRES ===

def load_stock_data(file_path):
    """
    Charge les données des actions, applique les calculs techniques et retourne le DataFrame.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = calculate_fibonacci(df, period=14)
    df = calculate_ichimoku(df)
    df = calculate_atr(df, period=14)
    return df

def load_indicator_data(file_path):
    """
    Charge les données des indicateurs économiques (CPI, PPI, etc.), calcule les variations et retourne le DataFrame.
    """
    df = pd.read_csv(file_path, header=None, names=['Date', 'Value'])
    df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Conversion de la colonne Date
    indicator_name = os.path.splitext(os.path.basename(file_path))[0]  # Nom de l'indicateur
    df.rename(columns={'Value': indicator_name}, inplace=True)  # Renomme la colonne 'Value'
    if 'cpi' in indicator_name.lower() or 'ppi' in indicator_name.lower():
        df = calculate_cpi_ppi_variations(df)
    return df

def merge_dataframes(stock_data, indicator_data_list):
    """
    Fusionne les données des actions avec les indicateurs économiques sur la colonne 'Date'.
    """
    df = stock_data
    for indicator_data in indicator_data_list:
        df = pd.merge_asof(df.sort_values('Date'), indicator_data.sort_values('Date'), on='Date', direction='backward')
    return df

# === TRAITEMENT PRINCIPAL ===

# Charger les données des actions
stock_data_list = [load_stock_data(file) for file in STOCK_FILES]
combined_stocks = pd.concat(stock_data_list, ignore_index=True)

# Charger les données des indicateurs économiques
indicator_data_list = [load_indicator_data(file) for file in INDICATOR_FILES]
combined_indicator= pd.concat(indicator_data_list, ignore_index=True)

print("=== Actions ===")
print(combined_stocks.tail())

print("=== Indicateurs économiques ===")
print(combined_indicator.tail())

# Fusionner les données
final_dataset = merge_dataframes(combined_stocks, combined_indicator)

# Nettoyage final
final_dataset.dropna(inplace=True)  # Supprime les lignes avec des NaN (nécessaire après les calculs)
final_dataset.reset_index(drop=True, inplace=True)

# === SAUVEGARDE ===
final_dataset.to_csv(OUTPUT_FILE, index=False)
print(f"Fichier final prêt pour ML : {OUTPUT_FILE}")