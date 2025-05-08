import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler

# Chargement des données
df = pd.read_csv("AAPL_historical_data.csv", parse_dates=["Date"], index_col="Date")

# Vérification des données
print(df.head())
print(df.info())

# Convertir la colonne Date (si elle existe) en datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# Calcul des indicateurs techniques
# Moyennes Mobiles
df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)

# RSI
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

# MACD
df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Bollinger Bands
df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Indicateurs basés sur le volume
df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()

# Variations de Prix
df['Daily_Change'] = df['Close'] - df['Open']
df['Price_Range'] = df['High'] - df['Low']

# Dividendes et Splits (Binaire)
df['Dividend_Indicator'] = (df['Dividends'] > 0).astype(int)
df['Stock_Split_Indicator'] = (df['Stock Splits'] > 0).astype(int)

# Création de la Target (Prédire le cours sur 7 jours)
df['Target'] = df['Close'].shift(-7) - df['Close']  # Variation de prix sur 7 jours
# Pour une prédiction binaire (hausse/baisse sur 7 jours)
df['Target_Binary'] = (df['Close'].shift(-7) > df['Close']).astype(int)

# Nettoyage des valeurs manquantes
df.dropna(inplace=True)

# Normalisation des données (MinMaxScaler)
scaler = MinMaxScaler()
features_to_scale = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'Avg_Volume_20', 'Daily_Change', 'Price_Range']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Résumé final des données
print(df.head())
print(f"Nombre de lignes après préparation : {len(df)}")