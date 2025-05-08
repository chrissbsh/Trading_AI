import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import IchimokuIndicator, PSARIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import uuid

directory = 'csv_data/consolidated_data/'

# Charger le fichier CSV
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df['High'] = df["SP500_historical_data_High"]
    df['Low'] = df["SP500_historical_data_Low"]
    df['Open'] = df["SP500_historical_data_Open"]
    df['Close'] = df["SP500_historical_data_Close"]
    df['Volume'] = df["SP500_historical_data_Volume"]
    return df

# Calculer les niveaux de Fibonacci (basé sur une fenêtre glissante)
def calculate_fibonacci_levels(df, window=20):
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
    df['Fib_High'] = df['Close'].rolling(window=window).max()
    df['Fib_Low'] = df['Close'].rolling(window=window).min()
    df['Fib_Range'] = df['Fib_High'] - df['Fib_Low']

    for level in fib_levels:
        df[f'Fib_Level_{int(level*100)}'] = df['Fib_Low'] + level * df['Fib_Range']

    return df

# Calculer tous les indicateurs
def calculate_indicators(df):
    # Add technical features based on SP500
    df['sp500_prev_close'] = df['SP500_historical_data_Close'].shift(1)
    df['sp500_return_1d'] = df['SP500_historical_data_Close'] / df['sp500_prev_close'] - 1

    # 1. Moyennes Mobiles (SMA, EMA)
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 2. RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # 3. MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - \
                  df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 4. Bandes de Bollinger
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()

    # 5. Niveaux de Fibonacci
    df = calculate_fibonacci_levels(df, window=20)

    # 6. Stochastique
    stoch = StochasticOscillator(high=df['High'], low=df['Low'],
                                close=df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # 7. Ichimoku
    ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()

    # 8. ATR
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()

    # 9. VWAP
    vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'],
                                      close=df['Close'], volume=df['Volume'], window=14)
    df['VWAP'] = vwap.volume_weighted_average_price()

    # 10. Parabolic SAR
    psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'])
    df['PSAR'] = psar.psar()

    # 11. Accumulation/Distribution
    df['AD'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
                (df['High'] - df['Low']) * df['Volume']
    df['AD'] = df['AD'].cumsum()

    # 12. OBV
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()

    # 13. DMI/ADX
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()

    # Supprimer les colonnes temporaires
    df.drop(['High', 'Low', 'Open', 'Volume', 'Fib_High', 'Fib_Low', 'Fib_Range'], axis=1, inplace=True)

    return df

# Sauvegarder les résultats
def save_results(df, output_path):
    df.to_csv(output_path)
    print("DataFrame shape: ", df.shape)
    print(f"Résultats sauvegardés dans {output_path}")

# Fonction principale
def main(input_file, output_file):
    # Charger les données
    df = load_data(input_file)

    # Calculer les indicateurs
    df = calculate_indicators(df)

    # Sauvegarder les résultats
    save_results(df, output_file)

if __name__ == "__main__":
    input_file = directory+"consolidated_data_filtered.csv"  # Remplacer par le chemin de votre fichier
    output_file = directory+"complete_data.csv"
    main(input_file, output_file)