import pandas as pd
from ta.trend import IchimokuIndicator, PSARIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

"""
Ce script enrichit le fichier de données `consolidated_data_filtered.csv` avec un ensemble complet 
d’indicateurs techniques financiers classiques et personnalisés, afin de générer un dataset final nommé `complete_data.csv`.

Fonctionnalités principales :
- Chargement des prix historiques du S&P500 (Open, High, Low, Close, Volume) et d’autres indicateurs macroéconomiques (or, dollar, taux).
- Calcul d’indicateurs techniques classiques :
  - Tendances : moyennes mobiles (SMA/EMA), MACD, Ichimoku, PSAR, ADX
  - Volatilité : Bollinger Bands, ATR, historique de volatilité
  - Momentum : RSI, Stochastic Oscillator
  - Volume : VWAP, OBV, Accumulation/Distribution
  - Niveaux de Fibonacci dynamiques (rolling window)
- Ajout d’indicateurs dérivés spécifiques :
  - Ratios SP500 / Or, Dollar, Taux US 10Y
  - Direction et tension du VIX
- Nettoyage final avec suppression des colonnes temporaires (OHLC, Range).

Ce fichier est utilisé pour construire une matrice d’apprentissage riche destinée à l’entraînement de modèles de prédiction financière.
"""

directory = 'csv_data/consolidated_data/'

# Charger les données
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df['High'] = df["SP500_historical_data_High"]
    df['Low'] = df["SP500_historical_data_Low"]
    df['Open'] = df["SP500_historical_data_Open"]
    df['Close'] = df["SP500_historical_data_Close"]
    df['Volume'] = df["SP500_historical_data_Volume"]
    return df

# Calcul des niveau de Fibonacci
def calculate_fibonacci_levels(df, window=20):
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
    df['Fib_High'] = df['Close'].rolling(window=window).max()
    df['Fib_Low'] = df['Close'].rolling(window=window).min()
    df['Fib_Range'] = df['Fib_High'] - df['Fib_Low']

    for level in fib_levels:
        df[f'Fib_Level_{int(level*100)}'] = df['Fib_Low'] + level * df['Fib_Range']

    return df

# Calcul de différents indicteurs techniques
def calculate_indicators(df):
    # Basique
    # df['sp500_prev_close'] = df['SP500_historical_data_Close'].shift(1)
    # df['sp500_return_1d'] = df['SP500_historical_data_Close'] / df['sp500_prev_close'] - 1

    # Ajout des indicateurs personnalisés
    df['std_21'] = df['sp500_return_1d'].rolling(21).std()
    df['hv_30']  = df['sp500_return_1d'].rolling(30).std()

    # Calcul de rapport x/SP500
    df['r_sp_gold'] = df['SP500_historical_data_Close'] / df['gold_historical_data_Close']
    df['r_sp_dxy'] = df['SP500_historical_data_Close'] / df['dollar_index_historical_data_Close']
    df['r_sp_bond'] = df['SP500_historical_data_Close'] / df['Market_yield_US_10_year_DGS10']

    vix = df['^VIX_historical_data_Close']
    df['vix_direction'] = vix.diff().fillna(0).gt(0).astype(int)
    df['vix_high'] = vix.gt(vix.rolling(63).median()).astype(int)

    # Indicateurs classiques
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()

    # df = calculate_fibonacci_levels(df, window=20)

    stoch = StochasticOscillator(high=df['High'], low=df['Low'],
                                 close=df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()

    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()

    vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'],
                                      close=df['Close'], volume=df['Volume'], window=14)
    df['VWAP'] = vwap.volume_weighted_average_price()

    psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'])
    df['PSAR'] = psar.psar()

    df['AD'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
               (df['High'] - df['Low']) * df['Volume']
    df['AD'] = df['AD'].cumsum()

    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()

    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()

    df.drop(['Close', 'High', 'Low', 'Open', 'Volume'], axis=1, inplace=True)

    # ── Features dérivées macro ──────────────────────────────────────────
    # Spread 2Y-10Y (indicateur de récession le plus fiable)
    if 'DGS2_DGS2' in df.columns and 'Market_yield_US_10_year_DGS10' in df.columns:
        df['yield_spread_2y10y'] = df['Market_yield_US_10_year_DGS10'] - df['DGS2_DGS2']

    # Momentum WTI sur 21 jours
    if 'CL=F_historical_data_Close' in df.columns:
        df['wti_momentum_21d'] = df['CL=F_historical_data_Close'].pct_change(21)

    # Rotation sectorielle : tech vs finance, énergie vs santé
    if 'XLK_historical_data_Close' in df.columns and 'XLF_historical_data_Close' in df.columns:
        df['sector_rotation_tech_fin'] = df['XLK_historical_data_Close'] / df['XLF_historical_data_Close']
    if 'XLE_historical_data_Close' in df.columns and 'XLV_historical_data_Close' in df.columns:
        df['sector_rotation_energy_health'] = df['XLE_historical_data_Close'] / df['XLV_historical_data_Close']

    # Accélération M2 sur 3 mois
    if 'M2SL_M2SL' in df.columns:
        df['m2_acceleration_3m'] = df['M2SL_M2SL'].pct_change(63)

    # Variation du credit spread sur 5 jours
    if 'BAA10Y_BAA10Y' in df.columns:
        df['credit_spread_change_5d'] = df['BAA10Y_BAA10Y'].diff(5)

    # Ratio vol implicite / vol réalisée (VIX vs HV30)
    if '^VIX_historical_data_Close' in df.columns and 'hv_30' in df.columns:
        df['vix_hv_ratio'] = df['^VIX_historical_data_Close'] / (df['hv_30'] * 100 + 1e-8)

    # Momentum ETFs sectoriels sur 21 jours
    for etf in ['XLK', 'XLF', 'XLE', 'XLV', 'XLI']:
        col = f'{etf}_historical_data_Close'
        if col in df.columns:
            df[f'{etf}_momentum_21d'] = df[col].pct_change(21)

    return df

def save_results(df, output_path):
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Résultats sauvegardés dans {output_path}, shape: {df.shape}")

def main(input_file, output_file):
    df = load_data(input_file)
    df = calculate_indicators(df)
    save_results(df, output_file)

if __name__ == "__main__":
    input_file = directory + "consolidated_data_filtered.csv"
    output_file = directory + "complete_data.csv"
    main(input_file, output_file)