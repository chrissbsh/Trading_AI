import pandas as pd
import numpy as np

df = pd.read_csv("AAPL_historical_data.csv", parse_dates=["Date"], index_col="Date")

def calculate_fibonacci(df, period):
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    df['Fibo_0.236'] = high - (high - low) * 0.236
    df['Fibo_0.382'] = high - (high - low) * 0.382
    df['Fibo_0.5'] = high - (high - low) * 0.5
    df['Fibo_0.618'] = high - (high - low) * 0.618
    df['Fibo_1.0'] = low
    return df

df_fibonacci = calculate_fibonacci(df, period=20)

print(df_fibonacci[['Fibo_0.236', 'Fibo_0.382', 'Fibo_0.5', 'Fibo_0.618', 'Fibo_1.0']].tail(20))


def calculate_ichimoku(df):
    df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)
    return df

df_ichimoku = calculate_ichimoku(df)

print(df_ichimoku[['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']].tail(20))

def calculate_atr(df, period):
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

df_atr = calculate_atr(df, period=14)

print(df_atr[['TR', 'ATR']].tail(20))


def calculate_volume_profile(df, bins=10):
    df['Price_Range'] = pd.cut(df['Close'], bins=bins)
    volume_profile = df.groupby('Price_Range', observed=False)['Volume'].sum()  # Explicitly set observed
    volume_profile = volume_profile / volume_profile.sum()  # Normalize
    return volume_profile

volume_profile = calculate_volume_profile(df, bins=20)


print(volume_profile)
