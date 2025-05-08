import talib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ticker = "AAPL"

# Fonctions définies précédemment
def ema_trend_meter(Close, len0=13, len1=21, len2=34, len3=55):
    ema0 = talib.EMA(Close, timeperiod=len0)
    ema1 = talib.EMA(Close, timeperiod=len1)
    ema2 = talib.EMA(Close, timeperiod=len2)
    ema3 = talib.EMA(Close, timeperiod=len3)
    return ema0, ema1, ema2, ema3

def stochastic_momentum_index(Low, High, Close, a=10, b=3):
    ll = Low.rolling(window=a).min()
    hh = High.rolling(window=a).max()
    diff = hh - ll
    rdiff = Close - (hh + ll) / 2

    avgrel = talib.EMA(talib.EMA(rdiff, timeperiod=b), timeperiod=b)
    avgdiff = talib.EMA(talib.EMA(diff, timeperiod=b), timeperiod=b)

    SMI = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    SMIsignal = talib.EMA(SMI, timeperiod=b)
    emasignal = talib.EMA(SMI, timeperiod=10)

    return SMI, SMIsignal, emasignal

def regression_analysis(arr):
    x = np.arange(len(arr))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, arr, rcond=None)[0]
    regression_line = slope * x + intercept
    mean_absolute_error = np.mean(np.abs(arr - regression_line))
    mean_value = np.mean(arr)
    percentage_mean_error = (mean_absolute_error / mean_value) * 100
    return slope, percentage_mean_error

def backtest_strategy(data, risk_per_trade=0.02, max_risk=0.08, atr_period=32, atr_factor=5):
    balance = 1000
    positions = []
    results = []
    transaction_dates = []
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_period)

    for i in range(len(data)):
        if i < atr_period:
            continue

        smi, smi_signal, ema_signal = stochastic_momentum_index(
            data['Low'], data['High'], data['Close']
        )
        slope, perc_error = regression_analysis(data['Close'][i - 80:i])

        if smi[i] < -40 and smi[i] > smi_signal[i] and slope > 0 and perc_error < 3:
            stop_loss = data['Low'].iloc[i] - (atr.iloc[i] * atr_factor)
            take_profit = data['Close'].iloc[i] + (atr.iloc[i] * atr_factor)
            risk = balance * risk_per_trade

            if stop_loss > 0:
                positions.append({
                    "entry_price": data['Close'].iloc[i],
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size": risk / abs(data['Close'].iloc[i] - stop_loss)
                })

        for position in positions:
            if data['Low'].iloc[i] <= position['stop_loss']:
                results.append(-risk_per_trade * balance)
                transaction_dates.append(data.index[i])
                positions.remove(position)
            elif data['High'].iloc[i] >= position['take_profit']:
                results.append(risk_per_trade * balance * 2)
                transaction_dates.append(data.index[i])
                positions.remove(position)

    balance += sum(results)

    return results, transaction_dates, balance

# Charger les données
data = pd.read_csv("GOOGL_historical_data.csv", parse_dates=["Date"], index_col="Date")

# Filtrer les données des 5 dernières années
data_5_years = data[data.index > (data.index.max() - pd.DateOffset(years=5))]

# Appliquer la stratégie de backtest
results, transaction_dates, final_balance = backtest_strategy(data_5_years)

# Calculer les résultats cumulés
cumulative_results = np.cumsum(results) + 1000

# Calculer la performance de la stratégie "buy and hold"
initial_price = data_5_years['Close'].iloc[0]
final_price = data_5_years['Close'].iloc[-1]
buy_and_hold_balance = 1000 * (final_price / initial_price)
buy_and_hold_performance = np.full(len(data_5_years), buy_and_hold_balance)

# Visualiser les résultats
plt.figure(figsize=(14, 7))
plt.plot(data_5_years.index, data_5_years['Close'], label=f'{ticker} Close Price')
plt.title(f'{ticker} Close Price')
plt.xlabel('Date')
plt.ylabel('Price')

plt.figure("strategie")
plt.plot(transaction_dates, cumulative_results, label='Strategy Performance', color='orange', marker='o', linestyle='-')
plt.title('Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Balance')

plt.legend()
plt.show()

print(f"Final Balance: {round(final_balance,0)}€")
print(f"Final Balance (Buy and Hold): {round(buy_and_hold_balance,0)}€")