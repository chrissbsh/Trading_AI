import requests
import yfinance as yf

# Endpoint pour les cryptomonnaies
url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 10,
    "page": 1,
    "sparkline": False
}

response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    for coin in data:
        print(f"{coin['name']}: {coin['current_price']} USD")
else:
    print("Erreur :", response.status_code, response.text)

# Liste des symboles des 10 plus grandes entreprises (exemple)
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK.B", "NVDA", "JPM", "JNJ", "V"]

# Récupérer les données des entreprises
for symbol in symbols:
    stock = yf.Ticker(symbol)
    info = stock.info
    print(f"{info['shortName']} ({symbol}): {info['regularMarketPrice']} USD")
