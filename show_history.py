import matplotlib.pyplot as plt
import pandas as pd

# ticker = "AAPL"
# ticker = "GOOGL"
# ticker = "ETH-USD"
ticker = "SOL-USD"

historical_data = pd.read_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/{ticker}_historical_data.csv", index_col=0, parse_dates=True)

# Graphique du prix de clôture
historical_data['Close'].plot(title=f"Prix de clôture de {ticker}", xlabel="Date", ylabel="Prix (USD)")
plt.show()
