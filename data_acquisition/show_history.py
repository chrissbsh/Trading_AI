import matplotlib.pyplot as plt
import pandas as pd

"""
Ce script permet de visualiser l'évolution du prix de clôture d'un actif financier donné 
(tel qu'une action ou une cryptomonnaie) à partir de son fichier CSV préalablement téléchargé.

Fonctionnalités :
- Chargement des données historiques depuis le fichier `csv_data/indicators/{ticker}_historical_data.csv`.
- Affichage d’un graphique de la variable `Close` (prix de clôture) avec l’axe des dates.

Ce script est utile pour une première exploration visuelle de l’évolution temporelle d’un actif.
"""

# ticker = "AAPL"
# ticker = "GOOGL"
# ticker = "ETH-USD"
ticker = "SOL-USD"

historical_data = pd.read_csv(f"csv_data/indicators/{ticker}_historical_data.csv", index_col=0, parse_dates=True)

# Graphique du prix de clôture
historical_data['Close'].plot(title=f"Prix de clôture de {ticker}", xlabel="Date", ylabel="Prix (USD)")
plt.show()
