import yfinance as yf
import pandas as pd

# Télécharger toutes les données historiques
ticker = "^VIX"
stock = yf.Ticker(ticker)

# Obtenir les données historiques (maximales disponibles)
historical_data = stock.history(period="max")

# Sauvegarder les données dans un fichier CSV pour analyse future
#historical_data.to_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/csv_data/indicators/{ticker}_historical_data.csv")
historical_data.to_csv(f'/Users/clementberthoud/Library/CloudStorage/OneDrive-CentraleSupelec/Fichiers de Christophe Boshra (Student at CentraleSupelec) - Trading_AI/csv_data/indicators/{ticker}_historical_data.csv')

# Afficher un aperçu des données
print(historical_data.head())
print(f"Total de données récupérées : {len(historical_data)} lignes")