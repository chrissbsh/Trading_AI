import yfinance as yf

"""
Ce script télécharge automatiquement l'historique complet de l'indice VIX (`^VIX`) ou autre via l'API Yahoo Finance 
en utilisant la librairie `yfinance`. Il enregistre les données journalières disponibles dans un fichier CSV 
dans le dossier des indicateurs (`csv_data/indicators/`).

Fonctionnalités :
- Récupération de toutes les données disponibles (depuis la première date jusqu'à aujourd'hui).
- Export au format CSV avec `Date` en index pour un usage futur dans des pipelines de consolidation.
- Aperçu rapide des données récupérées et du nombre total de lignes.

Ce fichier est utilisé comme indicateur de volatilité pour enrichir les données financières utilisées en modélisation.
"""

# Télécharger toutes les données historiques
ticker = "^VIX"
stock = yf.Ticker(ticker)

# Obtenir les données historiques (maximales disponibles)
# historical_data = stock.history(start="2000-01-01", interval="1d")
historical_data = stock.history(period="max")

# Sauvegarder les données dans un fichier CSV pour analyse future
csv_path = f"csv_data/indicators/{ticker}_historical_data.csv"
historical_data.to_csv(csv_path, index=True)

# Afficher un aperçu des données
print(historical_data.head())
print(f"Total de données récupérées : {len(historical_data)} lignes")