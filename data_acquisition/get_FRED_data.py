import pandas as pd
from fredapi import Fred

# Configurez votre clé API pour FRED
fred_api_key = '9b2486a3e8f118de3f85fe741db8670c'
fred = Fred(api_key=fred_api_key)

# 2. Téléchargez le CPI et le PPI via FRED
cpi = fred.get_series('CPIAUCSL')  # CPI pour tous les consommateurs
ppi = fred.get_series('PPIACO')   # PPI pour tous les produits
cpi.to_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/cpi_data.csv")
ppi.to_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/ppi_data.csv")
print("Données CPI et PPI sauvegardées !")

# 3. Téléchargez les taux d'intérêt via FRED
interest_rates = fred.get_series('FEDFUNDS')  # Taux directeur de la Fed
interest_rates.to_csv(f"C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/interest_rates.csv")
print("Données de taux d'intérêt sauvegardées !")