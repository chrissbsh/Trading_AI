import pandas as pd

# Charger les données consolidées
file_path = "C:/Users/chris/OneDrive - CentraleSupelec/Bureau/Trading_AI/consolidated_data.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Aperçu des données
print(df.head())
print(df.info())

# Statistiques descriptives
print(df.describe())

import matplotlib.pyplot as plt

# Visualisation des cours d'Apple, Google et Microsoft
df[['AAPL_Close', 'GOOGL_Close', 'MSFT_Close']].plot(figsize=(12, 6), title="Cours Clôture des Actions")
plt.xlabel('Date')
plt.ylabel('Prix')
plt.show()


import seaborn as sns
import numpy as np

# Sélection des colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()


# Calcul de la volatilité (par exemple, AAPL_Close)
df['AAPL_Returns'] = df['AAPL_Close'].pct_change()
df['AAPL_Volatility'] = df['AAPL_Returns'].rolling(window=20).std()

# Visualisation de la volatilité
df['AAPL_Volatility'].plot(figsize=(12, 6), title="Volatilité de Apple")
plt.xlabel('Date')
plt.ylabel('Volatilité (Rolling 20 jours)')
plt.show()


# Boxplot pour détecter les outliers sur les rendements
sns.boxplot(x=df['AAPL_Returns'].dropna())
plt.title("Distribution des Rendements d'Apple")
plt.show()


# Vérification des NaN
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])


# Visualisation de l'inflation (CPI) et du cours d'Apple
df[['CPI_Value', 'AAPL_Close']].plot(subplots=True, figsize=(12, 8), title="CPI vs Cours d'Apple")
plt.xlabel('Date')
plt.show()


# Taux d'intérêt et volatilité du marché
df[['interest_rates_Value', 'AAPL_Volatility']].plot(subplots=True, figsize=(12, 8), title="Taux d'Intérêt vs Volatilité d'Apple")
plt.xlabel('Date')
plt.show()

# Extraire les corrélations fortes
strong_corrs = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
print(strong_corrs)


subset = ['AAPL_Close', 'CPI_Value', 'PPI_Value', 'interest_rates_Value']
sns.heatmap(df[subset].corr(), annot=True, cmap='coolwarm')
plt.title("Zoom sur Corrélations Clés")
plt.show()