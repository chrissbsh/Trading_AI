# Ajout de données 
import pandas as pd

# 1. Lire le fichier CSV
df = pd.read_csv('/Users/clementberthoud/Library/CloudStorage/OneDrive-CentraleSupelec/Fichiers de Christophe Boshra (Student at CentraleSupelec) - Trading_AI/Clément/Données_Diff.csv')

# 2. Calcul du résultat
df["Dif_High"] = (df["SP500_historical_data_High"] - df["SP500_historical_data_Open"])

# 3. (Optionnel) Sauvegarder dans un nouveau fichieré
df.to_csv('/Users/clementberthoud/Library/CloudStorage/OneDrive-CentraleSupelec/Fichiers de Christophe Boshra (Student at CentraleSupelec) - Trading_AI/Clément/Données_Diff.csv', index=False)
