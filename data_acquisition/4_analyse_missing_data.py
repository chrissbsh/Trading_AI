import pandas as pd
import numpy as np

"""
Ce script permet d'analyser de manière détaillée les données manquantes dans un fichier CSV (par défaut `consolidated_data_filtered.csv`).

Fonctionnalités principales :
- Affiche, pour chaque colonne :
    - le **nombre** de valeurs manquantes,
    - le **pourcentage** de valeurs manquantes par rapport au total.
- Donne une analyse par **ligne** :
    - nombre de lignes contenant des NaN,
    - pourcentage global de lignes incomplètes,
    - aperçu des lignes les plus touchées par des valeurs manquantes.

Ce script est utilisé comme outil de diagnostic pour guider les étapes de prétraitement ou d'imputation sur les données consolidées.
"""

# Analyser/visualisez l'état des données manquantes
def analyze_missing_data(csv_file):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)
    
    # 1. Analyse des données manquantes par colonne
    print("\n=== Données manquantes par colonne ===")
    missing_by_column = df.isnull().sum()
    total_rows = len(df)
    
    # Pourcentage de valeurs manquantes par colonne
    missing_percent = (missing_by_column / total_rows * 100).round(2)
    
    # Afficher les résultats par colonne
    print("\nNombre de valeurs manquantes par colonne :")
    print(missing_by_column)
    print("\nPourcentage de valeurs manquantes par colonne :")
    print(missing_percent)
    
    # 2. Analyse des données manquantes par ligne
    print("\n=== Données manquantes par ligne ===")
    missing_by_row = df.isnull().sum(axis=1)
    
    # Statistiques sur les lignes
    print(f"\nNombre total de lignes : {total_rows}")
    print(f"Lignes avec au moins une valeur manquante : {(missing_by_row > 0).sum()}")
    print(f"Pourcentage de lignes avec des valeurs manquantes : {((missing_by_row > 0).sum() / total_rows * 100):.2f}%")
    
    # Afficher un aperçu des lignes avec le plus de valeurs manquantes
    print("\nTop 5 des lignes avec le plus de valeurs manquantes :")
    print(missing_by_row.sort_values(ascending=False).head())
    
    return missing_by_column, missing_by_row

# Exemple d'utilisation
if __name__ == "__main__":
    directory = 'csv_data/consolidated_data/'
    csv_file_path = directory+'consolidated_data_filtered.csv'
    missing_col, missing_row = analyze_missing_data(csv_file_path)