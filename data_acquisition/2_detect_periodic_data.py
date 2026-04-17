import pandas as pd
import numpy as np

"""
Ce script détecte automatiquement les colonnes contenant des données **périodiques** dans un fichier CSV 
(`consolidated_data.csv`) en analysant l’espacement des dates non nulles. Il distingue :

- Les **colonnes quotidiennes** (avec des données presque tous les jours),
- Les **colonnes périodiques** (publiées avec un délai supérieur à un seuil défini, ex. indicateurs mensuels).

Il remplit ensuite les valeurs manquantes des colonnes périodiques par propagation de la dernière valeur connue (méthode `ffill`),
afin d’avoir un dataset complet utilisable pour des modèles temporels.

Fonctionnalités principales :
- Détection automatique du type de fréquence par colonne numérique,
- Exclusion manuelle de certaines colonnes du traitement si nécessaire,
- Génération d’un nouveau fichier CSV enrichi : `consolidated_data_periodic_filled.csv`.

Ce fichier est utilisé pour prétraiter les données économiques en vue d’un apprentissage séquentiel (ex. LSTM).
"""

# Détecter les données périodiques
def detect_periodic_columns(csv_file, date_column='Date', threshold=2):
    df = pd.read_csv(csv_file)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    
    daily_columns = []
    periodic_columns = []
    
    for col in numeric_columns:
        non_null_dates = df[df[col].notnull()][date_column]
        
        if len(non_null_dates) < 2:
            periodic_columns.append(col)
            continue
        
        deltas = non_null_dates.diff().dt.days[1:]
        
        if len(deltas) == 0:
            periodic_columns.append(col)
            continue
        
        mean_delta = deltas.mean()
        
        if mean_delta <= threshold:
            daily_columns.append(col)
        else:
            periodic_columns.append(col)
    
    return df, daily_columns, periodic_columns

# Compléter les lignes manquantes pour les données périodiques (copie la dernière valeur jusqu'à changement)
def fill_periodic_values(df, periodic_columns, date_column='Date'):
    # On trie d'abord par date pour que le ffill fonctionne correctement
    df = df.sort_values(date_column)
    
    # Remplir les valeurs périodiques avec la dernière valeur connue
    df[periodic_columns] = df[periodic_columns].ffill()
    return df


if __name__ == "__main__":
    directory = 'csv_data/consolidated_data/'
    csv_file_path = directory + 'consolidated_data.csv'
    
    try:
        df, daily, periodic = detect_periodic_columns(csv_file_path)
        
        print("Colonnes quotidiennes :")
        print(daily)
        
        print("\nColonnes périodiques (avant remplissage) :")
        print(periodic)

        # Colonnes à exclure manuellement des colonnes périodiques
        manual_exclusions = ['transactions_summary_daily_Somme_Ventes', "transactions_summary_daily_Somme_Achats", "transactions_summary_daily_Nombre_Ventes", "transactions_summary_daily_Nombre_Achats"]

        # On filtre la liste periodic
        periodic = [col for col in periodic if col not in manual_exclusions]

        # Appliquer le remplissage
        df_filled = fill_periodic_values(df, periodic)
        
        # Sauvegarde facultative
        df_filled.to_csv(directory + 'consolidated_data_periodic_filled.csv', index=False)
        print("\nFichier complété enregistré")
        
    except FileNotFoundError:
        print(f"Le fichier {csv_file_path} n'a pas été trouvé. Veuillez vérifier le chemin.")
