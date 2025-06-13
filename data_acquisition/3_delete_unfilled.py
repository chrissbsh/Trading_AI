import pandas as pd
import os

def delete_unfilled(csv_file_path, output_file_path):
    # Lire le fichier CSV
    df = pd.read_csv(csv_file_path)

    # Supprimer les lignes où 'SP500_historical_data_Close' est NaN
    if 'SP500_historical_data_Close' in df.columns:
        df.dropna(subset=['SP500_historical_data_Close'], inplace=True)

    # Colonnes à remplir par forward-fill (si elles existent)
    columns_to_fill = [
        'transactions_summary_daily_Somme_Ventes',
        'transactions_summary_daily_Somme_Achats',
        'transactions_summary_daily_Nombre_Ventes',
        'transactions_summary_daily_Nombre_Achats'
    ]
    existing_columns_to_fill = [col for col in columns_to_fill if col in df.columns]
    df[existing_columns_to_fill] = df[existing_columns_to_fill].ffill()

    # Nettoyer les noms de colonnes (éliminer les espaces ou caractères invisibles)
    df.columns = df.columns.str.strip()

    # Mots à exclure et colonnes à préserver
    words_to_exclude = ['open', 'high', 'low', 'dividend', 'volume', 'stock', 'gains']
    exceptions = ['SP500_historical_data_Open', 'SP500_historical_data_High', 'SP500_historical_data_Low', 'SP500_historical_data_Volume']

    # Identifier les colonnes à supprimer
    columns_to_exclude = [
        col for col in df.columns
        if any(word in col.lower() for word in words_to_exclude) and col not in exceptions
    ]
    df.drop(columns=columns_to_exclude, inplace=True)

    # Supprimer les lignes contenant plus de 10% de NaN car ce sont des jours avec marché fermé
    df.dropna(thresh=len(df.columns) * 0.9, inplace=True)

    # Sauvegarder dans le fichier de sortie
    df.to_csv(output_file_path, index=False)

    return df

if __name__ == "__main__":
    directory = 'csv_data/consolidated_data/'
    input_file = os.path.join(directory, 'consolidated_data_periodic_filled.csv')
    output_file = os.path.join(directory, 'consolidated_data_filtered.csv')

    df = delete_unfilled(input_file, output_file)
