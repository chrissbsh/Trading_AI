import os
import glob
import pandas as pd

def parse_csv(file_path):
    df = pd.read_csv(file_path)

    # Identifier la colonne de date
    date_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "observation", "peak", "trough"]):
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"Aucune colonne de date trouvée dans {file_path}")

    # Parser la date et enlever l'heure (niveau jour)
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.date

    # Supprimer les colonnes non numériques
    for col in df.columns:
        if col != date_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Supprimer les lignes sans date valide
    df.dropna(subset=[date_col], inplace=True)

    # Regrouper par date et agréger par moyenne si doublon dans le même fichier
    df = df.groupby(date_col).mean(numeric_only=True)

    # Renommer les colonnes
    base = os.path.splitext(os.path.basename(file_path))[0]
    df.rename(columns=lambda c: f"{base}_{c}", inplace=True)

    return df

def combine_csvs(folder_path, pattern="*.csv"):
    file_paths = glob.glob(os.path.join(folder_path, pattern))
    if not file_paths:
        raise FileNotFoundError("Aucun fichier CSV trouvé dans le dossier.")

    combined = None

    for path in file_paths:
        df = parse_csv(path)

        if combined is None:
            combined = df
        else:
            # Avant de fusionner, vérifier s'il y a des conflits
            common_dates = combined.index.intersection(df.index)
            for date in common_dates:
                for col in combined.columns.intersection(df.columns):
                    val1 = combined.at[date, col]
                    val2 = df.at[date, col]
                    if pd.notna(val1) and pd.notna(val2) and val1 != val2:
                        raise ValueError(f"Conflit détecté pour la date {date} dans la colonne {col} : {val1} != {val2}")
            combined = combined.combine_first(df)

    return combined

if __name__ == "__main__":
    folder_in = "csv_data/indicators"
    folder_out = "csv_data/consolidated_data"

    combined_df = combine_csvs(folder_in)

    # Convertir l'index en DataFrame avec la date
    df_out = combined_df.reset_index()
    df_out.rename(columns={"index": "Date"}, inplace=True)
    df_out["Date"] = pd.to_datetime(df_out["Date"]).dt.strftime("%Y-%m-%d")

    # Filtrer après 2000-01-01
    df_out = df_out[df_out["Date"] >= "2000-01-01"]

    # Sauvegarde
    os.makedirs(folder_out, exist_ok=True)
    df_out.to_csv(os.path.join(folder_out, "consolidated_data.csv"), index=False)