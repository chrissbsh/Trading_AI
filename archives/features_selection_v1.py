import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

directory = 'csv_data/indicators/'

# 1. Chargement du fichier consolidé
df = pd.read_csv(directory+'consolidated_data.csv', parse_dates=True, index_col=0)

# 2. Sélection des colonnes numériques uniquement (évitant les colonnes date ou objets)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 3. Création de features temporelles (lags et statistiques roulantes) sur les colonnes numériques
def create_time_features(df, cols, lags=[1,2,3], windows=[3,7,14]):
    feat_df = pd.DataFrame(index=df.index)
    for col in cols:
        for lag in lags:
            feat_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        for window in windows:
            feat_df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
            feat_df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
    return feat_df

feat_df = create_time_features(df, numeric_cols)
feat_df.dropna(inplace=True)

# 4. Sélection non-supervisée
vt = VarianceThreshold(threshold=0.01)
vt.fit(feat_df)
feat_vt = feat_df.loc[:, vt.get_support()]

corr_matrix = feat_vt.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
feat_uncorr = feat_vt.drop(columns=to_drop)

# 5. Résultat final
feat_final = feat_uncorr
feat_final.to_csv(directory+'selected_features.csv')

# Aperçu
print(f"Colonnes originales numériques utilisées : {len(numeric_cols)}")
print(f"Features initiales créées : {feat_df.shape}")
print(f"Après VarianceThreshold : {feat_vt.shape}")
print(f"Colonnes à supprimer (corr > 0.95) : {len(to_drop)}")
print(f"Features finales sélectionnées : {feat_final.shape}")
print("Aperçu des 5 premières features :")
print(feat_final.iloc[:, :5].head())