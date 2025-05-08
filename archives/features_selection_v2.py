import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

directory = 'csv_data/indicators/'

def preprocess_data(
    file_path: str,
    missing_strategy: str = 'median',
    variance_thresh: float = 0.01,
    corr_thresh: float = 0.90,
    scale_method: str = 'standard',
    apply_pca: bool = True,
    pca_components = 0.95
):
    # 1. Chargement et sélection des colonnes numériques
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 2. Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy=missing_strategy)
    numeric_imputed = pd.DataFrame(
        imputer.fit_transform(numeric_df),
        columns=numeric_df.columns
    )
    
    # 3. Suppression des variables à faible variance
    vt = VarianceThreshold(threshold=variance_thresh)
    vt_array = vt.fit_transform(numeric_imputed)
    sel_features = numeric_imputed.columns[vt.get_support()]
    df_vt = pd.DataFrame(vt_array, columns=sel_features)
    
    # 4. Élimination des variables très corrélées
    corr_matrix = df_vt.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    df_uncorr = df_vt.drop(columns=to_drop)
    
    # 5. Normalisation
    scaler = StandardScaler() if scale_method == 'standard' else MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_uncorr),
        columns=df_uncorr.columns
    )
    
    # 6. Réduction de dimension (PCA)
    if apply_pca:
        pca = PCA(n_components=pca_components)
        pca_transformed = pca.fit_transform(df_scaled)
        df_processed = pd.DataFrame(
            pca_transformed,
            columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])]
        )
    else:
        df_processed = df_scaled
    
    return df, df_processed

# Exécution de l'exemple
raw_df, processed_df = preprocess_data(
    directory+'consolidated_data.csv',
    missing_strategy='median',
    variance_thresh=0.01,
    corr_thresh=0.90,
    scale_method='standard',
    apply_pca=True,
    pca_components=0.95
)

print(f"Raw data shape: {raw_df.shape}")
print(f"Processed data shape: {processed_df.shape}")
print("Aperçu des premières lignes du jeu de données prétraité :")
print(processed_df.head())

# to csv
processed_df.to_csv(directory+'processed_features.csv', index=False)