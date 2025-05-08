import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

directory = 'csv_data/indicators/'

# Chargement des données
df = pd.read_csv(directory+'consolidated_data.csv', parse_dates=True, index_col=0)

# Aperçu des données
print(df.info())

# Suppression des colonnes avec trop de valeurs manquantes (>50%)
df = df.loc[:, df.isnull().mean() < 0.5]

# # Imputation des valeurs manquantes par interpolation
# df = df.interpolate(method='time')

# Vérification des valeurs manquantes restantes
df = df.dropna()

# Normalisation MinMax
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# Sélection de features - 1 : Filtrage par faible variance
selector_var = VarianceThreshold(threshold=0.01)
var_filtered_data = selector_var.fit_transform(scaled_df)
var_filtered_df = pd.DataFrame(var_filtered_data, columns=scaled_df.columns[selector_var.get_support()], index=scaled_df.index)

# Sélection de features - 2 : Importance via RandomForest
X = var_filtered_df.copy()
# Suppose une target fictive si tu n'en as pas : la moyenne glissante
y = X.mean(axis=1).rolling(window=3).mean().dropna()
X = X.loc[y.index]

# Random Forest pour importance des features
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Sélection via Random Forest
model_selector = SelectFromModel(rf, prefit=True, threshold="median")
X_selected_rf = model_selector.transform(X)

# Sélection de features - 3 : RFE avec SVM
svm = SVR(kernel='linear')
rfe_selector = RFE(estimator=svm, n_features_to_select=10, step=1)
rfe_selector = rfe_selector.fit(X, y)
X_selected_rfe = rfe_selector.transform(X)

# Plot des importances RandomForest
plt.figure(figsize=(12, 6))
sns.barplot(x=[X.columns[i] for i in indices], y=importances[indices])
plt.xticks(rotation=90)
plt.title("Importance des Features (Random Forest)")
plt.show()

# Résumé des données prêtes pour le réseau de neurones
final_features = X.columns[model_selector.get_support()]
final_df = pd.DataFrame(X_selected_rf, columns=final_features, index=X.index)

print("Features finales sélectionnées :")
print(final_features)

# Exemple : préparation pour réseau de neurones
X_train, X_test, y_train, y_test = train_test_split(final_df, y, test_size=0.2, random_state=42)

# Tu peux maintenant utiliser X_train et y_train dans un modèle de deep learning.