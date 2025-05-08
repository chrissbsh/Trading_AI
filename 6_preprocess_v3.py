import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

directory = 'csv_data/consolidated_data/'

# Load the CSV file
df = pd.read_csv(directory + 'complete_data.csv')

print("df shape: ", df.shape)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Remove columns with only one unique value or no value
df = df.loc[:, df.nunique(dropna=True) > 1]

# Sort by date just to ensure chronological order
df = df.sort_values(by='Date')

# Function to determine if a column is numeric
def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series)

# Initialize the scaler for standardization
scaler = StandardScaler()

# Create a copy of the DataFrame to store normalized data
df_normalized = df.copy()

for column in df.columns:
    if column == 'Date' or column.startswith('SP500_historical_data_Close'):
        # Skip normalization for 'Date' and SP500-related technical indicators
        continue
    elif is_numeric_column(df[column]):
        # Replace inf/-inf with NaN
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        # Replace NaN with the median
        median_value = df[column].median()
        df_normalized[column] = df[column].fillna(median_value)
        # Apply StandardScaler
        df_normalized[column] = scaler.fit_transform(df_normalized[[column]])
    else:
        df_normalized[column] = df[column]

# Remove columns where all values are zero
df_normalized = df_normalized.loc[:, (df_normalized != 0).any(axis=0)]

# Move the 'SP500_historical_data_Close' column and its relatives after 'Date'
sp500_cols = [col for col in df_normalized.columns if col.startswith('sp500_')]
cols = list(df_normalized.columns)
for col in sp500_cols:
    cols.remove(col)
    cols.insert(1, col)
df_normalized = df_normalized[cols]

# Ensure 'SP500_historical_data_Close' is placed right after 'Date'
cols = list(df_normalized.columns)
cols.remove('SP500_historical_data_Close')
cols.insert(1, 'SP500_historical_data_Close')
df_normalized = df_normalized[cols]

# Save the normalized DataFrame to a new CSV file
df_normalized.to_csv(directory + 'normalized_complete_data.csv', index=False)

print("df_normalized shape: ", df_normalized.shape)
print("Le fichier normalisé avec les features SP500 a été sauvegardé")