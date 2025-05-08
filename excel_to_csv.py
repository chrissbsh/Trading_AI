import pandas as pd

excel_df = pd.read_excel('csv_data/indicators/sp500_PE_Ratio.xlsx', sheet_name=None, engine='openpyxl')

for sheet_name, data in excel_df.items():
    csv_file_name = f'csv_data/indicators/sp500_PE_Ratio.csv'
    data.to_csv(csv_file_name, index=False)