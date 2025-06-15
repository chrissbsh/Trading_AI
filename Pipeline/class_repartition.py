from config import TARGET_PRICE_COL, DATA_FILE_PATH, DATE_COL, HOLDOUT_START_DATE, HOLDOUT_END_DATE
import pandas as pd



def create_target(df, pred_horizon, fixed_threshold, tag=""):
    print(f"🔹 2) Création de la cible (horizon = {pred_horizon}, seuil={fixed_threshold}) {tag}")
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-pred_horizon) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    df.dropna(subset=['ret_future'], inplace=True)
    
    thresholds = [-fixed_threshold, fixed_threshold]
    def label_target(x):
        if x < thresholds[0]:
            return 0  # Baisse
        elif x <= thresholds[1]:
            return 1  # Neutre
        else:
            return 2  # Hausse
    df["target"] = df["ret_future"].apply(label_target)
    return df

def load_data():
    print("🔹 1) Chargement des données depuis :", DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    print("   → Aperçu des 5 premières lignes :")
    print(df.head())
    return df

pred_horizon = 3

fixed_threshold =  0.01

df_raw = load_data()
cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)

df_holdout_raw = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
df_main_raw = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()

# 1. Préparation des données avec les meilleurs hyperparamètres
df_main_final = create_target(df_main_raw.copy(), pred_horizon, fixed_threshold, tag="(final train)")
df_holdout_final = create_target(df_holdout_raw.copy(), pred_horizon, fixed_threshold, tag="(final holdout)")

split_idx = int(0.8 * len(df_main_final))
train_df = df_main_final.iloc[:split_idx]
val_df = df_main_final.iloc[split_idx:]


print("\nRépartition des classes (en pourcentage) :")
for name, df in [("Train", train_df), ("Val", val_df), ("Holdout", df_holdout_final)]:
    counts = df["target"].value_counts(normalize=True).sort_index() * 100
    print(f"   {name} :")
    for cls, pct in counts.items():
        print(f"     Classe {cls}: {pct:.2f}%")