import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import *

def assign_classes(ret_series, thresholds):
    return pd.cut(ret_series, bins=[-np.inf] + list(thresholds) + [np.inf], labels=False)

def build_adaptive_thresholds(df, step_months, n_classes, date_col, ret_col):
    df = df.copy()
    df["month"] = df[date_col].dt.to_period("M")
    thresholds_dict = {}

    for period in sorted(df["month"].unique()):
        ret_vals = df[df["month"] == period][ret_col].dropna()
        if len(ret_vals) >= n_classes:
            quantiles = np.quantile(ret_vals, np.linspace(0, 1, n_classes + 1)[1:-1])
            thresholds_dict[str(period)] = quantiles

    return thresholds_dict


def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)

def prepare_data(df, selected_features, date_col, target_price_col, pred_horizon, n_classes,
                 threshold_strategy, threshold_step_months, fixed_thresholds,
                 jitter_ratio, timesteps, train_years, val_years, test_months,
                 buffer_days, holdout_start, holdout_end):

    if threshold_strategy == "adaptive":
        thresholds_dict = build_adaptive_thresholds(df, threshold_step_months, n_classes, date_col, "ret_future")
        df["target"] = df.apply(lambda row: assign_classes(
            pd.Series(row["ret_future"]),
            thresholds_dict.get(str(row[date_col].to_period("M")), FIXED_THRESHOLDS)
        )[0], axis=1)
    else:
        df["target"] = assign_classes(df["ret_future"], fixed_thresholds)

    df = df.dropna(subset=["target"])

    walk_forward_splits = []
    start = pd.to_datetime(holdout_start)
    end = pd.to_datetime(holdout_end)

    while start + pd.DateOffset(years=train_years + val_years) + pd.DateOffset(months=test_months) <= end:
        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)

        val_start = train_end + pd.DateOffset(days=1 + buffer_days)
        val_end = val_start + pd.DateOffset(years=val_years) - pd.DateOffset(days=1)

        test_start = val_end + pd.DateOffset(days=1 + buffer_days)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)
        test_mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(df.loc[train_mask, selected_features])
        X_val = scaler.transform(df.loc[val_mask, selected_features])
        X_test = scaler.transform(df.loc[test_mask, selected_features])

        y_train = df.loc[train_mask, "target"].astype(int).values
        y_val = df.loc[val_mask, "target"].astype(int).values
        y_test = df.loc[test_mask, "target"].astype(int).values
        test_dates = df.loc[test_mask, date_col]

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, timesteps)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, timesteps)
        test_dates = test_dates[timesteps:].reset_index(drop=True)

        walk_forward_splits.append((X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, test_dates))

        start = start + pd.DateOffset(months=test_months)

    return walk_forward_splits
