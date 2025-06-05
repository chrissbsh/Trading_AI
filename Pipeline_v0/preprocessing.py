import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from collections import OrderedDict
import config

def compute_future_return(close_series: pd.Series, horizon: int) -> pd.Series:
    """Computes future returns over a given horizon."""
    return (close_series.shift(-horizon) - close_series) / close_series

def make_thresholds(series: pd.Series, k: int) -> np.ndarray:
    """Creates k-1 internal thresholds from quantiles of a series."""
    if len(series) == 0:
        return np.array([])
    q = np.linspace(0, 1, k + 1)
    return series.quantile(q).values[1:-1]

def label_from_thresholds(ret_value: float, thresholds: np.ndarray) -> int:
    """Labels a return value based on a set of thresholds."""
    if thresholds is None or len(thresholds) == 0: # Handle empty thresholds
        return 0 # Or some default/error indicator
    for i, t_val in enumerate(thresholds):
        if ret_value <= t_val:
            return i
    return len(thresholds) # Last class

def build_sliding_thresholds(df: pd.DataFrame,
                             price_col: str = config.TARGET_PRICE_COL,
                             date_col: str = config.DATE_COL,
                             horizon: int = config.PRED_HORIZON,
                             n_classes: int = config.N_CLASSES,
                             step_months: int = config.THRESHOLD_STEP_MONTHS):
    """Builds a map of thresholds calculated on sliding windows."""
    df_copy = df.copy()
    df_copy['ret_future'] = compute_future_return(df_copy[price_col], horizon)
    df_copy = df_copy.dropna(subset=['ret_future'])
    
    thresholds_map = OrderedDict()
    # Ensure dates are datetime objects for comparison
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    start_date = df_copy[date_col].min()
    end_date = df_copy[date_col].max()
    
    current_window_start = start_date
    while current_window_start + relativedelta(months=step_months) <= end_date:
        window_end = current_window_start + relativedelta(months=step_months)
        window_df = df_copy[(df_copy[date_col] >= current_window_start) & (df_copy[date_col] < window_end)]
        
        if len(window_df) >= 100:  # Avoid small windows
            thresholds = make_thresholds(window_df["ret_future"], n_classes)
            thresholds_map[(current_window_start, window_end)] = thresholds

            print(f"Window: {current_window_start} to {window_end}, Thresholds: {thresholds}")
        
        # Move window with 50% overlap if step_months is not too small
        # If step_months is small, simple increment might be better
        advance_months = max(1, step_months // 2) 
        current_window_start += relativedelta(months=advance_months)
    
    # Add a final catch-all window if the last period wasn't covered
    if not thresholds_map or list(thresholds_map.keys())[-1][1] < end_date:
        final_window_df = df_copy[df_copy[date_col] >= (end_date - relativedelta(months=step_months))]
        if len(final_window_df) >= 100:
             thresholds = make_thresholds(final_window_df["ret_future"], n_classes)
             thresholds_map[(end_date - relativedelta(months=step_months), end_date + pd.Timedelta(days=1))] = thresholds # Ensure end_date is included
        elif thresholds_map: # Fallback to last known good thresholds
            thresholds_map[(end_date - relativedelta(months=step_months), end_date + pd.Timedelta(days=1))] = list(thresholds_map.values())[-1]


    if not thresholds_map and len(df_copy) > 0: # Fallback if no windows generated but data exists
        thresholds = make_thresholds(df_copy["ret_future"], n_classes)
        thresholds_map[(start_date, end_date + pd.Timedelta(days=1))] = thresholds

    return thresholds_map


def label_with_sliding_thresholds(df: pd.DataFrame, 
                                  thresholds_map: OrderedDict,
                                  date_col: str = config.DATE_COL):
    """Applies labels to a DataFrame using a pre-computed thresholds_map."""
    labels = []
    if not thresholds_map: # Handle empty thresholds_map
        # This case should ideally be prevented by robust build_sliding_thresholds
        # Or, provide a default labeling strategy (e.g., all class 0 or based on simple quantiles of current ret_future)
        # For now, returning NaNs or raising an error might be appropriate
        return pd.Series([np.nan] * len(df), index=df.index)


    df[date_col] = pd.to_datetime(df[date_col]) # Ensure date column is datetime
    
    for _, row in df.iterrows():
        date = row[date_col]
        ret_val = row["ret_future"]
        applicable_thresholds = None
        
        # Find the most recent window that STARTS BEFORE or AT the current row's date
        best_match_start_date = None
        for (start, end), thres in thresholds_map.items():
            if start <= date:
                if best_match_start_date is None or start > best_match_start_date:
                    best_match_start_date = start
                    applicable_thresholds = thres
            # If date is beyond all defined windows, use the last one
            # This is implicitly handled if the loop finishes and applicable_thresholds is set to the last valid one
            
        if applicable_thresholds is None and thresholds_map:
             # Fallback: use the thresholds from the very last defined window
            applicable_thresholds = list(thresholds_map.values())[-1]
        
        if applicable_thresholds is not None:
            labels.append(label_from_thresholds(ret_val, applicable_thresholds))
        else:
            labels.append(np.nan) # Or some other placeholder if no thresholds apply
            
    return pd.Series(labels, index=df.index)


def create_sequences(X_data, y_data, timesteps: int = config.TIMESTEPS):
    """Creates sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(timesteps, len(X_data)):
        Xs.append(X_data[i - timesteps:i])
        ys.append(y_data[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

def add_jitter(X_data, ratio: float = config.JITTER_RATIO):
    """Adds Gaussian noise (jitter) to the data."""
    if X_data.ndim == 2: # If 2D (features, samples) before sequencing
        std_dev = X_data.std(axis=0, keepdims=True)
    elif X_data.ndim == 3: # If 3D (samples, timesteps, features)
        std_dev = X_data.std(axis=(0,1), keepdims=True)
    else:
        raise ValueError("Input X_data must be 2D or 3D for jitter.")
    noise = np.random.normal(0, std_dev * ratio, X_data.shape)
    return X_data + noise

def scale_features(X_train, X_val, X_test):
    """Scales features using StandardScaler fit on training data."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def preprocess_fold_data(tr_df, va_df, te_df, features_list,
                         price_col=config.TARGET_PRICE_COL,
                         horizon=config.PRED_HORIZON,
                         n_classes=config.N_CLASSES,
                         timesteps=config.TIMESTEPS,
                         jitter_ratio=config.JITTER_RATIO):
    """Full preprocessing pipeline for a single walk-forward fold."""
    
    # 1. Calculate future returns
    # Ensure 'ret_future' is calculated before thresholding and labeling
    # The thresholds are built on the combined (tr+va+te) data for THIS FOLD ONLY to avoid leakage
    # from future folds, but to have stable thresholds within the current evaluation window.
    
    # Create a temporary combined DataFrame for building thresholds for this specific fold
    # This is crucial for the "recalculated in each training window" requirement.
    combined_fold_df = pd.concat([tr_df, va_df, te_df]).copy()
    combined_fold_df['ret_future'] = compute_future_return(combined_fold_df[price_col], horizon)
    combined_fold_df.dropna(subset=['ret_future'], inplace=True)
    
    # 2. Build sliding thresholds for THIS FOLD based on combined_fold_df
    fold_thresholds_map = build_sliding_thresholds(
        combined_fold_df,
        price_col=price_col,
        horizon=horizon,
        n_classes=n_classes,
        step_months=config.THRESHOLD_STEP_MONTHS
    )

    # 3. Apply returns and labels to individual splits using the fold_thresholds_map
    for part_df in (tr_df, va_df, te_df):
        part_df['ret_future'] = compute_future_return(part_df[price_col], horizon)
        part_df.dropna(subset=['ret_future'], inplace=True) # Drop rows where future return can't be calculated
        part_df['target'] = label_with_sliding_thresholds(part_df, fold_thresholds_map)
        part_df.dropna(subset=['target'], inplace=True) # Drop rows where label could not be assigned

    # Drop trailing NaNs from feature calculation and rows unusable due to PRED_HORIZON lookahead for ret_future
    # This ensures that only complete data rows are used.
    # The iloc[:-horizon] was to remove rows that wouldn't have a future return label.
    # This is now handled by dropna after 'ret_future' and 'target' creation.
    # However, we must ensure an equal number of rows are kept if we don't use 'ret_future' directly as a feature.
    # Typically, data for which 'target' cannot be computed due to horizon is already dropped.
    
    tr_df.dropna(subset=features_list + ['target'], inplace=True)
    va_df.dropna(subset=features_list + ['target'], inplace=True)
    te_df.dropna(subset=features_list + ['target'], inplace=True)

    if tr_df.empty or va_df.empty or te_df.empty:
        print("Warning: One or more data splits are empty after preprocessing. Skipping fold.")
        return None, None, None, None, None, None, None, None, None # Indicate failure

    # 4. Scale features
    X_tr, X_va, X_te, scaler = scale_features(
        tr_df[features_list].values,
        va_df[features_list].values,
        te_df[features_list].values
    )
    y_tr, y_va, y_te = tr_df['target'].values, va_df['target'].values, te_df['target'].values

    # 5. Create sequences
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, timesteps)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, timesteps)
    X_te_seq, y_te_seq = create_sequences(X_te, y_te, timesteps)
    
    if X_tr_seq.shape[0] == 0 or X_va_seq.shape[0] == 0 or X_te_seq.shape[0] == 0:
        print("Warning: Not enough data to create sequences for one or more splits. Skipping fold.")
        return None, None, None, None, None, None, None, None, None # Indicate failure


    # 6. Jitter augmentation for training data
    X_aug_seq = add_jitter(X_tr_seq, jitter_ratio)
    y_aug_seq = y_tr_seq.copy()
    X_train_final = np.vstack([X_tr_seq, X_aug_seq])
    y_train_final = np.concatenate([y_tr_seq, y_aug_seq])

    # Return processed data, scaler, and the specific thresholds_map for this fold
    return (X_train_final, y_train_final, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
            scaler, fold_thresholds_map, tr_df, va_df, te_df)