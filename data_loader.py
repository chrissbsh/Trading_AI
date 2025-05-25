import pandas as pd
from dateutil.relativedelta import relativedelta
import config

def load_data(file_path: str, date_col: str = config.DATE_COL):
    """Loads and sorts data."""
    df = (pd.read_csv(file_path, parse_dates=[date_col])
            .sort_values(date_col)
            .reset_index(drop=True))
    return df

def split_data_for_rolling_holdout(df: pd.DataFrame, date_col: str = config.DATE_COL):
    """Splits data into rolling part and a final hold-out set."""
    holdout_mask = (df[date_col] >= pd.to_datetime(config.HOLDOUT_START_DATE)) & \
                   (df[date_col] <= pd.to_datetime(config.HOLDOUT_END_DATE))
    df_holdout = df[holdout_mask].copy()
    df_roll = df[~holdout_mask].copy()
    return df_roll, df_holdout

def get_walk_forward_starts(df_roll: pd.DataFrame, date_col: str = config.DATE_COL):
    """Generates start dates for walk-forward validation."""
    starts = []
    d0 = df_roll[date_col].min()
    last_possible_start = (df_roll[date_col].max() -
                           relativedelta(years=config.TRAIN_YEARS + config.VAL_YEARS) -
                           relativedelta(months=config.TEST_MONTHS))
    
    current_start = d0
    while current_start <= last_possible_start:
        starts.append(current_start)
        # Advance by the length of the test period for distinct test sets
        current_start += relativedelta(months=config.TEST_MONTHS) 
    return starts

def get_window_splits(df_roll: pd.DataFrame, start_date: pd.Timestamp, date_col: str = config.DATE_COL):
    """Gets train, validation, and test splits for a given start_date."""
    tr_start = start_date
    tr_end = start_date + relativedelta(years=config.TRAIN_YEARS) - pd.Timedelta(days=1)
    val_end = tr_end + relativedelta(years=config.VAL_YEARS)
    test_start_ideal = val_end + pd.Timedelta(days=config.BUFFER_DAYS)
    test_end_ideal = test_start_ideal + relativedelta(months=config.TEST_MONTHS) - pd.Timedelta(days=1)

    # Ensure test_end does not exceed available data
    max_data_date = df_roll[date_col].max()
    if test_end_ideal > max_data_date:
        # This split would be too short or invalid, should be handled by the caller
        # (e.g., by `get_walk_forward_starts` ensuring last_possible_start)
        # For robustness, we can return None or raise an error if this check is critical here
        pass

    tr_df = df_roll[(df_roll[date_col] >= tr_start) & (df_roll[date_col] <= tr_end)].copy()
    va_df = df_roll[(df_roll[date_col] > tr_end) & (df_roll[date_col] <= val_end)].copy()
    te_df = df_roll[(df_roll[date_col] >= test_start_ideal) & (df_roll[date_col] <= test_end_ideal)].copy()
    
    return tr_df, va_df, te_df, tr_start, tr_end, val_end, test_start_ideal, test_end_ideal