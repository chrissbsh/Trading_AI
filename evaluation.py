import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             precision_score, recall_score,
                             accuracy_score, confusion_matrix)
import config

def calculate_classification_metrics(y_true, y_pred):
    """Calculates various classification metrics."""
    metrics = {
        "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics

def calculate_pnl_sharpe_metrics(y_true_dummy, y_pred, actual_returns_series, horizon=config.PRED_HORIZON):
    """
    Calculates PnL and Sharpe ratio based on predictions and actual returns.
    y_true_dummy is not used but kept for similar signature to other metrics functions if needed.
    """
    if len(y_pred) == 0 or len(actual_returns_series) == 0:
        return 0.0, 0.0
        
    # Ensure y_pred aligns with the returns series.
    # Predictions are made on sequences, so returns should match the end of these sequences.
    # If actual_returns_series is from df.iloc[TIMESTEPS:], it should align.
    
    # Make sure actual_returns_series is at least as long as y_pred
    if len(actual_returns_series) < len(y_pred):
        print(f"Warning: Returns series (len {len(actual_returns_series)}) is shorter than predictions (len {len(y_pred)}). PnL/Sharpe might be inaccurate.")
        # Truncate y_pred or handle as error, for now, proceed with available returns
        aligned_returns = actual_returns_series.values
    else:
        # Take the relevant part of returns, assuming y_pred corresponds to the end of sequences
        aligned_returns = actual_returns_series.iloc[:len(y_pred)].values


    pos = np.select(
        [y_pred == 2, y_pred == 0],  # Assuming 3 classes: 0=Short, 1=Cash, 2=Long
        [1, -1],                     # Corresponding positions: +1 for Long, -1 for Short
        default=0                    # Default is Cash (0)
    )
    
    strat_ret = pos * aligned_returns
    
    # Handle cases where strat_ret might be all zeros or NaN
    if np.all(strat_ret == 0) or np.isnan(strat_ret).all():
        return 0.0, 0.0

    # PnL (cumulative product of (1 + strategy returns))
    # Ensure no NaNs in strat_ret that would propagate
    strat_ret = np.nan_to_num(strat_ret) 
    
    # Simple sum of returns for PnL if preferred over geometric, or geometric as below
    # pnl = np.sum(strat_ret) # Arithmetic PnL
    
    # Geometric PnL (cumulative product)
    # Add 1 for cumulative product, then subtract 1 at the end.
    # Ensure strat_ret values are such that 1+strat_ret is not negative if using cumprod for value.
    # For returns, often it's fine.
    if len(strat_ret) > 0:
        portfolio_value_over_time = np.cumprod(1 + strat_ret)
        pnl = portfolio_value_over_time[-1] - 1
    else:
        pnl = 0.0

    # Sharpe Ratio
    # Annualize: sqrt(252 trading days / horizon in days)
    # Using daily returns for Sharpe calculation
    if np.std(strat_ret) == 0 or horizon == 0:
        sharpe = 0.0
    else:
        sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(252 / horizon if horizon > 0 else 252)
        
    return float(pnl), float(sharpe)

def print_metrics_summary(metrics_dict, title="Metrics Summary"):
    print(f"\n=== {title} ===")
    for name, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            print(f"{name:<15}: {value:.4f}")
        else:
            print(f"{name:<15}:\n{value}")