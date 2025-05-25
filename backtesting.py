import numpy as np
import pandas as pd
import config
import os
import evaluation
import data_loader
import inference

def run_simple_backtest(predictions_df: pd.DataFrame,
                        price_col: str = config.TARGET_PRICE_COL, # Price used for transactions
                        prediction_col: str = 'Prediction',
                        initial_cash: float = config.INITIAL_CASH,
                        fee: float = config.TRANSACTION_FEE,
                        position_map: dict = None): # e.g. config.POSITION_SIZE_MAP
    """
    Performs a simple vectorized backtest based on signals.
    """
    if predictions_df.empty:
        print("Predictions DataFrame is empty. Cannot run backtest.")
        return None, {}

    if position_map is None:
        position_map = config.POSITION_SIZE_MAP # Default from config

    df_bt = predictions_df.copy()
    
    # Ensure price_col exists
    if price_col not in df_bt.columns:
        raise ValueError(f"Price column '{price_col}' not found in predictions_df for backtesting.")

    # --- Vectorized approach (simpler, assumes rebalancing at each step fully) ---
    # df_bt['Signal'] = df_bt[prediction_col]
    # df_bt['TargetPositionPct'] = df_bt['Signal'].map(position_map).fillna(0) # Map signal to target % of portfolio in asset

    # # Calculate daily returns of the asset
    # df_bt['AssetReturn'] = df_bt[price_col].pct_change().fillna(0)
    
    # # Strategy Return: TargetPositionPct (from previous day's signal) * AssetReturn (of current day)
    # # This assumes position is taken at close of t-1 (based on signal at t-1) and benefits from return from t-1 to t.
    # df_bt['StrategyReturnNoFee'] = df_bt['TargetPositionPct'].shift(1) * df_bt['AssetReturn']
    # df_bt['StrategyReturnNoFee'].fillna(0, inplace=True)

    # # Simplified fee: apply on change in position
    # df_bt['PositionChange'] = df_bt['TargetPositionPct'].diff().abs().fillna(0)
    # df_bt['TransactionCosts'] = df_bt['PositionChange'] * fee
    # df_bt['StrategyReturn'] = df_bt['StrategyReturnNoFee'] - df_bt['TransactionCosts']
    
    # df_bt['PortfolioValue'] = initial_cash * (1 + df_bt['StrategyReturn']).cumprod()
    # df_bt['PortfolioValue'].iloc[0] = initial_cash # Start with initial cash

    # --- Iterative approach (more realistic for fees and cash management) ---
    cash = initial_cash
    position_units = 0.0 # Number of units of the asset
    portfolio_values = []

    for idx, row in df_bt.iterrows():
        current_price = row[price_col]
        signal = row[prediction_col]
        
        current_portfolio_value = cash + position_units * current_price
        portfolio_values.append(current_portfolio_value)

        target_position_pct = position_map.get(signal, 0.0) # Get desired % allocation from signal
        
        # Value of asset we want to hold
        target_asset_value = current_portfolio_value * target_position_pct
        
        # How many units is that?
        target_units = target_asset_value / current_price if current_price > 0 else 0
        
        units_to_trade = target_units - position_units
        
        if units_to_trade > 0: # Buy
            cost_per_unit = current_price * (1 + fee)
            can_buy_units = cash / cost_per_unit
            actual_units_bought = min(units_to_trade, can_buy_units)
            
            cash -= actual_units_bought * cost_per_unit
            position_units += actual_units_bought
            
        elif units_to_trade < 0: # Sell
            proceeds_per_unit = current_price * (1 - fee)
            actual_units_sold = min(abs(units_to_trade), position_units) # Can't sell more than held
            
            cash += actual_units_sold * proceeds_per_unit
            position_units -= actual_units_sold
            
    df_bt['PortfolioValue'] = portfolio_values
    df_bt['PortfolioReturn'] = df_bt['PortfolioValue'].pct_change().fillna(0)
    df_bt['CumulativeReturn'] = (df_bt['PortfolioValue'] / initial_cash) - 1
    
    # --- Calculate Backtest Metrics ---
    final_value = df_bt['PortfolioValue'].iloc[-1]
    total_pnl_abs = final_value - initial_cash
    total_pnl_pct = total_pnl_abs / initial_cash

    # Max Drawdown
    rolling_max = df_bt['PortfolioValue'].cummax()
    daily_drawdown = df_bt['PortfolioValue'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()

    # Sharpe Ratio (annualized)
    # Assuming daily data, 252 trading days
    if df_bt['PortfolioReturn'].std() != 0:
        sharpe_ratio = (df_bt['PortfolioReturn'].mean() / df_bt['PortfolioReturn'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Benchmark (S&P 500 simple return over the period)
    if not df_bt[price_col].empty and df_bt[price_col].iloc[0] != 0 :
        benchmark_return = (df_bt[price_col].iloc[-1] / df_bt[price_col].iloc[0]) - 1
    else:
        benchmark_return = 0.0

    metrics = {
        "Final Portfolio Value": final_value,
        "Total PnL ($)": total_pnl_abs,
        "Total PnL (%)": total_pnl_pct,
        "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        f"Benchmark ({price_col}) Return (%)": benchmark_return
    }
    evaluation.print_metrics_summary(metrics, "Backtest Results")
    
    # Save backtest results
    os.makedirs(config.PREDICTION_SAVE_DIR, exist_ok=True) # Same dir as predictions
    bt_filename = os.path.join(config.PREDICTION_SAVE_DIR, f"backtest_results_v{config.MODEL_VERSION}.csv") # Use global version for now
    df_bt.to_csv(bt_filename, index=False)
    print(f"Backtest details saved to {bt_filename}")
    
    return df_bt, metrics


if __name__ == "__main__":
    df_all_data = data_loader.load_data(config.DATA_FILE_PATH)
    _, df_inference_data = data_loader.split_data_for_rolling_holdout(df_all_data) # Using holdout as inference data

    predicted_data_df, _ = inference.predict_with_model(
    df_new_data=df_inference_data.copy(), # Use a copy
    model_version=config.MODEL_VERSION,
    price_col=config.TARGET_PRICE_COL)

    if predicted_data_df is not None and not predicted_data_df.empty:
        # --- Phase 4: Backtest on Inference Results ---
        print("\n--- Running Backtest on Inference Results ---")
        backtest_df, backtest_metrics = run_simple_backtest(
            predictions_df=predicted_data_df,
            price_col=config.TARGET_PRICE_COL, # S&P500 close for transactions
            prediction_col='Prediction',
            position_map=config.POSITION_SIZE_MAP 
        )