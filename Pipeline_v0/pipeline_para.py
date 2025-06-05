import pandas as pd
import numpy as np
import os
import warnings
import config
import data_loader
import feature_selection
import training_para_v2
import model_architectures_v2
import preprocessing_v2
import evaluation # For holdout evaluation
import inference # For inference example
import backtesting # For backtesting example

warnings.filterwarnings("ignore") # From original script, use judiciously

def run_training_pipeline(model_name="lstm_v2", feature_selection_on=False):
    """
    Full training pipeline: Load data, feature select, train walk-forward, save best model.
    """
    print("Starting Training Pipeline...")
    config.set_seeds() # Ensure reproducibility from the start

    # 1. Load Data
    df_full = data_loader.load_data(config.DATA_FILE_PATH)
    df_roll, df_holdout = data_loader.split_data_for_rolling_holdout(df_full)
    print(f"Data loaded: df_roll ({len(df_roll)} rows), df_holdout ({len(df_holdout)} rows)")

    # 2. Feature Selection (on df_roll only)

    if feature_selection_on:
        print("\nPerforming Feature Selection on df_roll...")
        # Exclude Date and the target price column from features used in SHAP's X
        # Also exclude any pre-computed ret_future or target if they exist from previous runs
        initial_exclude_for_shap_X = {config.DATE_COL, config.TARGET_PRICE_COL, 'ret_future', 'target'}
        
        selected_features, shap_scores = feature_selection.shap_feature_selection(
            df_roll.copy(), # Use a copy
            top_n=config.TOP_N_FEATURES,
            price_col=config.TARGET_PRICE_COL,
            horizon=config.PRED_HORIZON,
            n_classes=config.N_CLASSES,
            exclude_cols_from_X = initial_exclude_for_shap_X
        )
        if not selected_features:
            print("Feature selection did not return any features. Exiting.")
            return None, None
        
        print(f"Selected features: {selected_features}")

    else:
        selected_features = df_roll.columns.tolist()
        selected_features.remove(config.DATE_COL)
        selected_features.remove(config.TARGET_PRICE_COL)
        selected_features = [f for f in selected_features if f not in config.EXCLUDE_COLS_FROM_FEATURES]

    # 3. Walk-Forward Training
    print("\nStarting Walk-Forward Training...")
    model_builder_fn = model_architectures_v2.AVAILABLE_MODELS.get(model_name)
    if not model_builder_fn:
        raise ValueError(f"Model architecture '{model_name}' not found in model_architectures.py")

    wf_results_df, best_model_set = training_para_v2.train_walk_forward_parallel(
        df_roll.copy(), # Use a copy
        features_list=selected_features,
        model_builder_fn=model_builder_fn,
        price_col=config.TARGET_PRICE_COL,
        max_workers = None # detection automatique
    )

    print("\nWalk-Forward Training Summary:")
    if wf_results_df is not None and not wf_results_df.empty:
        print(wf_results_df.to_string(index=False))
    else:
        print("No results from walk-forward training.")

    if not best_model_set:
        print("No best model identified from walk-forward training. Cannot proceed to save or holdout evaluation.")
        return wf_results_df, None # Return results even if no best model

    # 4. Save Best Model
    print("\nSaving best model from walk-forward...")
    training_para_v2.save_best_model(best_model_set, version=config.MODEL_VERSION)
    
    return wf_results_df, best_model_set


def run_holdout_evaluation(df_holdout_original: pd.DataFrame, best_model_set_from_training):
    """
    Evaluates the best model on the hold-out set.
    """
    if not best_model_set_from_training:
        print("No best model available for holdout evaluation.")
        return None
    if df_holdout_original.empty:
        print("Holdout dataframe is empty. Skipping holdout evaluation.")
        return None

    print("\nStarting Hold-Out Evaluation...")
    config.set_seeds() # For consistency if any randomness in eval (though less common)
    
    df_holdout = df_holdout_original.copy()

    # Load components from the best_model_set
    model = best_model_set_from_training["model"]
    scaler = best_model_set_from_training["scaler"]
    # Use the thresholds_map from the best FOLD of training for holdout evaluation
    # This is a common practice: apply what was learned as 'best' to unseen data
    fold_thresholds_map = best_model_set_from_training["fold_thresholds_map"] 
    features_list = best_model_set_from_training["features_list"]
    timesteps = config.TIMESTEPS # Or from best_model_set_from_training if stored

    # Preprocess holdout data
    # 1. Compute future returns and labels for evaluation
    df_holdout['ret_future'] = preprocessing_v2.compute_future_return(
        df_holdout[config.TARGET_PRICE_COL], config.PRED_HORIZON
    )
    # For labeling holdout, use the fold_thresholds_map from the best training split.
    # This assumes the "rules" (thresholds) learned by the best model are most generalizable.
    df_holdout['target'] = preprocessing_v2.label_with_sliding_thresholds(
        df_holdout, fold_thresholds_map, config.DATE_COL
    )
    
    # Drop rows where target or essential features might be NaN
    df_holdout.dropna(subset=features_list + ['target', 'ret_future'], inplace=True)

    if len(df_holdout) < timesteps:
        print(f"Not enough data in holdout set after preprocessing (have {len(df_holdout)}, need {timesteps}) for evaluation.")
        return None

    # 2. Scale features
    X_hd_scaled = scaler.transform(df_holdout[features_list])
    y_hd_true = df_holdout['target'].values

    # 3. Create sequences
    X_hd_seq, y_hd_seq_true = preprocessing_v2.create_sequences(X_hd_scaled, y_hd_true, timesteps)

    if X_hd_seq.shape[0] == 0:
        print("No sequences created from holdout data. Cannot evaluate.")
        return None

    # Predictions
    y_hd_pred_probs = model.predict(X_hd_seq, verbose=0)
    y_hd_pred = np.argmax(y_hd_pred_probs, axis=1)

    # Metrics
    print("\nHold-out Set Metrics:")
    holdout_cls_metrics = evaluation.calculate_classification_metrics(y_hd_seq_true, y_hd_pred)
    evaluation.print_metrics_summary(holdout_cls_metrics, "Hold-Out Classification Metrics")
    
    # PnL Metrics for Holdout
    # Returns for PnL should align with y_hd_seq_true
    # y_hd_seq_true corresponds to df_holdout['target'].iloc[timesteps-1:]
    returns_for_pnl_holdout = df_holdout['ret_future'].iloc[timesteps-1:timesteps-1+len(y_hd_seq_true)]

    # Debug: Print lengths to confirm alignment
    print(f"y_hd_seq_true length: {len(y_hd_seq_true)}")
    print(f"y_hd_pred length: {len(y_hd_pred)}")
    print(f"returns_for_pnl_holdout length: {len(returns_for_pnl_holdout)}")

    # Check for NaNs
    if returns_for_pnl_holdout.isna().any():
        print("Warning: NaNs in returns_for_pnl_holdout. Filling with 0.")
        returns_for_pnl_holdout = returns_for_pnl_holdout.fillna(0)

    pnl_hd, sharpe_hd = evaluation.calculate_pnl_sharpe_metrics(
        y_hd_seq_true, y_hd_pred, returns_for_pnl_holdout, config.PRED_HORIZON
    )

    print(f"{'PnL Holdout':<20}: {pnl_hd:.4f}")
    print(f"{'Sharpe Holdout':<20}: {sharpe_hd:.2f}")
    
    holdout_cls_metrics['PnL'] = pnl_hd
    holdout_cls_metrics['Sharpe'] = sharpe_hd

    # Store results in a DataFrame similar to inference output for consistency
    results_df_holdout = df_holdout.iloc[timesteps-1:].copy() # Align with sequences
    if len(results_df_holdout) > len(y_hd_pred):
         results_df_holdout = results_df_holdout.iloc[:len(y_hd_pred)]

    results_df_holdout['Prediction'] = y_hd_pred
    results_df_holdout.rename(columns={'target': 'Actual'}, inplace=True)

    return results_df_holdout, holdout_cls_metrics


if __name__ == "__main__":
    # --- Phase 1: Training and Saving Best Model ---
    # You can choose the model architecture here
    selected_model_arch = "cnn_lstm" # or "lstm_v2" or "gru" if you implement it

    version = config.MODEL_VERSION
    print(f"\n--- Running Training Pipeline for Model Version: {version} ---")

    if os.path.exists(f"IA_training/model/best_model_v{version}.keras"):
        print(f"Warning: best_model_v{version}.keras already exists.")
        input("Press Enter to continue or Ctrl+C to cancel...")
    
    wf_results, best_set = run_training_pipeline(model_name=selected_model_arch, feature_selection_on=True)

    if best_set:
        # --- Phase 2: Evaluate Best Model on Hold-Out Data ---
        # Load fresh holdout data for clean evaluation
        df_full_for_holdout = data_loader.load_data(config.DATA_FILE_PATH)
        _, df_holdout_eval = data_loader.split_data_for_rolling_holdout(df_full_for_holdout)
        
        holdout_predictions_df, holdout_metrics = run_holdout_evaluation(df_holdout_eval, best_set)
        
        if holdout_predictions_df is not None:
            print("\nHoldout evaluation completed. Predictions stored in holdout_predictions_df.")
            # Optionally save holdout_predictions_df and metrics to file here
            os.makedirs(config.PREDICTION_SAVE_DIR, exist_ok=True)
            holdout_predictions_df.to_csv(os.path.join(config.PREDICTION_SAVE_DIR, f"holdout_predictions_v{config.MODEL_VERSION}.csv"), index=False)
            with open(os.path.join(config.PREDICTION_SAVE_DIR, f"holdout_metrics_v{config.MODEL_VERSION}.txt"), "w") as f_met:
                for k,v in holdout_metrics.items(): f_met.write(f"{k}: {v}\n")

    # --- Phase 3: Inference with a Saved Model (example) ---
    # This part assumes a model (e.g., version from config.MODEL_VERSION) has already been trained and saved.
    # You might run this independently after training.
    
    print(f"\n--- Running Inference & Backtest for Model Version: {config.MODEL_VERSION} ---")
    # For inference, we might use the full dataset or a specific recent part.
    # Here, let's use the original holdout data as an example of "new" data.
    df_all_data = data_loader.load_data(config.DATA_FILE_PATH)
    _, df_inference_data = data_loader.split_data_for_rolling_holdout(df_all_data) # Using holdout as inference data

    if df_inference_data.empty:
        print("Inference data (holdout) is empty. Skipping inference and backtest.")
    else:
        predicted_data_df, _ = inference.predict_with_model(
            df_new_data=df_inference_data.copy(), # Use a copy
            model_version=config.MODEL_VERSION,
            price_col=config.TARGET_PRICE_COL
        )

        if predicted_data_df is not None and not predicted_data_df.empty:
            # --- Phase 4: Backtest on Inference Results ---
            print("\n--- Running Backtest on Inference Results ---")
            backtest_df, backtest_metrics = backtesting.run_simple_backtest(
                predictions_df=predicted_data_df,
                price_col=config.TARGET_PRICE_COL, # S&P500 close for transactions
                prediction_col='Prediction',
                position_map=config.POSITION_SIZE_MAP 
            )
            if backtest_df is not None:
                print("Backtesting completed.")
        else:
            print("No predictions generated from inference, skipping backtest.")

    print("\nPipeline finished.")