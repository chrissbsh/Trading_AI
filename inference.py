import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

import config
import preprocessing_v2
import evaluation # For metrics calculation
from sklearn.metrics import confusion_matrix # For confusion matrix

from collections import OrderedDict
import data_loader # For loading data

def _make_sequences_for_inference(X_scaled, timesteps):
    """Creates sequences from scaled data without labels for inference."""
    if len(X_scaled) < timesteps:
        return np.array([]) # Not enough data to form a single sequence
    
    X_seq = []
    for i in range(timesteps, len(X_scaled) + 1): # Note: len(X_scaled) + 1 to include the last possible sequence
        X_seq.append(X_scaled[i - timesteps:i])
    return np.array(X_seq, dtype=np.float32)


def predict_with_model(df_new_data: pd.DataFrame,
                       model_version: str,
                       price_col: str = config.TARGET_PRICE_COL,
                       date_col: str = config.DATE_COL):
    """
    Loads a saved model and its configuration, preprocesses new data,
    makes predictions, and evaluates if actuals are available.
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_v{model_version}.keras")
    config_path = os.path.join(config.MODEL_SAVE_DIR, f"model_config_v{model_version}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Error: Model or config file not found for version {model_version}")
        return None, None

    print(f"\nLoading model version {model_version} for inference...")
    model = tf.keras.models.load_model(model_path)
    with open(config_path, "rb") as f:
        model_cfg = pickle.load(f)

    scaler = model_cfg["scaler"]
    features_list = model_cfg["features_list"]
    timesteps = model_cfg["timesteps"]
    # Thresholds_map from the best *training fold* is loaded. 
    # For true inference on new data where future is unknown, you might need a strategy:
    # 1. Use the last known thresholds_map from training.
    # 2. Or, if enough new data, try to build a new thresholds_map on recent historical part of df_new_data.
    # Current implementation will use the thresholds_map saved from the best training fold.
    # This is reasonable if the market regime hasn't drastically changed.
    fold_thresholds_map = model_cfg["fold_thresholds_map"] 
    pred_horizon = model_cfg["pred_horizon"]
    n_classes = model_cfg["n_classes"]

    df = df_new_data.copy()

    # Preprocessing for inference:
    # 1. Calculate future returns and labels IF evaluation is desired.
    #    If just predicting, target calculation is not strictly needed but helps for consistency.
    df['ret_future'] = preprocessing_v2.compute_future_return(df[price_col], pred_horizon)
    
    # Labeling using the loaded thresholds_map (from the best training fold)
    # This is for creating 'actual' labels for evaluation on this new_data.
    # If new_data is truly unseen future data, 'ret_future' and 'target' would not be available yet.
    # In that case, this step would be skipped, and only predictions would be generated.
    
    # Create a temporary map to label new data if it's outside the range of fold_thresholds_map
    # This is a practical approach for inference if new data is later than training data.
    # We use the *last* set of thresholds from the loaded map.
    
    # Ensure dates in df are datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    if fold_thresholds_map is not None:
        if isinstance(fold_thresholds_map, OrderedDict):
            last_trained_thresholds = list(fold_thresholds_map.values())[-1]
            # Create a new map that applies these last thresholds to all dates in df
            inference_thresholds_map = preprocessing_v2.OrderedDict()
            if not df.empty:
                min_date, max_date = df[date_col].min(), df[date_col].max()
                inference_thresholds_map[(min_date, max_date + pd.Timedelta(days=1))] = last_trained_thresholds
            else: # if df is empty, this won't be used
                inference_thresholds_map = fold_thresholds_map

        elif isinstance(fold_thresholds_map, np.ndarray):
            inference_thresholds_map = fold_thresholds_map
        else:
            print("Warning: fold_thresholds_map is of an unsupported type.")
            inference_thresholds_map = None

    else: # Should not happen if model_cfg is valid
        print("Warning: fold_thresholds_map is empty in model_cfg.")
        inference_thresholds_map = None

    if inference_thresholds_map is not None:
        df['target_actual'] = preprocessing_v2.label_with_sliding_thresholds(df, inference_thresholds_map, date_col)
    else:
        df['target_actual'] = np.nan


    # Drop rows where features or target might be NaN after calculations
    # Crucially, keep enough rows for at least one sequence *before* scaling features
    # We need at least `timesteps` rows of feature data.
    df.dropna(subset=features_list, inplace=True) # Drop rows if essential features are missing
    # If 'target_actual' is all NaN, evaluation won't be possible, but prediction can still proceed.
    
    if len(df) < timesteps:
        print(f"Not enough data (have {len(df)}, need {timesteps}) to make predictions.")
        return None, None

    # Scale features
    X_scaled = scaler.transform(df[features_list])

    # Create sequences
    X_seq = _make_sequences_for_inference(X_scaled, timesteps)

    if X_seq.shape[0] == 0:
        print("No sequences created from the new data. Cannot predict.")
        return None, None

    # Predict
    predictions_probs = model.predict(X_seq, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)

    # Align predictions with original DataFrame
    # Predictions correspond to df.iloc[timesteps-1:] or df.iloc[timesteps:] depending on _make_sequences_for_inference
    # _make_sequences_for_inference produces sequences where the last element of the sequence is X_scaled[i-1]
    # So, the prediction for X_seq[j] (made from X_scaled[j:j+timesteps]) corresponds to a decision for time df.index[j+timesteps-1]
    # We need to attach predictions to the correct rows in `df`
    
    # The first prediction corresponds to the data point at index `timesteps-1` (if 0-indexed) in the scaled data,
    # which in turn corresponds to original `df` row `timesteps-1` *after* initial NaNs were dropped.
    # Let's align with df starting from the first row that *could* have a prediction.
    
    # df_eval will contain the original data rows for which predictions are available.
    # The first prediction is for the sequence ending at row `timesteps-1` of X_scaled.
    # This means the prediction is valid for the state at `df.index[timesteps-1]`.
    # The `df` used here is already filtered for NaNs in features.
    
    # Create a DataFrame for results
    # The predictions align with the original df from the `timesteps`-th row onwards
    # (after initial NaN dropping for features).
    results_df = df.iloc[timesteps-1:].copy() # -1 because sequence X[0:timesteps] predicts for time t=timesteps-1
    
    if len(results_df) > len(predictions): # Should not happen if logic is correct
        results_df = results_df.iloc[:len(predictions)]
    elif len(predictions) > len(results_df): # Also an issue
        predictions = predictions[:len(results_df)]

    results_df['Prediction'] = predictions
    
    # Optional: Evaluate if 'target_actual' is available and not all NaN
    metrics = None
    if 'target_actual' in results_df.columns and not results_df['target_actual'].isnull().all():
        # Drop rows where target_actual is NaN for fair evaluation
        eval_subset = results_df.dropna(subset=['target_actual'])
        y_true_eval = eval_subset['target_actual'].astype(int)
        y_pred_eval = eval_subset['Prediction'].astype(int)
        
        if len(y_true_eval) > 0:
            print("\nEvaluating predictions on new data (where actuals are available):")
            cls_metrics = evaluation.calculate_classification_metrics(y_true_eval, y_pred_eval)
            evaluation.print_metrics_summary(cls_metrics, "Classification Metrics on New Data")
            
            # PnL metrics if 'ret_future' is available
            if 'ret_future' in eval_subset.columns and not eval_subset['ret_future'].isnull().all():
                returns_for_pnl = eval_subset['ret_future']
                pnl, sharpe = evaluation.calculate_pnl_sharpe_metrics(
                    y_true_eval, y_pred_eval, returns_for_pnl, pred_horizon
                )
                print(f"{'PnL':<15}: {pnl:.4f}")
                print(f"{'Sharpe':<15}: {sharpe:.2f}")
                cls_metrics['PnL'] = pnl
                cls_metrics['Sharpe'] = sharpe
            metrics = cls_metrics
            # Confusion Matrix
            cm = confusion_matrix(y_true_eval, y_pred_eval)
            print("Confusion matrix on new data:\n", cm)


    # Save predictions
    os.makedirs(config.PREDICTION_SAVE_DIR, exist_ok=True)
    pred_filename = os.path.join(config.PREDICTION_SAVE_DIR, f"predictions_v{model_version}_on_new_data.csv")
    results_df.to_csv(pred_filename, index=False)
    print(f"Predictions saved to {pred_filename}")

    if metrics:
        metrics_filename = os.path.join(config.PREDICTION_SAVE_DIR, f"predictions_v{model_version}_on_new_data_metrics.txt")
        with open(metrics_filename, "w") as f_metrics:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    f_metrics.write(f"{k}: {v:.4f}\n")
                else:
                    f_metrics.write(f"{k}:\n{v}\n")
            if 'cm' in locals(): # if confusion matrix was computed
                 f_metrics.write("\nConfusion matrix:\n")
                 f_metrics.write(np.array2string(cm))
        print(f"Prediction metrics saved to {metrics_filename}")

    return results_df, metrics


if __name__ == "__main__":
    # avec ce code et le model 15, on obtient 136 000 en PNL
    # new_data_path = "csv_data/consolidated_data/normalized_complete_data.csv"  # Replace with actual path
    # df_new_data = pd.read_csv(new_data_path)
    
    # # Ensure date column is in datetime format
    # df_new_data[config.DATE_COL] = pd.to_datetime(df_new_data[config.DATE_COL])
    
    # # Predict with the latest model version
    # results, metrics = predict_with_model(df_new_data, model_version=config.MODEL_VERSION)
    
    # if results is not None:
    #     print("Predictions and metrics generated successfully.")
    # else:
    #     print("Prediction failed.")


    df_all_data = data_loader.load_data(config.DATA_FILE_PATH)
    _, df_inference_data = data_loader.split_data_for_rolling_holdout(df_all_data) # Using holdout as inference data

    predicted_data_df, _ = predict_with_model(
    df_new_data=df_inference_data.copy(), # Use a copy
    model_version=config.MODEL_VERSION,
    price_col=config.TARGET_PRICE_COL
)