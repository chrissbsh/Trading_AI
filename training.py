import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import pickle

import config
import data_loader
import preprocessing
import model_architectures
import evaluation

def train_walk_forward(df_roll: pd.DataFrame,
                       features_list: list,
                       model_builder_fn, # e.g., model_architectures.build_lstm_model
                       price_col: str = config.TARGET_PRICE_COL):
    """
    Performs walk-forward validation, training, and evaluation.
    Returns a list of results per fold and the best model set.
    """
    config.set_seeds() # Ensure reproducibility for each training run if desired
    
    starts = data_loader.get_walk_forward_starts(df_roll)
    results = []
    best_f1_val = -np.inf
    best_model_set = {} # To store model, scaler, fold_thresholds_map, info

    print(f"\nStarting walk-forward training with {len(starts)} splits...")

    for split_num, start_date in enumerate(starts, 1):
        print(f"\n--- Processing Split {split_num}/{len(starts)}: Start Date {start_date.date()} ---")

        tr_df, va_df, te_df, tr_s, tr_e, v_e, te_s, te_e = \
            data_loader.get_window_splits(df_roll, start_date)

        if tr_df.empty or va_df.empty or te_df.empty:
            print(f"Warning: Empty split(s) for start_date {start_date}. Skipping.")
            continue
        
        print(f"Train: {tr_s.date()} - {tr_e.date()} ({len(tr_df)} days)")
        print(f"Val:   {(tr_e + pd.Timedelta(days=1)).date()} - {v_e.date()} ({len(va_df)} days)")
        print(f"Test:  {te_s.date()} - {te_e.date()} ({len(te_df)} days)")


        (X_train_final, y_train_final, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
         scaler, fold_thresholds_map, 
         tr_df_processed, va_df_processed, te_df_processed) = \
            preprocessing.preprocess_fold_data(
                tr_df.copy(), va_df.copy(), te_df.copy(), features_list,
                price_col=price_col,
                horizon=config.PRED_HORIZON,
                n_classes=config.N_CLASSES,
                timesteps=config.TIMESTEPS,
                jitter_ratio=config.JITTER_RATIO
            )
        
        if X_train_final is None: # Preprocessing failed (e.g. empty sequences)
            print(f"Preprocessing failed for split {split_num}, skipping.")
            continue
        
        if X_train_final.shape[0] < config.BATCH_SIZE or X_va_seq.shape[0] < config.BATCH_SIZE :
            print(f"Warning: Not enough samples in train/val for batch size {config.BATCH_SIZE}. Skipping split {split_num}.")
            continue


        # Class weights
        if len(np.unique(y_train_final)) < config.N_CLASSES:
            print(f"Warning: Not all classes present in training data for split {split_num}. Using uniform weights or skipping.")
            # Handle this case: either skip, use uniform weights, or ensure N_CLASSES reflects actual data
            class_weights = None 
        else:
            cw = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
            class_weights = dict(zip(np.unique(y_train_final), cw))

        # Model
        model = model_builder_fn(
            input_shape=(config.TIMESTEPS, X_train_final.shape[2]),
            n_classes=config.N_CLASSES
        )
        early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=0
        )
        
        print(f"Training model for split {split_num}...")
        history = model.fit(
            X_train_final, y_train_final,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_va_seq, y_va_seq),
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=0 # Set to 1 or 2 for more verbose training
        )
        print(f"Training completed. Best val_loss: {min(history.history['val_loss']):.4f} at epoch {np.argmin(history.history['val_loss'])+1}")

        # Predictions
        y_pred_va_probs = model.predict(X_va_seq, verbose=0)
        y_pred_va = np.argmax(y_pred_va_probs, axis=1)
        
        y_pred_te_probs = model.predict(X_te_seq, verbose=0)
        y_pred_te = np.argmax(y_pred_te_probs, axis=1)

        # Metrics for Validation set
        val_metrics = evaluation.calculate_classification_metrics(y_va_seq, y_pred_va)
        f1_val = val_metrics["F1"]

        # Metrics for Test set
        test_cls_metrics = evaluation.calculate_classification_metrics(y_te_seq, y_pred_te)
        
        # PnL Metrics for Test set
        # Ensure returns align with predictions (they are made on sequences of length TIMESTEPS)
        # te_df_processed['ret_future'] should have been calculated in preprocess_fold_data
        # The returns used for PnL should correspond to the actual returns *after* the prediction point
        # `te_df_processed` is already filtered and contains `ret_future`
        # The sequences start from `TIMESTEPS`, so predictions align with `te_df_processed.iloc[TIMESTEPS-1+PRED_HORIZON-1:]` if we consider the end of the sequence
        # Simpler: `te_df_processed['ret_future']` is used. Need to align it with `y_te_seq`
        # `y_te_seq` corresponds to `te_df_processed['target'].iloc[TIMESTEPS:]`
        # So, `te_df_processed['ret_future'].iloc[TIMESTEPS:]` is the correct series for PnL
        
        if len(te_df_processed) > config.TIMESTEPS and 'ret_future' in te_df_processed.columns:
            returns_for_pnl_test = te_df_processed['ret_future'].iloc[config.TIMESTEPS:]
            pnl_test, sharpe_test = evaluation.calculate_pnl_sharpe_metrics(
                y_te_seq, y_pred_te, returns_for_pnl_test, config.PRED_HORIZON
            )
        else:
            pnl_test, sharpe_test = 0.0, 0.0
            print("Warning: Not enough data in te_df_processed or 'ret_future' missing for PnL calculation.")


        fold_result = {
            "split": split_num,
            "train_window": f"{tr_s.date()} to {tr_e.date()}",
            "val_window": f"{(tr_e + pd.Timedelta(days=1)).date()} to {v_e.date()}",
            "test_window": f"{te_s.date()} to {te_e.date()}",
            "F1_val": f1_val,
            "F1_test": test_cls_metrics["F1"],
            "BalAcc_test": test_cls_metrics["BalAcc"],
            "Precision_test": test_cls_metrics["Precision"],
            "Recall_test": test_cls_metrics["Recall"],
            "PnL_test": pnl_test,
            "Sharpe_test": sharpe_test,
        }
        results.append(fold_result)
        print(f"Split {split_num} results: F1_val={f1_val:.4f}, F1_test={test_cls_metrics['F1']:.4f}, PnL_test={pnl_test:.4f}, Sharpe_test={sharpe_test:.2f}")

        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_model_set = {
                "model": model, # The trained Keras model object
                "scaler": scaler,
                "fold_thresholds_map": fold_thresholds_map, # Thresholds specific to this best fold
                "features_list": features_list,
                "info": fold_result,
                "model_input_shape": (config.TIMESTEPS, X_train_final.shape[2]) # Save for inference
            }
            print(f"*** New best model found based on F1_val: {best_f1_val:.4f} ***")
            
    results_df = pd.DataFrame(results)
    
    if not best_model_set:
        print("Warning: No best model was found. This could be due to all splits failing or having poor validation scores.")
    else:
        print(f"\nBest model overall from split {best_model_set['info']['split']} with F1_val: {best_model_set['info']['F1_val']:.4f}")

    return results_df, best_model_set


def save_best_model(best_model_set, version=config.MODEL_VERSION):
    """Saves the best model and its configuration."""
    if not best_model_set:
        print("No best model to save.")
        return

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_v{version}.keras")
    config_path = os.path.join(config.MODEL_SAVE_DIR, f"model_config_v{version}.pkl")

    best_model_set["model"].save(model_path)
    print(f"Best model saved to {model_path}")

    # Prepare config for saving (model object itself is not pickled here)
    model_cfg_to_save = {
        "scaler": best_model_set["scaler"],
        "features_list": best_model_set["features_list"],
        "fold_thresholds_map": best_model_set["fold_thresholds_map"], # The thresholds from the best fold
        "timesteps": config.TIMESTEPS, # From global config, but good to save with model
        "n_classes": config.N_CLASSES, # Same
        "pred_horizon": config.PRED_HORIZON,
        "price_col_for_returns": config.TARGET_PRICE_COL,
        "info": best_model_set["info"], # Info about the best split
        "model_input_shape": best_model_set.get("model_input_shape") # Important for recreating model or validation
    }
    with open(config_path, "wb") as f:
        pickle.dump(model_cfg_to_save, f)
    print(f"Model configuration saved to {config_path}")