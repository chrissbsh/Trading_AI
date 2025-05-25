import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tensorflow as tf
from functools import partial
import psutil

import config
import data_loader
import preprocessing_v2
import model_architectures
import evaluation

# Configuration optimisée pour TensorFlow
def configure_tensorflow():
    """Configure TensorFlow pour utiliser optimalement CPU et NPU"""
    # Configuration CPU
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
    
    
    # Optimisations générales
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
    os.environ['TF_NUM_INTEROP_THREADS'] = str(psutil.cpu_count())
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(psutil.cpu_count())

def train_single_fold(args):
    """
    Entraîne un seul fold - fonction isolée pour le multiprocessing
    """
    (split_num, start_date, df_roll, features_list, model_builder_fn, price_col) = args
    
    # Reconfigurer TensorFlow dans chaque processus
    configure_tensorflow()
    config.set_seeds()
    
    print(f"\n--- Processing Split {split_num}: Start Date {start_date.date()} ---")
    
    try:
        tr_df, va_df, te_df, tr_s, tr_e, v_e, te_s, te_e = \
            data_loader.get_window_splits(df_roll, start_date)

        if tr_df.empty or va_df.empty or te_df.empty:
            print(f"Warning: Empty split(s) for start_date {start_date}. Skipping.")
            return None
        
        print(f"Split {split_num} - Train: {tr_s.date()} - {tr_e.date()} ({len(tr_df)} days)")
        print(f"Split {split_num} - Val:   {(tr_e + pd.Timedelta(days=1)).date()} - {v_e.date()} ({len(va_df)} days)")
        print(f"Split {split_num} - Test:  {te_s.date()} - {te_e.date()} ({len(te_df)} days)")

        (X_train_final, y_train_final, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
         scaler, fold_thresholds_map, 
         tr_df_processed, va_df_processed, te_df_processed) = \
            preprocessing_v2.preprocess_fold_data(
                tr_df.copy(), va_df.copy(), te_df.copy(), features_list,
                threshold_strategy=config.THRESHOLD_STRATEGY,
                price_col=price_col,
                horizon=config.PRED_HORIZON,
                n_classes=config.N_CLASSES,
                timesteps=config.TIMESTEPS,
                jitter_ratio=config.JITTER_RATIO
            )
        
        if X_train_final is None:
            print(f"Preprocessing failed for split {split_num}, skipping.")
            return None
        
        if X_train_final.shape[0] < config.BATCH_SIZE or X_va_seq.shape[0] < config.BATCH_SIZE:
            print(f"Warning: Not enough samples in train/val for batch size {config.BATCH_SIZE}. Skipping split {split_num}.")
            return None

        # Class weights
        if len(np.unique(y_train_final)) < config.N_CLASSES:
            print(f"Warning: Not all classes present in training data for split {split_num}. Using uniform weights.")
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
            verbose=1
        )
        
        print(f"Training model for split {split_num}...")
        history = model.fit(
            X_train_final, y_train_final,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_va_seq, y_va_seq),
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=0
        )

        print(f"Split {split_num} - Training completed. Best val_loss: {min(history.history['val_loss']):.4f} at epoch {np.argmin(history.history['val_loss'])+1}. Train_loss at that epoch: {history.history['loss'][np.argmin(history.history['val_loss'])]:.4f}")

        # Predictions
        y_pred_va_probs = model.predict(X_va_seq, verbose=0)
        y_pred_va = np.argmax(y_pred_va_probs, axis=1)
        
        y_pred_te_probs = model.predict(X_te_seq, verbose=0)
        y_pred_te = np.argmax(y_pred_te_probs, axis=1)

        # Metrics
        val_metrics = evaluation.calculate_classification_metrics(y_va_seq, y_pred_va)
        f1_val = val_metrics["F1"]
        test_cls_metrics = evaluation.calculate_classification_metrics(y_te_seq, y_pred_te)
        
        # PnL Metrics
        if len(te_df_processed) > config.TIMESTEPS and 'ret_future' in te_df_processed.columns:
            # Align returns_for_pnl_test with y_te_seq
            returns_for_pnl_test = te_df_processed['ret_future'].iloc[config.TIMESTEPS-1:config.TIMESTEPS-1+len(y_te_seq)]
            
            # Debug: Print lengths to confirm alignment
            print(f"Split {split_num} - y_te_seq length: {len(y_te_seq)}")
            print(f"Split {split_num} - y_pred_te length: {len(y_pred_te)}")
            print(f"Split {split_num} - returns_for_pnl_test length: {len(returns_for_pnl_test)}")
            
            # Check for NaNs
            if returns_for_pnl_test.isna().any():
                print(f"Split {split_num} - Warning: NaNs in returns_for_pnl_test. Filling with 0.")
                returns_for_pnl_test = returns_for_pnl_test.fillna(0)
            
            pnl_test, sharpe_test = evaluation.calculate_pnl_sharpe_metrics(
                y_te_seq, y_pred_te, returns_for_pnl_test, config.PRED_HORIZON
            )
        else:
            pnl_test, sharpe_test = 0.0, 0.0
            print(f"Split {split_num} - Warning: Not enough data for PnL calculation.")

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
        
        # Préparer les données du modèle pour le retour
        model_data = {
            "model": model,
            "scaler": scaler,
            "fold_thresholds_map": fold_thresholds_map,
            "features_list": features_list,
            "info": fold_result,
            "model_input_shape": (config.TIMESTEPS, X_train_final.shape[2])
        }
        
        print(f"Split {split_num} results: F1_val={f1_val:.4f}, F1_test={test_cls_metrics['F1']:.4f}, PnL_test={pnl_test:.4f}, Sharpe_test={sharpe_test:.2f}")
        
        return fold_result, model_data, f1_val
        
    except Exception as e:
        print(f"Error in split {split_num}: {str(e)}")
        return None

def train_walk_forward_parallel(df_roll: pd.DataFrame,
                               features_list: list,
                               model_builder_fn,
                               price_col: str = config.TARGET_PRICE_COL,
                               max_workers: int = None):
    """
    Version parallélisée de train_walk_forward utilisant ProcessPoolExecutor
    """
    # Configuration TensorFlow
    configure_tensorflow()
    
    # Déterminer le nombre optimal de workers
    if max_workers is None:
        cpu_count = psutil.cpu_count()
        # Utiliser moins de workers que de CPU pour éviter la surcharge
        max_workers = max(1, min(cpu_count // 2, 4))  # Max 4 processus parallèles
    
    print(f"Using {max_workers} parallel workers (CPU count: {psutil.cpu_count()})")
    
    starts = data_loader.get_walk_forward_starts(df_roll)
    results = []
    best_f1_val = -np.inf
    best_model_set = {}

    print(f"\nStarting parallel walk-forward training with {len(starts)} splits...")
    
    # Préparer les arguments pour chaque fold
    fold_args = [
        (split_num, start_date, df_roll, features_list, model_builder_fn, price_col)
        for split_num, start_date in enumerate(starts, 1)
    ]
    
    # Exécution parallèle
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les jobs
        future_to_split = {
            executor.submit(train_single_fold, args): args[0] 
            for args in fold_args
        }
        
        # Collecter les résultats au fur et à mesure
        for future in as_completed(future_to_split):
            split_num = future_to_split[future]
            try:
                result = future.result()
                if result is not None:
                    fold_result, model_data, f1_val = result
                    results.append(fold_result)
                    
                    # Vérifier si c'est le meilleur modèle
                    if f1_val > best_f1_val:
                        best_f1_val = f1_val
                        best_model_set = model_data
                        print(f"*** New best model found from split {split_num} with F1_val: {best_f1_val:.4f} ***")
                        
            except Exception as e:
                print(f"Split {split_num} generated an exception: {e}")
    
    results_df = pd.DataFrame(results)
    
    if not best_model_set:
        print("Warning: No best model was found. This could be due to all splits failing or having poor validation scores.")
    else:
        print(f"\nBest model overall from split {best_model_set['info']['split']} with F1_val: {best_model_set['info']['F1_val']:.4f}")

    return results_df, best_model_set

# def train_walk_forward_batch_parallel(df_roll: pd.DataFrame,
#                                      features_list: list,
#                                      model_builder_fn,
#                                      price_col: str = config.TARGET_PRICE_COL,
#                                      batch_size: int = 2):
#     """
#     Version alternative qui traite les folds par petits lots pour un meilleur contrôle mémoire
#     """
#     configure_tensorflow()
    
#     starts = data_loader.get_walk_forward_starts(df_roll)
#     results = []
#     best_f1_val = -np.inf
#     best_model_set = {}
    
#     print(f"\nStarting batch parallel walk-forward training with {len(starts)} splits in batches of {batch_size}...")
    
#     # Traiter par lots
#     for i in range(0, len(starts), batch_size):
#         batch_starts = starts[i:i+batch_size]
#         print(f"\nProcessing batch {i//batch_size + 1}/{(len(starts) + batch_size - 1)//batch_size}")
        
#         fold_args = [
#             (split_num, start_date, df_roll, features_list, model_builder_fn, price_col)
#             for split_num, start_date in enumerate(batch_starts, i+1)
#         ]
        
#         with ProcessPoolExecutor(max_workers=min(len(fold_args), psutil.cpu_count()//2)) as executor:
#             futures = [executor.submit(train_single_fold, args) for args in fold_args]
            
#             for future, args in zip(futures, fold_args):
#                 split_num = args[0]
#                 try:
#                     result = future.result()
#                     if result is not None:
#                         fold_result, model_data, f1_val = result
#                         results.append(fold_result)
                        
#                         if f1_val > best_f1_val:
#                             best_f1_val = f1_val
#                             best_model_set = model_data
#                             print(f"*** New best model found from split {split_num} with F1_val: {best_f1_val:.4f} ***")
                            
#                 except Exception as e:
#                     print(f"Split {split_num} generated an exception: {e}")
    
#     results_df = pd.DataFrame(results)
    
#     if not best_model_set:
#         print("Warning: No best model was found.")
#     else:
#         print(f"\nBest model overall from split {best_model_set['info']['split']} with F1_val: {best_model_set['info']['F1_val']:.4f}")

#     return results_df, best_model_set


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

    model_cfg_to_save = {
        "scaler": best_model_set["scaler"],
        "features_list": best_model_set["features_list"],
        "fold_thresholds_map": best_model_set["fold_thresholds_map"],
        "timesteps": config.TIMESTEPS,
        "n_classes": config.N_CLASSES,
        "pred_horizon": config.PRED_HORIZON,
        "price_col_for_returns": config.TARGET_PRICE_COL,
        "info": best_model_set["info"],
        "model_input_shape": best_model_set.get("model_input_shape")
    }
    with open(config_path, "wb") as f:
        pickle.dump(model_cfg_to_save, f)
    print(f"Model configuration saved to {config_path}")