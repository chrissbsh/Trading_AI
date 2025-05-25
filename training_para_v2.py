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
import evaluation

# Configuration optimisée pour TensorFlow
def configure_tensorflow():
    """Configure TensorFlow pour utiliser optimalement CPU et NPU"""
    # Configuration CPU plus conservative
    # Configuration CPU
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
    
    
    # Optimisations générales
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count())
    os.environ['TF_NUM_INTEROP_THREADS'] = str(psutil.cpu_count())
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(psutil.cpu_count())


def diagnose_data_quality(X_train, y_train, X_val, y_val, split_num):
    """Diagnostique la qualité des données avant entraînement"""
    issues = []
    
    # Vérifier les NaN/Inf
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        issues.append("NaN/Inf détectés dans X_train")
    
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        issues.append("NaN/Inf détectés dans X_val")
    
    # Vérifier la distribution des classes
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    if len(unique_train) != config.N_CLASSES:
        issues.append(f"Classes manquantes dans train: {len(unique_train)}/{config.N_CLASSES}")
    
    if len(unique_val) != config.N_CLASSES:
        issues.append(f"Classes manquantes dans val: {len(unique_val)}/{config.N_CLASSES}")
    
    # Vérifier l'équilibre des classes
    min_class_count = min(counts_train)
    if min_class_count < 5:  # Seuil minimum
        issues.append(f"Classe sous-représentée: {min_class_count} échantillons minimum")
    
    # Vérifier la plage des données
    if X_train.std() < 1e-6:
        issues.append("Variance très faible dans les features")
    
    if len(issues) > 0:
        print(f"⚠️  Split {split_num} - Problèmes détectés:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True

def train_single_fold(args):
    """
    Entraîne un seul fold - fonction isolée pour le multiprocessing
    """
    (split_num, start_date, df_roll, features_list, model_builder_fn, price_col) = args
    
    # Configuration TensorFlow plus conservative pour multiprocessing
    # ⚠️ IMPORTANT: Configuration minimaliste pour éviter les conflits
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'  # Un seul thread par processus
    
    # Seed unique par processus
    process_seed = config.SEED + split_num
    np.random.seed(process_seed)
    tf.random.set_seed(process_seed)
    
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
        
        # Vérification de taille minimum plus robuste
        min_samples = max(config.BATCH_SIZE * 2, 50)  # Au moins 2 batchs ou 50 échantillons
        if X_train_final.shape[0] < min_samples or X_va_seq.shape[0] < config.BATCH_SIZE:
            print(f"Warning: Not enough samples in train ({X_train_final.shape[0]}) or val ({X_va_seq.shape[0]}) for split {split_num}. Skipping.")
            return None

        # 🔍 DIAGNOSTIC CRITIQUE: Vérifier la qualité des données
        if not diagnose_data_quality(X_train_final, y_train_final, X_va_seq, y_va_seq, split_num):
            print(f"Data quality issues detected for split {split_num}. Skipping.")
            return None

        # Class weights avec gestion d'erreur
        try:
            if len(np.unique(y_train_final)) < config.N_CLASSES:
                print(f"Warning: Not all classes present in training data for split {split_num}. Using uniform weights.")
                class_weights = None 
            else:
                cw = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
                class_weights = dict(zip(np.unique(y_train_final), cw))
                # Limiter les poids extrêmes
                max_weight = 10.0
                class_weights = {k: min(v, max_weight) for k, v in class_weights.items()}
        except Exception as e:
            print(f"Error computing class weights for split {split_num}: {e}. Using uniform weights.")
            class_weights = None

        # Model avec gestion d'erreur
        try:
            model = model_builder_fn(
                input_shape=(config.TIMESTEPS, X_train_final.shape[2]),
                n_classes=config.N_CLASSES
            )
        except Exception as e:
            print(f"Error building model for split {split_num}: {e}")
            return None
        
        # ⚠️ EARLY STOPPING PLUS PERMISSIF
        early_stopping = EarlyStopping(
            patience=config.PATIENCE,  # Au minimum 15 epochs de patience
            restore_best_weights=True,
            monitor='val_loss',
            mode='min',
            min_delta=0.001,  # Seuil plus fin pour détecter les améliorations
            verbose=1
        )
        
        # Callback pour surveiller l'entraînement
        class TrainingMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch == 0:  # Premier epoch
                    if logs.get('val_loss', float('inf')) > 0.9:
                        print(f"⚠️  Split {split_num} - High initial val_loss: {logs.get('val_loss', 0):.4f}")
                    if logs.get('loss', float('inf')) > 0.9:
                        print(f"⚠️  Split {split_num} - High initial train_loss: {logs.get('loss', 0):.4f}")
        
        training_monitor = TrainingMonitor()
        
        print(f"Training model for split {split_num} with improved configuration...")
        
        # 🔧 PARAMÈTRES D'ENTRAÎNEMENT AMÉLIORÉS
        history = model.fit(
            X_train_final, y_train_final,
            epochs=config.EPOCHS,
            batch_size=max(16, min(config.BATCH_SIZE, X_train_final.shape[0] // 4)),  # Batch size adaptatif
            validation_data=(X_va_seq, y_va_seq),
            callbacks=[early_stopping, training_monitor],
            class_weight=class_weights,
            verbose=1,  # Afficher le progrès pour diagnostic
            shuffle=True,
            validation_freq=1
        )

        # Vérifier si l'entraînement s'est bien passé
        if len(history.history['loss']) <= 2:
            print(f"⚠️  Split {split_num} - Training stopped too early ({len(history.history['loss'])} epochs)")
            return None

        best_epoch = np.argmin(history.history['val_loss'])
        best_val_loss = min(history.history['val_loss'])
        train_loss_at_best = history.history['loss'][best_epoch]
        
        print(f"Split {split_num} - Training completed:")
        print(f"  Best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        print(f"  Train_loss at best epoch: {train_loss_at_best:.4f}")
        print(f"  Total epochs: {len(history.history['loss'])}")

        # Vérifier si le modèle a appris quelque chose
        if best_val_loss > 0.95:  # Seuil d'alerte
            print(f"⚠️  Split {split_num} - Suspicious high val_loss: {best_val_loss:.4f}")

        # Predictions avec gestion d'erreur
        try:
            y_pred_va_probs = model.predict(X_va_seq, verbose=0)
            y_pred_va = np.argmax(y_pred_va_probs, axis=1)
            
            y_pred_te_probs = model.predict(X_te_seq, verbose=0)
            y_pred_te = np.argmax(y_pred_te_probs, axis=1)
        except Exception as e:
            print(f"Error during prediction for split {split_num}: {e}")
            return None

        # Metrics
        val_metrics = evaluation.calculate_classification_metrics(y_va_seq, y_pred_va)
        f1_val = val_metrics["F1"]
        test_cls_metrics = evaluation.calculate_classification_metrics(y_te_seq, y_pred_te)
        
        # PnL Metrics avec meilleure gestion d'erreur
        try:
            if len(te_df_processed) > config.TIMESTEPS and 'ret_future' in te_df_processed.columns:
                returns_for_pnl_test = te_df_processed['ret_future'].iloc[config.TIMESTEPS-1:config.TIMESTEPS-1+len(y_te_seq)]
                
                if len(returns_for_pnl_test) != len(y_te_seq):
                    print(f"⚠️  Split {split_num} - Length mismatch: returns={len(returns_for_pnl_test)}, predictions={len(y_te_seq)}")
                
                if returns_for_pnl_test.isna().any():
                    print(f"Split {split_num} - Warning: {returns_for_pnl_test.isna().sum()} NaNs in returns. Filling with 0.")
                    returns_for_pnl_test = returns_for_pnl_test.fillna(0)
                
                pnl_test, sharpe_test = evaluation.calculate_pnl_sharpe_metrics(
                    y_te_seq, y_pred_te, returns_for_pnl_test, config.PRED_HORIZON
                )
            else:
                pnl_test, sharpe_test = 0.0, 0.0
                print(f"Split {split_num} - Warning: Not enough data for PnL calculation.")
        except Exception as e:
            print(f"Error calculating PnL for split {split_num}: {e}")
            pnl_test, sharpe_test = 0.0, 0.0

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
            "epochs_trained": len(history.history['loss']),
            "best_val_loss": best_val_loss,
            "train_loss_at_best": train_loss_at_best
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
        
        print(f"✅ Split {split_num} results: F1_val={f1_val:.4f}, F1_test={test_cls_metrics['F1']:.4f}, PnL_test={pnl_test:.4f}, Sharpe_test={sharpe_test:.2f}")
        
        return fold_result, model_data, f1_val
        
    except Exception as e:
        print(f"❌ Error in split {split_num}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_walk_forward_parallel(df_roll: pd.DataFrame,
                               features_list: list,
                               model_builder_fn,
                               price_col: str = config.TARGET_PRICE_COL,
                               max_workers: int = None):
    """
    Version parallélisée améliorée de train_walk_forward
    """
    
    # Déterminer le nombre optimal de workers - PLUS CONSERVATEUR
    if max_workers is None:
        cpu_count = psutil.cpu_count()
        # Réduire encore plus le parallélisme pour éviter les conflits TensorFlow
        max_workers = max(1, min(cpu_count // 4, 2))  # Max 2 processus parallèles
    
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
    
    # 🔧 OPTION 1: Exécution séquentielle pour le debug (recommandé initialement)
    print("🔧 Running sequentially for better debugging...")
    for args in fold_args:
        result = train_single_fold(args)
        if result is not None:
            fold_result, model_data, f1_val = result
            results.append(fold_result)
            
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_model_set = model_data
                print(f"*** New best model found from split {args[0]} with F1_val: {best_f1_val:.4f} ***")
    
    # 🔧 OPTION 2: Exécution parallèle (à activer une fois le debug terminé)
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_split = {
            executor.submit(train_single_fold, args): args[0] 
            for args in fold_args
        }
        
        for future in as_completed(future_to_split):
            split_num = future_to_split[future]
            try:
                result = future.result()
                if result is not None:
                    fold_result, model_data, f1_val = result
                    results.append(fold_result)
                    
                    if f1_val > best_f1_val:
                        best_f1_val = f1_val
                        best_model_set = model_data
                        print(f"*** New best model found from split {split_num} with F1_val: {best_f1_val:.4f} ***")
                        
            except Exception as e:
                print(f"Split {split_num} generated an exception: {e}")
    """
    
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