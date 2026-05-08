import os
import sys
import pandas as pd
import numpy as np
from pipeline.config import *
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

from pipeline.feature_selection import select_top_features_pca, select_top_features_shap
from pipeline.tf_metrics import *
from pipeline.backtest import run_backtest

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from focal_loss import SparseCategoricalFocalLoss

"""
Ce script constitue le pipeline complet d'entraînement et d'évaluation d'un modèle LSTM 
pour la classification du S&P500 en 3 classes (baisse, neutre, hausse) à partir de données temporelles enrichies.

Fonctionnalités principales :
1. Chargement, nettoyage et séparation temporelle des données (train, validation, holdout).
2. Création de la target via le rendement futur à un horizon défini (`ret_future`) et labellisation par seuils.
3. Option de sélection de features via SHAP (importance sur LightGBM).
4. Préparation des séquences temporelles avec `TimeseriesGenerator`.
5. Entraînement du modèle LSTM (avec Focal Loss pondérée) sur des séquences glissantes.
6. Option de validation croisée temporelle avec reporting détaillé (accuracy, f1, balanced acc).
7. Évaluation finale sur l’ensemble holdout avec seuil de confiance ajustable (`ECART_MIN`).
8. Impression des métriques, des prédictions par classe, et des dates de fenêtre de prédiction.

Ce pipeline permet de tester de manière robuste des modèles séquentiels pour la prévision de mouvement de marché 
en tenant compte de la temporalité, des déséquilibres de classe, et d’une logique de confiance dans la décision.
"""


# ==================== REPRODUCTIBILITÉ ====================

# Fixer toutes les graines aléatoires
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration TensorFlow pour la reproductibilité
tf.config.experimental.enable_op_determinism()

# Afficher plus de lignes et colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


def create_results_dir():
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)
    run_id = datetime.now().strftime(f"model_v{MODEL_VERSION}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_training_artifacts(history_obj, output_dir, prefix, save_history_csv=True):
    history_df = pd.DataFrame(history_obj.history)
    if save_history_csv:
        history_csv_path = os.path.join(output_dir, f"{prefix}_history.csv")
        history_df.to_csv(history_csv_path, index=False)

    metric_groups = [
        ("loss", "val_loss", "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("f1_macro", "val_f1_macro", "F1 Macro"),
        ("balanced_accuracy", "val_balanced_accuracy", "Balanced Accuracy"),
    ]

    available_groups = []
    for train_metric, val_metric, title in metric_groups:
        if train_metric in history_df.columns or val_metric in history_df.columns:
            available_groups.append((train_metric, val_metric, title))

    if not available_groups:
        return

    n_plots = len(available_groups)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (train_metric, val_metric, title) in enumerate(available_groups):
        ax = axes[idx]
        if train_metric in history_df.columns:
            ax.plot(history_df[train_metric], label=train_metric)
        if val_metric in history_df.columns:
            ax.plot(history_df[val_metric], label=val_metric)
        ax.set_title(f"{title} - {prefix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(cm, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_prediction_distribution_plot(y_true, y_pred, output_path):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    true_counts = [np.sum(y_true == c) for c in classes]
    pred_counts = [np.sum(y_pred == c) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, true_counts, width, label="True")
    plt.bar(x + width / 2, pred_counts, width, label="Pred")
    plt.xticks(x, classes)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("True vs Predicted class distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# Définir l'architecture du modèle
def lstm_model_v2(input_shape, n_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.05),
                   recurrent_dropout=0.2,
                   ))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model

# Charger les données CSV
def load_data(debug_on):
    print("🔹 1) Chargement des données depuis :", DATA_FILE_PATH)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    print("   → Aperçu des 5 premières lignes :")
    print(df.head())

    if debug_on:
        print("   [DEBUG] Vérifiez le DataFrame chargé.")

    return df

# Créer la cible
def create_target(df, tag="", debug_on=False):
    print(f"🔹 2) Création de la cible (horizon = {PRED_HORIZON}) {tag}")
    df['ret_future'] = (df[TARGET_PRICE_COL].shift(-PRED_HORIZON) - df[TARGET_PRICE_COL]) / df[TARGET_PRICE_COL]
    print("   → Avant suppression des NaN, nombre de lignes :", len(df))
    df.dropna(subset=['ret_future'], inplace=True)
    print("   → Après suppression, nombre de lignes :", len(df))
    print("   → Aperçu ret_future :")
    print(df[[DATE_COL, TARGET_PRICE_COL, 'ret_future']].head(10))

    if debug_on:
        print("   [DEBUG] Vérifiez ret_future.")

    if THRESHOLD_STRATEGY == "fixed":
        thresholds = FIXED_THRESHOLDS
        print(f"   → Seuils utilisés pour labellisation : {thresholds}")
        def label_target(x):
            if x < thresholds[0]:
                return 0
            elif x <= thresholds[1]:
                return 1
            else:
                return 2
        df["target"] = df["ret_future"].apply(label_target)
        print("   → Répartition des classes après labellisation :")
        print(df["target"].value_counts(normalize=True))
    else:
        raise ValueError(f"Stratégie de seuil non reconnue : {THRESHOLD_STRATEGY}")

    if debug_on:
        print("   [DEBUG] Vérifiez la colonne 'target'.")

    return df


# Fonction principale du pipeline (CV ou non, avec ou sans sélection de features, debug ou non)
def main(cross_validation, debug_on, feature_selection_on):
    results_dir = create_results_dir()
    print(f"🔹 Dossier de sortie des résultats : {results_dir}")

    # -- Chargement et split --
    df_raw = load_data(debug_on)
    # Supprimer certaines colonnes inutiles à la prédiction
    cols_to_drop = ["sp500_prev_close", "sp500_return_1d", "vix_direction", "vix_high"]
    df_raw.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    print(f"🔹 Colonnes restantes : {df_raw.columns.tolist()}")
    print(f"🔹 Total lignes après nettoyage : {len(df_raw)}")

    if debug_on:
        print("   [DEBUG] Vérifiez le nettoyage initial.")

    # Création des splits temporels (train vs holdout)
    df_holdout = df_raw[(df_raw[DATE_COL] >= HOLDOUT_START_DATE) & (df_raw[DATE_COL] <= HOLDOUT_END_DATE)].copy()
    df_main    = df_raw[df_raw[DATE_COL] < HOLDOUT_START_DATE].copy()
    print(f"🔹 Lignes pour entraînement (avant target) : {len(df_main)}")
    print(f"🔹 Lignes pour holdout (avant target)    : {len(df_holdout)}")

    if debug_on:
        print("   [DEBUG] Vérifiez les splits temporels.")

    df_main    = create_target(df_main, tag="(train)", debug_on=debug_on)
    df_holdout = create_target(df_holdout, tag="(holdout)", debug_on=debug_on)

    # -- Distribution des classes --
    print("🔹 Distribution classes TRAIN :")
    print(df_main["target"].value_counts(normalize=True))
    print("🔹 Distribution classes HOLDOUT :")
    print(df_holdout["target"].value_counts(normalize=True))
    if debug_on:
        print("   [DEBUG] Vérifiez distributions.")

    # -- Cross-validation ou pipeline complet --
    if cross_validation:
        print("🔹 Mode Validation Croisée activé")
        # Initialisation de la validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        fold_stats = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_main)):
            print(f"\n========== FOLD {fold+1} ==========")
            df_train = df_main.iloc[train_idx].copy()
            df_val   = df_main.iloc[val_idx].copy()
            date_train = df_train[DATE_COL].values
            date_val   = df_val[DATE_COL].values
            print(f"👉 Train: {len(df_train)}, Val: {len(df_val)}")
            if debug_on:
                print("   [DEBUG] Vérifiez les indexes de split.")
            
            # Appliquer SHAP pour sélectionner les top N features (sur df_train)
            if feature_selection_on:
                # Sélection de features sur df_train uniquement
                print("🔹 Sélection features SHAP...")
                raw_features = select_top_features_shap(df_train, top_n=TOP_N_FEATURES, target_col="ret_future")
                features = [f for f in raw_features if f not in (DATE_COL, 'target', 'ret_future')]
                if 'SP500_historical_data_Close' not in features:
                    features.append('SP500_historical_data_Close')
                print(f"   → features finales utilisées : {features}")

                if debug_on:
                    print("   [DEBUG] Vérifiez features.")

            else:
                features = [f for f in df_train.columns if f not in (DATE_COL, 'target', 'ret_future')]

            # Préparation X/y
            scaler = StandardScaler()
            X_train = scaler.fit_transform(df_train[features])
            X_val   = scaler.transform(df_val[features])
            y_train = df_train['target'].to_numpy()
            y_val   = df_val['target'].to_numpy()
            print("🔹 Shape X_train, X_val, y_train, y_val :", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            
            if debug_on:
                print("   [DEBUG] Vérifiez les shapes.")

            # Générateurs
            train_gen = TimeseriesGenerator(X_train, y_train, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, stride=STRIDE)
            val_gen   = TimeseriesGenerator(X_val,   y_val,   length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, stride=STRIDE)
            print("🔹 Nombre de batches train/val :", len(train_gen), len(val_gen))

            # Calcul des poids de classes (balanced × boost manuel pour renforcer les classes rares)
            cw_raw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            cw = np.array([cw_raw[c] * CLASS_WEIGHT_BOOST.get(c, 1.0) for c in range(N_CLASSES)])

            cw_dict = dict(enumerate(cw))

            print("🔹 Class weights pli :", cw_dict)
            
            if debug_on:
                print("   [DEBUG] Vérifiez les generators.")

            # Modèle
            model = lstm_model_v2((SEQUENCE_LENGTH, len(features)), N_CLASSES)
            metrics = ['accuracy', F1Macro(3), BalancedAcc(3)]

            gamma = 2.0

            loss_fn = SparseCategoricalFocalLoss(gamma=gamma, class_weight=cw)

            model.compile(optimizer='adam',
                        loss=loss_fn,
                        metrics=metrics)

            print("🔹 Modèle compilé. Architecture :")
            model.summary()

            if debug_on:
                print("   [DEBUG] Vérifiez l'architecture.")

            # Entraînement avec early stopping basé sur F1 macro
            early_stop = EarlyStopping(
            monitor='val_f1_macro',   # ou 'val_balanced_accuracy'
            mode='max',
            patience=PATIENCE,
            restore_best_weights=True)

            history_cv = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                                   callbacks=[early_stop], verbose=1)

            save_training_artifacts(history_cv, results_dir, f"cv_fold_{fold+1}")
            
            if debug_on:
                print("   [DEBUG] Entraînement terminé.")

           # Prédictions sur les données de validation
            y_proba = model.predict(val_gen)
            # On reconstruit y_true de la même façon que plus bas
            y_true = np.concatenate([val_gen[i][1] for i in range(len(val_gen))])
            y_proba = y_proba[:len(y_true)]

            # Exemples de dates + prédictions
            print(f"🔹 Quelques exemples de fenêtres et dates (Fold {fold+1}):")
            for j in range(min(3, len(y_true))):
                start_date      = pd.to_datetime(date_val[j])
                end_input_date  = pd.to_datetime(date_val[j + SEQUENCE_LENGTH - 1])
                pred_date       = pd.to_datetime(date_val[j + SEQUENCE_LENGTH + PRED_HORIZON - 1])
                print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}, y_true={y_true[j]}")
            
            if debug_on:
                print("   [DEBUG] Vérifiez dates des fenêtres CV.")

            # Conversion des proba → classes avec logique de seuil de confiance
            ecart_min = ECART_MIN
            y_pred = []
            for p in y_proba:
                top2 = np.sort(p)[::-1][:2]
                y_pred.append(np.argmax(p) if (top2[0]-top2[1]>=ecart_min) else 1)
            y_pred = np.array(y_pred)

            # Résumé des performances sur chaque fold
            acc     = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1_m    = f1_score(y_true, y_pred, average='macro')
            f1_w    = f1_score(y_true, y_pred, average='weighted')
            print(f"   → Fold {fold+1} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | F1-macro: {f1_m:.4f} | F1-weighted: {f1_w:.4f}")
            fold_stats.append((acc, bal_acc, f1_m, f1_w))

        # Résumé CV
        accs, bals, f1s, f1w = zip(*fold_stats)
        print("\n===== RÉSULTATS CV =====")
        print(f"Acc Moyenne: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"BalAcc Moyenne: {np.mean(bals):.4f} ± {np.std(bals):.4f}")
        print(f"F1-macro Moyenne: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"F1-weighted Moyenne: {np.mean(f1w):.4f} ± {np.std(f1w):.4f}")

        cv_df = pd.DataFrame(
            fold_stats,
            columns=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
        )
        cv_df.insert(0, "fold", np.arange(1, len(cv_df) + 1))
        cv_df.to_csv(os.path.join(results_dir, "cv_fold_metrics.csv"), index=False)
        
    if debug_on:
        print("   [DEBUG] Fin validation croisée.")


    # Finalisation avec entraînement sur tout df_main (pipeline complet)
    if feature_selection_on:
        # -- Pipeline final sur holdout --
        print("\n🔹 Sélection finale des features sur tout df_main")
        raw_feats = select_top_features_shap(df_main, top_n=TOP_N_FEATURES, target_col="ret_future")
        final_feats = [f for f in raw_feats if f not in (DATE_COL, 'target', 'ret_future')]
        if 'SP500_historical_data_Close' not in final_feats:
            final_feats.append('SP500_historical_data_Close')
        print("   → Features finales :", final_feats)

    else:
        final_feats = [f for f in df_main.columns if f not in ('target', 'ret_future')]
    
    if debug_on:
        print("   [DEBUG] Vérifiez les features finales.")

    # --- 1. Division des données en Train / Validation / Test ---
    print("🔹 Division chronologique des données...")

    # Définir le point de séparation pour un split 80/20 sur df_main.
    split_idx = int(0.8 * len(df_main))

    # Diviser df_main en ensembles d'entraînement et de validation
    train_df = df_main.iloc[:split_idx]
    val_df = df_main.iloc[split_idx:]

    # df_holdout est notre ensemble de test final
    test_df = df_holdout

    print(f"Shapes des DataFrames : Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")

    # --- 2. Préparation des X/y et mise à l'échelle ---
    X_train = train_df[final_feats]
    y_train = train_df['target'].to_numpy()

    X_val = val_df[final_feats]
    y_val = val_df['target'].to_numpy()

    X_test = test_df[final_feats]
    y_test = test_df['target'].to_numpy()
    date_test = test_df[DATE_COL].values

    # Normalisation : .fit() uniquement sur le train, .transform() sur tous.
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_val_scaled   = final_scaler.transform(X_val)
    X_test_scaled  = final_scaler.transform(X_test)

    # Les noms sont plus clairs que X_tr, X_te, etc.
    print("🔹 Shapes finales X (scaled) :", X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)
    print("🔹 Shapes finales y :", y_train.shape, y_val.shape, y_test.shape)

    if debug_on:
        print("   [DEBUG] Vérifiez les données train/val/test.")

    # --- 3. Création des TimeseriesGenerators ---
    gen_train = TimeseriesGenerator(X_train_scaled, y_train, length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, stride=STRIDE)
    gen_val   = TimeseriesGenerator(X_val_scaled,   y_val,   length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, stride=STRIDE)
    gen_test  = TimeseriesGenerator(X_test_scaled,  y_test,  length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, stride=1)

    print("🔹 Batches finaux train/validation/test :", len(gen_train), len(gen_val), len(gen_test))

    # Vérification des séquences
    print(f"Vérification : première séquence a {gen_train[0][0].shape[1]} pas temporels")
    assert gen_train[0][0].shape[1] == SEQUENCE_LENGTH, "Incohérence dans la longueur des séquences"

    cw_raw_f = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_f = np.array([cw_raw_f[c] * CLASS_WEIGHT_BOOST.get(c, 1.0) for c in range(N_CLASSES)])
    cw_dict_f = dict(enumerate(cw_f))
    print("Class weights final :", cw_dict_f)

    if debug_on:
        print("   [DEBUG] Vérifiez les generators finaux.")

    # Entraînement final
    final_model = lstm_model_v2((SEQUENCE_LENGTH, len(final_feats)), N_CLASSES)
    metrics = ['accuracy', F1Macro(3), BalancedAcc(3)]

    gamma = 2.0
    loss_fn = SparseCategoricalFocalLoss(gamma=gamma, class_weight=cw_f)

    final_model.compile(optimizer='adam',
                        loss=loss_fn,
                        metrics=metrics)

    early_stop = EarlyStopping(
        monitor='val_f1_macro',
        mode='max',
        patience=PATIENCE,
        restore_best_weights=True)

    history_final = final_model.fit(gen_train,
                                    validation_data=gen_val,
                                    epochs=EPOCHS,
                                    callbacks=[early_stop], verbose=1)

    save_training_artifacts(history_final, results_dir, "final_train", save_history_csv=True)

    model_path = os.path.join(results_dir, f"model_v{MODEL_VERSION}.keras")
    final_model.save(model_path)
    print(f"🔹 Modèle sauvegardé : {model_path}")

    if debug_on:
        print("   [DEBUG] Entraînement final OK.")

    # Optimisation de ECART_MIN sur la validation set
    print("\n🔹 Optimisation du seuil de confiance ECART_MIN sur val...")
    proba_val_opt = final_model.predict(gen_val)
    y_true_val_opt = np.concatenate([gen_val[i][1] for i in range(len(gen_val))])
    proba_val_opt = proba_val_opt[:len(y_true_val_opt)]
    best_ecart, best_f1 = ECART_MIN, 0.0
    for threshold in np.arange(0.0, 0.5, 0.01):
        preds = []
        for p in proba_val_opt:
            s = np.sort(p)[::-1]
            preds.append(int(np.argmax(p)) if (s[0] - s[1]) >= threshold else 1)
        score = f1_score(y_true_val_opt, preds, average='macro', zero_division=0)
        if score > best_f1:
            best_f1, best_ecart = score, threshold
    print(f"   → ECART_MIN optimal : {best_ecart:.2f} (F1-macro val = {best_f1:.4f}, valeur config = {ECART_MIN})")
    ecart_used = best_ecart

    # Évaluation holdout
    print("\n--- Évaluation HOLDOUT ---")
    proba_final = final_model.predict(gen_test)
    y_true_final = np.concatenate([gen_test[i][1] for i in range(len(gen_test))])
    proba_final = proba_final[:len(y_true_final)]

    # Exemples de dates + prédictions (Hold-Out)
    print("🔹 Quelques exemples de fenêtres et dates (Hold-Out):")
    for j in range(min(3, len(y_true_final))):
        start_date     = pd.to_datetime(date_test[j])
        end_input_date = pd.to_datetime(date_test[j + SEQUENCE_LENGTH - 1])
        pred_date      = pd.to_datetime(date_test[j + SEQUENCE_LENGTH + PRED_HORIZON - 1])
        print(f"  Sample {j}: début={start_date.date()}, fin_input={end_input_date.date()}, date_prediction={pred_date.date()}, y_true={y_true_final[j]}")
        
    if debug_on:
        print("   [DEBUG] Vérifiez dates des fenêtres HoldOut.")

    # Transformation probabilités → prédictions avec logique de confiance
    ecart_min = ecart_used
    y_pred_final_list = []
    for proba in proba_final:
        sorted_proba = np.sort(proba)[::-1]
        if sorted_proba[0] - sorted_proba[1] >= ecart_min:
            y_pred_final_list.append(np.argmax(proba))
        else:
            y_pred_final_list.append(1) # Classe neutre par défaut pour incertitude (à adapter, 0, 1 ou 2)
    y_pred_final = np.array(y_pred_final_list)

    # Calcul des métriques de classification
    acc_f  = accuracy_score(y_true_final, y_pred_final)
    bal_f  = balanced_accuracy_score(y_true_final, y_pred_final)
    f1m_f  = f1_score(y_true_final, y_pred_final, average='macro')
    f1w_f  = f1_score(y_true_final, y_pred_final, average='weighted')
    print(f"   → Accuracy final : {acc_f:.4f}")
    print(f"   → Balanced Acc. : {bal_f:.4f}")
    print(f"   → F1-macro      : {f1m_f:.4f}")
    print(f"   → F1-weighted      : {f1w_f:.4f}")
    classif_report_text = classification_report(y_true_final, y_pred_final, zero_division=0)
    classif_report_dict = classification_report(y_true_final, y_pred_final, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_final, y_pred_final)
    print(classif_report_text)
    print(cm)

    print(f"Nombre de classe 0 prédite (y_pred_final): {(y_pred_final == 0).sum()}")
    print(f"Nombre de classe 0 réelle (y_true_final): {(y_true_final == 0).sum()}")

    print(f"Nombre de classe 1 prédite (y_pred_final): {(y_pred_final == 1).sum()}")
    print(f"Nombre de classe 1 réelle (y_true_final): {(y_true_final == 1).sum()}")

    print(f"Nombre de classe 2 prédite (y_pred_final): {(y_pred_final == 2).sum()}")
    print(f"Nombre de classe 2 réelle (y_true_final): {(y_true_final == 2).sum()}")

    pred_dates = pd.to_datetime(date_test[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(y_true_final)])
    sorted_proba_final = np.sort(proba_final, axis=1)[:, ::-1]
    pred_df = pd.DataFrame({
        "date_prediction": pred_dates,
        "y_true": y_true_final,
        "y_pred": y_pred_final,
        "proba_0": proba_final[:, 0],
        "proba_1": proba_final[:, 1],
        "proba_2": proba_final[:, 2],
        "top_proba": sorted_proba_final[:, 0],
        "second_proba": sorted_proba_final[:, 1],
        "confidence_gap": sorted_proba_final[:, 0] - sorted_proba_final[:, 1],
    })
    pred_df.to_csv(os.path.join(results_dir, "holdout_predictions.csv"), index=False)

    metrics_df = pd.DataFrame([
        {
            "model_version": MODEL_VERSION,
            "ecart_min": ECART_MIN,
            "accuracy": acc_f,
            "balanced_accuracy": bal_f,
            "f1_macro": f1m_f,
            "f1_weighted": f1w_f,
            "n_holdout_samples": len(y_true_final),
        }
    ])
    metrics_df.to_csv(os.path.join(results_dir, "holdout_metrics.csv"), index=False)

    report_df = pd.DataFrame(classif_report_dict).transpose()
    report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))

    save_confusion_matrix_plot(cm, os.path.join(results_dir, "confusion_matrix.png"))
    save_prediction_distribution_plot(
        y_true_final,
        y_pred_final,
        os.path.join(results_dir, "class_distribution_true_vs_pred.png"),
    )

    # Backtest financier sur le holdout
    price_series = test_df.set_index(DATE_COL)[TARGET_PRICE_COL]
    run_backtest(pred_df, price_series, results_dir)

    print("🔹 Sauvegardes terminées dans le dossier résultats :")
    print(f"   → {results_dir}")


    if debug_on:
        print("   [DEBUG] Vérifiez le rapport de classification.")


if __name__ == "__main__":
    cross_validation = False  # False pour pipeline complet sans CV
    debug_on = False

    feature_selection_on = True # True avec feature selection, False sans

    print(f"\n--- Lancement pipeline version : {MODEL_VERSION} ---")
    if os.path.exists(f"Pipeline/model/model_v{MODEL_VERSION}.keras"):
        print("⚠️  Le modèle existe déjà :", f"model_v{MODEL_VERSION}.keras")
        print("   (Continuera automatiquement. Ctrl+C pour annuler.)")

    main(cross_validation, debug_on, feature_selection_on)
