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
    # Pour 3 classes, on veut 2 seuils, donc k=3.
    # np.linspace(0, 1, k + 1) donne [0, 0.33, 0.66, 1] si k=3
    # On prend les quantiles à q[1] et q[2] (soit 0.33 et 0.66 par défaut)
    # Si vous voulez que les seuils séparent les terciles (33%/33%/33%), c'est correct.
    # Si vous voulez des seuils plus extrêmes (ex: 5% / 90% / 5%), ajustez les quantiles.
    # Par défaut, on utilise N_CLASSES pour définir les quantiles
    q = np.linspace(0, 1, k + 1) # k est N_CLASSES ici
    
    thresholds = series.quantile(q).values[1:-1]
    
    # Optionnel: Ajouter des contraintes si vous le souhaitez
    # if len(thresholds) == 2: # Pour 3 classes
    #     thresholds[0] = min(thresholds[0], -0.001)  # S'assurer que le seuil bas est négatif
    #     thresholds[1] = max(thresholds[1], 0.001)   # S'assurer que le seuil haut est positif
    #     if thresholds[0] >= thresholds[1]: # S'assurer que seuil_bas < seuil_haut
    #         # Gérer ce cas: peut-être utiliser des valeurs par défaut ou une fraction de l'std
    #         std_dev = series.std()
    #         thresholds[0] = -0.5 * std_dev
    #         thresholds[1] = 0.5 * std_dev
    return thresholds


def label_from_thresholds(ret_value: float, thresholds: np.ndarray) -> int:
    """Labels a return value based on a set of thresholds."""
    if thresholds is None or len(thresholds) == 0:
        return 0 # Ou np.nan si vous préférez gérer les erreurs plus tard
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
    # 'ret_future' est déjà calculé dans preprocess_fold_data avant d'appeler cette fonction
    # df_copy['ret_future'] = compute_future_return(df_copy[price_col], horizon)
    # df_copy = df_copy.dropna(subset=['ret_future']) # Déjà géré aussi
    
    thresholds_map = OrderedDict()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    start_date = df_copy[date_col].min()
    end_date = df_copy[date_col].max()
    
    current_window_start = start_date
    # S'assurer qu'il y a assez de données pour au moins une fenêtre
    if start_date + relativedelta(months=step_months) > end_date and len(df_copy) >= 100 :
        thresholds = make_thresholds(df_copy["ret_future"], n_classes)
        thresholds_map[(start_date, end_date + pd.Timedelta(days=1))] = thresholds
        print(f"Single Window (due to short data span): {start_date} to {end_date}, Thresholds: {thresholds}")
        return thresholds_map

    while current_window_start + relativedelta(months=step_months) <= end_date:
        window_end = current_window_start + relativedelta(months=step_months)
        window_df = df_copy[(df_copy[date_col] >= current_window_start) & (df_copy[date_col] < window_end)]
        
        if len(window_df) >= 100:
            thresholds = make_thresholds(window_df["ret_future"], n_classes)
            # Vérification pour éviter les seuils problématiques (optionnel mais recommandé)
            if len(thresholds) == (n_classes -1) and (n_classes < 2 or np.all(np.diff(thresholds) > 0)): # S'assure que les seuils sont ordonnés
                 thresholds_map[(current_window_start, window_end)] = thresholds
                 print(f"Window: {current_window_start.date()} to {window_end.date()}, Thresholds: {thresholds}")
            elif thresholds_map: # Si seuils invalides et map non vide, utiliser les derniers bons seuils
                thresholds_map[(current_window_start, window_end)] = list(thresholds_map.values())[-1]
                print(f"Window: {current_window_start.date()} to {window_end.date()}, Used previous thresholds due to issue.")
            # else: On pourrait ici définir des seuils par défaut globaux si c'est la première fenêtre et qu'elle est problématique

        advance_months = max(1, step_months // 2 if step_months > 1 else 1) # Assure au moins 1 mois d'avancement
        current_window_start += relativedelta(months=advance_months)
    
    if not thresholds_map and len(df_copy) >= 100: # Fallback global si aucune fenêtre n'a pu être traitée
        thresholds = make_thresholds(df_copy["ret_future"], n_classes)
        thresholds_map[(start_date, end_date + pd.Timedelta(days=1))] = thresholds
        print(f"Global Fallback Window: {start_date.date()} to {end_date.date()}, Thresholds: {thresholds}")
    elif not thresholds_map:
        print("Warning: No thresholds could be generated due to insufficient data or windowing issues.")
        # Retourner une map vide, qui sera gérée par label_with_sliding_thresholds
        return thresholds_map


    # Gestion de la dernière période si elle n'est pas couverte
    # ou si la dernière clé de la map ne va pas jusqu'à end_date
    last_map_end_date = list(thresholds_map.keys())[-1][1] if thresholds_map else start_date - pd.Timedelta(days=1)

    if last_map_end_date < end_date + pd.Timedelta(days=1):
        # Essayer de créer une fenêtre finale
        final_window_start = max(start_date, end_date - relativedelta(months=step_months))
        final_window_df = df_copy[df_copy[date_col] >= final_window_start]

        final_key = (final_window_start, end_date + pd.Timedelta(days=1))

        if len(final_window_df) >= 100:
            thresholds = make_thresholds(final_window_df["ret_future"], n_classes)
            if len(thresholds) == (n_classes-1) and (n_classes < 2 or np.all(np.diff(thresholds) > 0)):
                thresholds_map[final_key] = thresholds
            elif thresholds_map: # Si seuils invalides et map non vide
                 thresholds_map[final_key] = list(thresholds_map.values())[-1] # Utiliser les derniers bons seuils
        elif thresholds_map: # Pas assez de données pour la fenêtre finale, mais d'autres existent
            thresholds_map[final_key] = list(thresholds_map.values())[-1]

    return thresholds_map


def label_with_sliding_thresholds(df: pd.DataFrame,
                                  threshold_source, # Peut être OrderedDict (adaptatif) ou np.ndarray (fixe)
                                  date_col: str = config.DATE_COL):
    """Applies labels to a DataFrame using a pre-computed thresholds_map or fixed thresholds."""
    labels = []
    
    # Vérification si threshold_source est valide
    is_adaptive = isinstance(threshold_source, OrderedDict)
    is_fixed = isinstance(threshold_source, np.ndarray)

    if not is_adaptive and not is_fixed:
        print("Warning: Invalid threshold_source provided to label_with_sliding_thresholds. Returning NaNs.")
        return pd.Series([np.nan] * len(df), index=df.index)
    
    if is_adaptive and not threshold_source: # Map adaptative vide
        print("Warning: Empty thresholds_map provided. Returning NaNs.")
        return pd.Series([np.nan] * len(df), index=df.index)

    df[date_col] = pd.to_datetime(df[date_col])
    
    last_known_thresholds_adaptive = None
    if is_adaptive and threshold_source:
        last_known_thresholds_adaptive = list(threshold_source.values())[-1]

    for _, row in df.iterrows():
        date = row[date_col]
        ret_val = row["ret_future"]
        applicable_thresholds = None
        
        if pd.isna(ret_val): # Si ret_future est NaN, on ne peut pas labelliser
            labels.append(np.nan)
            continue

        if is_adaptive:
            best_match_start_date = pd.Timestamp.min.tz_localize('UTC') if date.tzinfo else pd.Timestamp.min
            found_match = False
            for (start, end), thres in threshold_source.items():
                if start <= date < end: # La date est dans l'intervalle de la fenêtre
                    applicable_thresholds = thres
                    found_match = True
                    break
                if start <= date: # Garder le plus récent qui commence avant ou à la date
                    if best_match_start_date is None or start > best_match_start_date:
                        best_match_start_date = start
                        applicable_thresholds = thres
            
            if not found_match and applicable_thresholds is None: # Si aucune fenêtre ne correspondait (ex: date avant la 1ère fenêtre)
                if threshold_source: # Utiliser les seuils de la première fenêtre disponible
                    applicable_thresholds = list(threshold_source.values())[0]
                # Si applicable_thresholds est toujours None, on utilisera last_known_thresholds_adaptive plus bas
            
            if applicable_thresholds is None: # Fallback ultime pour adaptatif
                 applicable_thresholds = last_known_thresholds_adaptive

        elif is_fixed:
            applicable_thresholds = threshold_source
        
        if applicable_thresholds is not None and len(applicable_thresholds) > 0 :
            labels.append(label_from_thresholds(ret_val, applicable_thresholds))
        else:
            # Ce cas peut arriver si threshold_source est une map vide ou si un np.ndarray vide est passé
            print(f"Warning: No applicable thresholds found for date {date}. Appending NaN.")
            labels.append(np.nan)
            
    return pd.Series(labels, index=df.index)


def create_sequences(X_data, y_data, timesteps: int = config.TIMESTEPS):
    """Creates sequences for LSTM input."""
    Xs, ys = [], []
    # On commence à timesteps pour avoir assez de données passées pour la première séquence
    for i in range(timesteps -1, len(X_data)): # Ajustement de l'indice de départ
        # La séquence X va de i-(timesteps-1) à i inclus.
        # La cible y est à l'indice i.
        # Exemple: timesteps=3. Pour i=2 (3ème élément), X=[data[0], data[1], data[2]], y=label[2]
        # Cela signifie que X_data[i-timesteps+1 : i+1]
        if i - timesteps + 1 < 0: # Protection pour les premiers éléments
            continue
        Xs.append(X_data[i - timesteps + 1 : i + 1])
        ys.append(y_data[i])
    if not Xs: # Si aucune séquence n'a pu être créée
        return np.array([]), np.array([])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def add_jitter(X_data, ratio: float = config.JITTER_RATIO):
    """Adds Gaussian noise (jitter) to the data."""
    if X_data.ndim == 2:
        std_dev = X_data.std(axis=0, keepdims=True)
    elif X_data.ndim == 3:
        std_dev = X_data.std(axis=(0,1), keepdims=True)
    else:
        raise ValueError("Input X_data must be 2D or 3D for jitter.")
    # Gérer le cas où std_dev est zéro pour certaines features
    std_dev[std_dev == 0] = 1e-8 # Remplacer par une petite valeur pour éviter la division par zéro ou bruit nul
    noise = np.random.normal(0, std_dev * ratio, X_data.shape)
    return X_data + noise

def scale_features(X_train, X_val, X_test):
    """Scales features using StandardScaler fit on training data."""
    # Gérer les cas où les ensembles sont vides ou n'ont qu'un seul échantillon
    if X_train.shape[0] < 2: # StandardScaler a besoin d'au moins 2 échantillons pour calculer la variance
        print("Warning: Training data for scaling has less than 2 samples. Scaling might be ineffective.")
        # Retourner les données non mises à l'échelle et un scaler "factice" ou non ajusté
        return X_train, X_val, X_test, StandardScaler() 

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    X_val_scaled = X_val # Initialiser au cas où X_val est vide
    if X_val.shape[0] > 0:
        X_val_scaled = scaler.transform(X_val)
    
    X_test_scaled = X_test # Initialiser au cas où X_test est vide
    if X_test.shape[0] > 0:
        X_test_scaled = scaler.transform(X_test)
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def preprocess_fold_data(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame, 
                         features_list: list,
                         threshold_strategy: str = 'adaptive', # 'adaptive' or 'fixed'
                         price_col: str = config.TARGET_PRICE_COL,
                         horizon: int = config.PRED_HORIZON,
                         n_classes: int = config.N_CLASSES,
                         timesteps: int = config.TIMESTEPS,
                         jitter_ratio: float = config.JITTER_RATIO):
    """Full preprocessing pipeline for a single walk-forward fold."""
    
    print(f"Starting preprocessing for fold with threshold_strategy: {threshold_strategy}")

    # 1. Calculer les rendements futurs pour tous les DFs
    # On le fait ici pour s'assurer qu'il est présent avant le nettoyage ou la construction des seuils
    for df_part in [tr_df, va_df, te_df]:
        df_part['ret_future'] = compute_future_return(df_part[price_col], horizon)

    # 2. Déterminer la source des seuils (adaptatif ou fixe)
    threshold_info_for_fold = None # Ce qui sera retourné (map ou array fixe)
    threshold_source_for_labeling = None # Ce qui sera passé à label_with_sliding_thresholds

    if threshold_strategy == 'adaptive':
        print("Using adaptive thresholds.")
        # Combiner les DFs du fold pour construire les seuils adaptatifs
        # On utilise tr_df et va_df pour construire les seuils, te_df est "invisible" pour cette étape
        # pour simuler une situation réelle où les seuils sont basés sur des données passées/présentes.
        # Ou, si votre approche est que les seuils sont recalculés sur l'ensemble des données disponibles *avant* le test,
        # alors incluez te_df. Pour la robustesse, il est souvent préférable de baser les seuils
        # uniquement sur les données d'entraînement (et de validation si elle est utilisée pour la calibration).
        # Ici, nous allons utiliser tr_df + va_df pour la construction des seuils adaptatifs.
        
        # Créer une copie pour éviter les SettingWithCopyWarning
        combined_df_for_thresholds = pd.concat([tr_df, va_df]).copy()
        combined_df_for_thresholds.dropna(subset=['ret_future'], inplace=True) # Important pour build_sliding_thresholds

        if not combined_df_for_thresholds.empty:
            threshold_info_for_fold = build_sliding_thresholds(
                combined_df_for_thresholds, # Utiliser tr+va pour construire les seuils
                price_col=price_col,
                date_col=config.DATE_COL, # S'assurer que DATE_COL est bien dans config
                horizon=horizon,
                n_classes=n_classes,
                step_months=config.THRESHOLD_STEP_MONTHS
            )
            threshold_source_for_labeling = threshold_info_for_fold
        else:
            print("Warning: combined_df_for_thresholds (tr+va) is empty. Cannot build adaptive thresholds.")
            # Fallback vers des seuils fixes si adaptatifs échouent ? Ou erreur ?
            # Pour l'instant, on laisse threshold_source_for_labeling à None, ce qui sera géré
            # par label_with_sliding_thresholds (retournera des NaNs pour les labels).
            # Ou, vous pourriez explicitement passer à 'fixed' ici.
            # Par exemple:
            # print("Fallback to fixed thresholds due to empty data for adaptive calculation.")
            # threshold_info_for_fold = config.FIXED_THRESHOLDS
            # threshold_source_for_labeling = config.FIXED_THRESHOLDS
            pass # Laisser threshold_source_for_labeling None si échec adaptatif


    elif threshold_strategy == 'fixed':
        print(f"Using fixed thresholds from config: {config.FIXED_THRESHOLDS}")
        threshold_info_for_fold = config.FIXED_THRESHOLDS
        threshold_source_for_labeling = config.FIXED_THRESHOLDS
    else:
        raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}. Choose 'adaptive' or 'fixed'.")

    # Si threshold_source_for_labeling est None à ce stade (ex: adaptatif a échoué et pas de fallback),
    # label_with_sliding_thresholds retournera des NaNs, qui seront ensuite supprimés.
    if threshold_source_for_labeling is None and threshold_strategy == 'adaptive':
        print("CRITICAL WARNING: Adaptive threshold calculation failed and no fallback. Labels will be NaN.")
        # Optionnel: forcer un fallback ici si vous ne voulez pas de NaNs
        # threshold_info_for_fold = config.FIXED_THRESHOLDS # Exemple de fallback
        # threshold_source_for_labeling = config.FIXED_THRESHOLDS
        # print("Forcing fallback to fixed thresholds.")


    # 3. Appliquer les labels aux DFs individuels
    for part_df in (tr_df, va_df, te_df):
        # ret_future a déjà été calculé et les NaNs non supprimés pour l'instant
        # On supprime les NaNs de ret_future *avant* de labelliser
        part_df.dropna(subset=['ret_future'], inplace=True) 
        if not part_df.empty:
            part_df['target'] = label_with_sliding_thresholds(part_df, threshold_source_for_labeling, date_col=config.DATE_COL)
        else:
            part_df['target'] = pd.Series(dtype='float64') # Colonne vide si df vide
        
        # Supprimer les lignes où le target n'a pas pu être assigné (si label_with_sliding_thresholds retourne NaN)
        part_df.dropna(subset=['target'], inplace=True)
        part_df['target'] = part_df['target'].astype(int) # Convertir en int après suppression des NaNs

    # Nettoyage final des NaNs dans les features
    tr_df.dropna(subset=features_list + ['target'], inplace=True)
    va_df.dropna(subset=features_list + ['target'], inplace=True)
    te_df.dropna(subset=features_list + ['target'], inplace=True)

    empty_data_flag = False
    if tr_df.empty:
        print("Warning: Training data (tr_df) is empty after preprocessing.")
        empty_data_flag = True
    if va_df.empty:
        # La validation peut parfois être vide pour le dernier fold si on ne fait pas attention.
        print("Warning: Validation data (va_df) is empty after preprocessing.")
        # empty_data_flag = True # Ne pas bloquer si juste la validation est vide, mais le modèle pourrait ne pas bien s'entraîner/évaluer
    if te_df.empty:
        print("Warning: Test data (te_df) is empty after preprocessing.")
        empty_data_flag = True
        
    if empty_data_flag and (tr_df.empty or te_df.empty) : # Surtout si tr_df ou te_df est vide
        print("One or more critical data splits are empty. Skipping fold.")
        return (None,) * 11 # Retourner des Nones pour tous les éléments attendus

    # 4. Mettre à l'échelle les features
    X_tr_raw = tr_df[features_list].values if not tr_df.empty else np.array([])
    X_va_raw = va_df[features_list].values if not va_df.empty else np.array([])
    X_te_raw = te_df[features_list].values if not te_df.empty else np.array([])

    # S'assurer qu'il y a des données avant de scaler
    if X_tr_raw.shape[0] == 0:
        print("Error: Training features are empty before scaling. Cannot proceed.")
        return (None,) * 11

    X_tr, X_va, X_te, scaler = scale_features(X_tr_raw, X_va_raw, X_te_raw)
    
    y_tr = tr_df['target'].values if not tr_df.empty else np.array([])
    y_va = va_df['target'].values if not va_df.empty else np.array([])
    y_te = te_df['target'].values if not te_df.empty else np.array([])

    # 5. Créer les séquences
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, timesteps)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va, timesteps)
    X_te_seq, y_te_seq = create_sequences(X_te, y_te, timesteps)
    
    # Vérifier si les séquences ne sont pas vides
    if X_tr_seq.shape[0] == 0:
        print("Warning: No sequences created for training data. Check timesteps and data length.")
        # Selon votre logique, cela pourrait être une erreur fatale pour le fold
        return (None,) * 11 
    # if X_va_seq.shape[0] == 0 and not va_df.empty : # Si va_df n'était pas vide mais que les séquences le sont
    #     print("Warning: No sequences created for validation data.")
    # if X_te_seq.shape[0] == 0 and not te_df.empty:
    #     print("Warning: No sequences created for test data.")


    # 6. Augmentation par Jitter pour les données d'entraînement
    X_train_final, y_train_final = X_tr_seq, y_tr_seq # Par défaut si pas de jitter ou si X_tr_seq vide
    if X_tr_seq.shape[0] > 0 and jitter_ratio > 0:
        X_aug_seq = add_jitter(X_tr_seq, jitter_ratio)
        y_aug_seq = y_tr_seq.copy() # Les labels restent les mêmes pour les données augmentées
        X_train_final = np.vstack([X_tr_seq, X_aug_seq])
        y_train_final = np.concatenate([y_tr_seq, y_aug_seq])
    
    print(f"Prepro data shapes: X_train_final: {X_train_final.shape}, y_train_final: {y_train_final.shape}")
    if X_va_seq.shape[0] > 0 : print(f"X_va_seq: {X_va_seq.shape}, y_va_seq: {y_va_seq.shape}")
    else: print("X_va_seq is empty")
    if X_te_seq.shape[0] > 0 : print(f"X_te_seq: {X_te_seq.shape}, y_te_seq: {y_te_seq.shape}")
    else: print("X_te_seq is empty")

    # Retourner les données traitées, le scaler, et les informations sur les seuils utilisés pour ce fold
    return (X_train_final, y_train_final, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
            scaler, threshold_info_for_fold, tr_df, va_df, te_df)