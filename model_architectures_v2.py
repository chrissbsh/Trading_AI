import config
from tensorflow.keras import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
import tensorflow as tf # type: ignore
import numpy as np

def build_lstm_model(input_shape, n_classes=config.N_CLASSES):
    """Builds the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LayerNormalization(),
        LSTM(32),
        Dropout(0.3),
        LayerNormalization(),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    lr_schedule = CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS
    )
    loss_fn = SparseCategoricalCrossentropy()
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


def build_lstm_model_v2(input_shape, n_classes=config.N_CLASSES):
    """Builds a more complex unidirectional LSTM model using Functional API."""
    
    # 1. Définir la couche d'entrée
    inputs = Input(shape=input_shape)

    # 2. Première couche LSTM avec normalisation simple
    x = LSTM(128, return_sequences=True)(inputs)
    x = LayerNormalization()(x)  # Une seule normalisation
    x = Dropout(0.2)(x)  # Dropout réduit

    # 3. Deuxième couche LSTM
    x = LSTM(64, return_sequences=True)(x)
    x_residual = LayerNormalization()(x)  # Garder pour connexion résiduelle
    x = Dropout(0.2)(x_residual)

    # 4. MultiHeadAttention avec connexion résiduelle
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(
        query=x, 
        value=x, 
        key=x
    )
    
    # IMPORTANT: Connexion résiduelle restaurée
    x = x_residual + attn_output  # Connexion résiduelle
    x = LayerNormalization()(x)   # Normalisation après addition
    x = Dropout(0.2)(x)

    # 5. Dernière LSTM (sans return_sequences)
    x = LSTM(32)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # 6. Couches de classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    # 7. Créer le modèle
    model = Model(inputs=inputs, outputs=outputs)

    # 8. Compilation avec learning rate plus conservateur
    # Option 1: Learning rate fixe plus bas
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # LR plus bas
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

# Fonction utilitaire pour diagnostiquer les problèmes d'entraînement
def diagnose_training_issues(model, X_train, y_train, X_val, y_val):
    """
    Fonction pour diagnostiquer les problèmes d'entraînement
    """
    print("=== DIAGNOSTIC D'ENTRAÎNEMENT ===")
    
    # 1. Vérifier la distribution des classes
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Distribution des classes d'entraînement: {dict(zip(unique, counts))}")
    
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"Distribution des classes de validation: {dict(zip(unique_val, counts_val))}")
    
    # 2. Vérifier la plage des données d'entrée
    print(f"Plage des données X_train: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Moyenne X_train: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    
    # 3. Prédictions initiales (avant entraînement)
    initial_pred = model.predict(X_val[:100])
    print(f"Prédictions initiales - Min: {initial_pred.min():.4f}, Max: {initial_pred.max():.4f}")
    print(f"Entropie moyenne des prédictions initiales: {-np.mean(np.sum(initial_pred * np.log(initial_pred + 1e-8), axis=1)):.4f}")
    
    # 4. Test avec un batch pour vérifier les gradients
    with tf.GradientTape() as tape:
        pred = model(X_train[:32])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train[:32], pred)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in gradients]
    print(f"Normes des gradients: Min={min(grad_norms):.6f}, Max={max(grad_norms):.6f}, Moyenne={np.mean(grad_norms):.6f}")
    
    return True

# Configuration d'entraînement améliorée
def get_improved_training_config():
    """
    Retourne une configuration d'entraînement plus stable
    """
    return {
        'batch_size': 32,  # Plus petit pour plus de stabilité
        'epochs': 100,
        'patience': 15,  # Plus de patience pour l'early stopping
        'min_delta': 0.001,  # Seuil plus petit pour l'amélioration
        'restore_best_weights': True,
        'monitor': 'val_loss',
        'mode': 'min',
        'verbose': 1
    }

AVAILABLE_MODELS = {
    "lstm": build_lstm_model,
    "lstm_v2": build_lstm_model_v2,
}