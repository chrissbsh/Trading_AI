import config
from tensorflow.keras import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Conv1D, GRU, MaxPooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
import tensorflow as tf # type: ignore
import numpy as np
from tensorflow.keras.metrics import F1Score, AUC # type: ignore

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
    
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                  loss=loss_fn,
                  metrics=['accuracy', F1Score(), AUC()])
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
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', F1Score(), AUC()]
    )
    
    return model

def build_cnn_gru_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)

    # 1D Convolution pour capter les patterns locaux
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = LayerNormalization()(x)

    # GRU pour mémoire temporelle
    x = GRU(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(32)(x)
    x = Dropout(0.3)(x)

    # Dense
    x = Dense(32, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', F1Score(), AUC()])
    return model


def build_cnn_lstm_model(input_shape, n_classes=config.N_CLASSES):
    """
    Builds a CNN + LSTM model for time series classification.
    Args:
        input_shape: (timesteps, n_features)
        n_classes: number of classes to predict (e.g., 3 for up/neutral/down)
    Returns:
        Compiled Keras model.
    """
    model = Sequential([
        # --- CNN part ---
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LayerNormalization(),

        # --- LSTM part ---
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LayerNormalization(),
        LSTM(32),
        Dropout(0.3),
        LayerNormalization(),

        # --- Fully connected ---
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    # Loss and optimizer
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy', F1Score(), AUC()]
    )
    
    return model


AVAILABLE_MODELS = {
    "lstm": build_lstm_model,
    "lstm_v2": build_lstm_model_v2,
    "cnn_gru": build_cnn_gru_model,
    "cnn_lstm": build_cnn_lstm_model
}