import tensorflow as tf
from tensorflow.keras import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
import config 

def build_lstm_model(input_shape, n_classes=config.N_CLASSES):
    """Builds the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(),
        LSTM(32),
        BatchNormalization(),
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

    # 2. Chaîner les couches
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = LayerNormalization()(x)

    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x_norm_before_mha = LayerNormalization()(x) # Normaliser avant MHA

    # 3. MultiHeadAttention
    # Pour la self-attention, query, value, et key sont identiques.
    # Il faut les passer explicitement comme arguments nommés.
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(
        query=x_norm_before_mha, 
        value=x_norm_before_mha, 
        key=x_norm_before_mha  # Peut parfois être omis si value est utilisé pour key par défaut
    )
    # Note: MHA peut nécessiter une connexion résiduelle et une normalisation supplémentaire
    # x = x_norm_before_mha + attn_output # Connexion résiduelle (si les dimensions correspondent)
    # x = LayerNormalization()(x)       # Normalisation après addition
    # Pour cet exemple, nous allons simplement passer la sortie de MHA à la suite.
    x = BatchNormalization()(attn_output) # Ou appliquer BN/LN directement sur attn_output
    x = Dropout(0.4)(x)
    x = LayerNormalization()(x)

    x = LSTM(32)(x) # Cette LSTM prend maintenant la sortie du bloc MHA
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    # 4. Créer le modèle
    model = Model(inputs=inputs, outputs=outputs)

    # 5. Compiler le modèle (identique à avant)
    lr_schedule = CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=config.LR_DECAY_STEPS
    )
    loss_fn = SparseCategoricalCrossentropy()
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


AVAILABLE_MODELS = {
    "lstm": build_lstm_model,
    "lstm_v2": build_lstm_model_v2,
}

# Pour tester (exemple)
if __name__ == '__main__':
    dummy_input_shape = (10, 50) # Séquence de 10 pas, 50 features par pas
    model_v2 = build_lstm_model_v2(input_shape=dummy_input_shape, n_classes=3)
    model_v2.summary()
    
    # Vous pouvez aussi essayer de construire l'ancien modèle pour vérifier
    model_base = build_lstm_model(input_shape=dummy_input_shape, n_classes=3)
    model_base.summary()