import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecay # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
import pickle

np.random.seed(42)
tf.random.set_seed(42)


def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_binary_lstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['AUC'])
    return model

def train(df, features, timesteps=30):
    X = df[features].values
    y = df["target"].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, timesteps)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_seq), y=y_seq)
    class_weights = dict(zip(np.unique(y_seq), class_weights))

    model = build_binary_lstm(input_shape=(timesteps, X_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    split = int(len(X_seq) * 0.8)
    model.fit(X_seq[:split], y_seq[:split], validation_data=(X_seq[split:], y_seq[split:]),
              epochs=100, batch_size=32, callbacks=[early_stop],
              class_weight=class_weights, verbose=0)

    return model, scaler
