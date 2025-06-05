from tensorflow.keras.models import Sequential # type : ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input 

def get_model(input_shape, n_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model
