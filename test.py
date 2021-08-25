import os

import pandas as pd
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm
import tensorflow as tf

from WindowGenerator import WindowGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_dataset(dataset):
    print("Loading Dataset")
    frame = []
    for state in tqdm(os.listdir(f"dataset/{dataset}")):
        label = encode_state(state)
        for folder in os.listdir(f"dataset/{dataset}/{state}"):
            dataframe = pd.read_csv(f"dataset/{dataset}/{state}/{folder}")
            dataframe = dataframe.drop(dataframe.columns[0], axis=1)
            dataframe["state"] = label
            frame.append(dataframe)
    return pd.concat(frame)


def encode_state(state):
    if "dws" in state:
        return 0
    elif "jog" in state:
        return 1
    elif "sit" in state:
        return 2
    elif "std" in state:
        return 3
    elif "ups" in state:
        return 4
    elif "wlk" in state:
        return 5


if __name__ == '__main__':
    dataframe = get_dataset("Device Motion")
    n = len(dataframe)
    train_df = dataframe[0:int(n * 0.7)]
    val_df = dataframe[int(n * 0.7):int(n * 0.9)]
    test_df = dataframe[int(n * 0.9):]
    window = WindowGenerator(input_width=32, label_width=1, shift=1, label_columns=['state'], train_df=train_df, val_df=val_df, test_df=test_df)

    model = Sequential()
    model.add(LSTM(128, input_shape=(32, 13)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    model.fit(window.train, validation_data=window.val, epochs=8, batch_size=1024)
