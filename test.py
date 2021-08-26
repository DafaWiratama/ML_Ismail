import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, ConvLSTM2D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_window(dataframe, window_size=32, stride=1):
    data = np.array(dataframe)
    dataframe_size = len(dataframe)
    feature_size = dataframe.values.shape[1] - 1
    dataset_x = data[:, :-1]
    dataset_y = data[:, -1]

    windows_x = np.empty(shape=(int(dataframe_size / stride), window_size, feature_size))
    windows_y = np.empty(shape=(int(dataframe_size / stride), 1))

    for offset in range(0, dataframe_size - window_size, stride):
        indices = np.asarray([i for i in range(window_size)]) + offset
        windows_x[int(offset / stride)] = dataset_x[indices]
        windows_y[int(offset / stride)] = dataset_y[offset + window_size - 1]

    return windows_x, to_categorical(windows_y)


STATE = {
    "dws": 0,
    "jog": 1,
    "sit": 2,
    "std": 3,
    "ups": 4,
    "wlk": 5,
}


def dataset_motion_sense(type="Device Motion"):
    list = []
    for folder in os.listdir(f"dataset/Motion Sense/{type}"):
        state = STATE.get(folder[:3])
        for file in os.listdir(f"dataset/Motion Sense/{type}/{folder}"):
            df = pd.read_csv(f"dataset/Motion Sense/{type}/{folder}/{file}")
            df.pop(df.columns[0])
            df["state"] = state
            list.append(df)
    return pd.concat(list)


if __name__ == '__main__':
    dataframe = dataset_motion_sense("Accelerometer")

    print(np.concatenate((a.values, b.values), axis=-1).shape)
    print(a.head())
    print(b.head())
    print(dataframe.head())
    exit()
    x, y = create_window(dataframe, window_size=256, stride=32)

    n_features, n_outputs = x.shape[2], x.shape[1]
    n_steps, n_length = 4, 64
    x = x.reshape((x.shape[0], n_steps, 1, n_length, n_features))

    size = len(x)
    train_x = x[:round(size * .8)]
    train_y = y[:round(size * .8)]
    val_x = x[round(size * .7):round(size * .9)]
    val_y = y[round(size * .7):round(size * .9)]
    test_x = x[round(size * .9):]
    test_y = y[round(size * .9):]

    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(1, 3), input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=16, batch_size=1024)

    model.evaluate(test_x, test_y)
