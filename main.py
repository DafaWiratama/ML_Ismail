import datetime
import os

from tensorflow.keras.layers import Bidirectional, LSTM
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    dataframe = pd.read_csv("dataset.csv")

    input_features = 6
    input_len = 32
    label_size = 5

    y = np.asarray(tf.one_hot(dataframe.pop("state").astype("category").cat.codes, label_size))
    x = np.asarray(dataframe.values.tolist())

    dataset_x = np.empty(shape=((len(x) - input_len), input_len, input_features + label_size))
    dataset_y = np.empty(shape=((len(x) - input_len), label_size))

    model = tf.keras.models.Sequential([
        Bidirectional(LSTM(32, return_sequences=True), input_shape=(input_len, input_features + label_size)),
        Bidirectional(LSTM(8, return_sequences=False)),
        tf.keras.layers.Dense(units=5, activation='softmax')
    ])
    model.compile(optimizer=tf.optimizers.Adam(lr=0.001), loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()

    for index in tqdm(range(len(dataset_x))):
        data = np.empty(shape=(input_len, input_features + label_size))
        for t in range(input_len):
            data[t] = [*x[index + t], *y[index + t]]
        dataset_x[index] = data
        dataset_y[index] = y[index + input_len]

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(dataset_x[:1000], dataset_y[:1000], validation_data=(dataset_x[1000:], dataset_y[1000:]), shuffle=True, epochs=32, callbacks=[callback])
