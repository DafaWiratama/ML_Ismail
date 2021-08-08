import datetime
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.activations import softmax
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_model(input_shape, output_shape):
    input = Input(shape=input_shape)
    x = Bidirectional(LSTM(32, return_sequences=True))(input)
    x = Bidirectional(LSTM(16, return_sequences=False))(x)
    output = Dense(output_shape, activation=softmax)(x)
    model = Model(input, output)

    model.compile(optimizer=tf.optimizers.Adam(lr=0.001), loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


dict = {
    "UNKNOWN": 0,
    "SIT": 1,
    "STAND": 2,
    "WALK": 3,
    "RUN": 4,
}


def get_dataset():
    dataframe = pd.read_csv("dataset.csv")
    dataframe["state"] = [dict[x] for x in dataframe.state]
    label = dataframe.pop("state")
    return dataframe.values, label.values


def create_window(dataset, input_shape):
    time_frame, input_size = input_shape
    x, y = dataset
    dataset_x = np.empty(shape=((len(x) - time_frame), time_frame, input_size))
    dataset_y = np.empty(shape=(len(x) - time_frame))

    for index in tqdm(range(len(dataset_x))):
        data = np.empty(shape=(time_frame, input_size))
        for t in range(time_frame):
            data[t] = [*x[index + t], y[index + t]]
        dataset_x[index] = data
        dataset_y[index] = y[index + time_frame]

    return dataset_x, dataset_y


def split_dataset(dataset, train=.7, val=.2):
    x, y = dataset
    size = len(x)
    train_x = x[:int(size * train)]
    train_y = y[:int(size * train)]
    val_x = x[int(size * train):int(size * (train + val))]
    val_y = y[int(size * train):int(size * (train + val))]
    test_x = x[int(size * (train + val)):]
    test_y = y[int(size * (train + val)):]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


if __name__ == '__main__':
    input_shape = (32, 7)
    output_shape = 5

    dataset = get_dataset()
    dataset = create_window(dataset, input_shape=input_shape)
    dataset_train, dataset_val, test = split_dataset(dataset)

    model = get_model(input_shape=input_shape, output_shape=output_shape)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(*dataset_train, validation_data=dataset_val, shuffle=True, epochs=128, batch_size=128, callbacks=[callback])
