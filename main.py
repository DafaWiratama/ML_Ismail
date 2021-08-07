import datetime
import numpy as  np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.layers import Dense, LSTM, Flatten
from tensorflow.python.keras.models import Sequential


def createModel():
    model = Sequential()
    model.add(LSTM(8, input_dim=(6, 60)))
    model.add(LSTM(8, input_dim=(6, 60)))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))
    return model


if __name__ == '__main__':
    dataframe = pd.read_csv("sensor.csv")
    dataframe.pop('_key')

    y = pd.get_dummies(dataframe.pop('label'))
    x = dataframe

    train_x = np.asarray(x)[0:64]
    train_y = np.asarray(y)[0:64]

    val_x = np.asarray(x)[64:]
    val_y = np.asarray(y)[64:]

    print(train_y)

    model = createModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=.00001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', tf.keras.metrics.Precision()]
                  )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=64, batch_size=8, callbacks=[callback])
