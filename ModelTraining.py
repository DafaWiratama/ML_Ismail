import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from DatasetGenerator import DatasetGenerator
from LSTMModel import LSTMModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dict = {
    "UNKNOWN": 0,
    "SIT": 1,
    "STAND": 2,
    "WALK": 3,
    "RUN": 4,
}

METRICS = [
    'accuracy'
]


def show_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    input_shape = (32, 7)
    output_shape = 5

    dataset = DatasetGenerator("dataset.csv", input_shape, dict)

    model = LSTMModel(input_shape, output_shape)
    model.compile(optimizer=tf.optimizers.Adam(lr=1e-3),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=METRICS)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=32, restore_best_weights=True)

    history = model.fit(*dataset.train, validation_data=dataset.val, shuffle=True, epochs=8, batch_size=32, callbacks=[tensorboard, early_stop])

    input_data = np.asarray([[[-0.62249243, 2.2361844, 8.93546, 0.036193766, -0.026267206, 0.6966918, 3],
                              [0.11492168, 2.825158, 8.672098, 0.036193766, -0.026267206, 0.6966918, 3],
                              [0.11492168, 2.825158, 8.672098, 0.25060746, 0.08002335, 0.46822822, 3],
                              [0.51235914, 4.2544966, 8.806173, 0.25060746, 0.08002335, 0.46822822, 3],
                              [0.51235914, 4.2544966, 8.806173, 0.22006421, -0.07452556, 0.033292156, 3],
                              [-0.20829555, 3.6942532, 8.921095, 0.22006421, -0.07452556, 0.033292156, 3],
                              [-0.20829555, 3.6942532, 8.921095, -0.22403483, -0.23640485, -0.65270954, 3],
                              [-0.60094464, 1.5897499, 8.863634, -0.22403483, -0.23640485, -0.65270954, 3],
                              [-0.60094464, 1.5897499, 8.863634, -0.3022256, -0.65118235, -0.64843345, 3],
                              [-0.62967503, 1.970428, 9.648932, -0.3022256, -0.65118235, -0.64843345, 3],
                              [-0.62967503, 1.970428, 9.648932, 0.02031127, -0.105068825, -0.4981606, 3],
                              [-0.06943185, 2.3104045, 9.222764, 0.02031127, -0.105068825, -0.4981606, 3],
                              [-0.06943185, 2.3104045, 9.222764, -0.123242065, 0.1142318, 0.022296581, 3],
                              [0.40462008, 2.6791117, 9.328109, -0.123242065, 0.1142318, 0.022296581, 0],
                              [0.40462008, 2.6791117, 9.328109, 0.1339322, 0.30787608, 0.68936145, 0],
                              [0.5410896, 2.743755, 7.156568, 0.1339322, 0.30787608, 0.68936145, 2],
                              [0.5410896, 2.743755, 7.156568, 0.06612616, 0.049480084, 0.4260785, 2],
                              [0.19393034, 3.0262709, 9.627384, 0.06612616, 0.049480084, 0.4260785, 2],
                              [0.19393034, 3.0262709, 9.627384, -0.47448957, -0.099571034, 0.013744468, 2],
                              [0.5650316, 3.0789435, 9.481338, -0.47448957, -0.099571034, 0.013744468, 2],
                              [0.5650316, 3.0789435, 9.481338, -0.15317446, 0.040927973, 0.093767814, 2],
                              [0.2705448, 2.8083985, 9.648932, -0.15317446, 0.040927973, 0.093767814, 2],
                              [0.2705448, 2.8083985, 9.648932, 0.016035212, -0.05009095, 0.09254608, 2],
                              [0.46926352, 2.9712043, 9.081507, 0.016035212, -0.05009095, 0.09254608, 2],
                              [0.46926352, 2.9712043, 9.081507, -0.009010263, -0.08368854, -0.00580322, 2],
                              [0.260968, 2.9735985, 9.478944, -0.009010263, -0.08368854, -0.00580322, 2],
                              [0.260968, 2.9735985, 9.478944, -0.057268616, 0.026267206, -0.20372356, 2],
                              [0.059855044, 2.705448, 9.241918, -0.057268616, 0.026267206, -0.20372356, 2],
                              [0.059855044, 2.705448, 9.241918, -0.07192938, 0.040317107, -0.12064589, 2],
                              [0.110133275, 2.6767175, 9.466972, -0.07192938, 0.040317107, -0.12064589, 2],
                              [0.110133275, 2.6767175, 9.466972, -0.001069014, 0.06047566, -0.04917465, 2],
                              [0.06464344, 2.7581203, 9.466972, -0.001069014, 0.06047566, -0.04917465, 2],
                              ]], dtype=np.float32)

    print(model.predict(input_data))
    # show_graph(history)

    # model.save("model", overwrite=True)
