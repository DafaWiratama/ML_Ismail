import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

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
                  loss=tf.losses.SparseCategoricalCrossentropy(),
                  metrics=METRICS)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=32, restore_best_weights=True)

    history = model.fit(*dataset.train, validation_data=dataset.val, shuffle=True, epochs=4, batch_size=128, callbacks=[tensorboard, early_stop])

    show_graph(history)

    model.save("model", overwrite=True)

