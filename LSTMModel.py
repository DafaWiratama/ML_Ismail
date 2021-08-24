import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input


class LSTMModel(tf.keras.Model):  # Definisi Model

    # Penentuan Layer
    def __init__(self, input_shape, output_shape):
        super(LSTMModel, self).__init__()
        self._layer_lstm_1 = Bidirectional(LSTM(16, return_sequences=False), input_shape=input_shape, name="input")
        self._layer_lstm_2 = Bidirectional(LSTM(8))
        self._layer_output = Dense(output_shape, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self._layer_lstm_1(inputs)
        # x = self._layer_lstm_2(x)
        return self._layer_output(x)
