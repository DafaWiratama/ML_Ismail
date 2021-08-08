import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input


class LSTMModel(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(LSTMModel, self).__init__()
        self._layer_lstm_1 = Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape)
        self._layer_lstm_2 = Bidirectional(LSTM(32))
        self._layer_output = Dense(output_shape, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self._layer_lstm_1(inputs)
        x = self._layer_lstm_2(x)
        return self._layer_output(x)

