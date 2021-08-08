import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_saved_model("model")
    model = converter.convert()
    with open('model.tflite', 'wb') as file:
        file.write(model)
