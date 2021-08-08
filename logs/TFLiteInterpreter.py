import tensorflow as tf

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    my_signature = interpreter.get_signature_runner()
