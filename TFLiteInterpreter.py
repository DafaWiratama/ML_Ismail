import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    my_signature = interpreter.allocate_tensors()
    print(my_signature)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(np.random.random_sample(size=(1, 32, 7)), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print("UKNOWN", round(output_data[0][0] * 100), "%")
    print("SIT", round(output_data[0][1] * 100), "%")
    print("DOWN", round(output_data[0][2] * 100), "%")
    print("WALK", round(output_data[0][3] * 100), "%")
    print("RUN", round(output_data[0][4] * 100), "%")
