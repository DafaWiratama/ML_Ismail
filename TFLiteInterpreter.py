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

    input_data = np.asarray([[[-1.1492168, 3.4332852, 10.757447, -0.24174993, -0.036651913, -0.3295618, 0],
                              [-1.9297266, 4.831499, 7.266701, -0.24174993, -0.036651913, -0.3295618, 0],
                              [-1.9297266, 4.831499, 7.266701, 0.54137933, 1.0396926, 0.15363261, 0],
                              [-3.1196449, 4.9012303, 7.252336, 0.54137933, 1.0396926, 0.15363261, 0],
                              [-3.1196449, 4.9012303, 7.252336, 0.3239113, 0.70371675, 0.10903945, 0],
                              [-4.1515455, 5.303456, 8.305785, 0.3239113, 0.70371675, 0.10903945, 1],
                              [-4.1515455, 5.303456, 8.305785, 0.3098614, 0.6957755, 0.035124753, 1],
                              [-3.5242648, 5.332186, 7.2714896, 0.3098614, 0.6957755, 0.035124753, 1],
                              [-3.5242648, 5.332186, 7.2714896, 0.016035212, -0.0421497, 0.07788532, 1],
                              [-3.667917, 5.2292356, 7.3720465, 0.016035212, -0.0421497, 0.07788532, 1],
                              [-3.667917, 5.2292356, 7.3720465, 0.001374447, -0.12095132, 0.049785517, 1],
                              [-3.4620156, 5.2364182, 7.0655885, 0.001374447, -0.12095132, 0.049785517, 1],
                              [-3.4620156, 5.2364182, 7.0655885, -0.030390546, 0.018325957, 0.005192355, 1],
                              [-3.7493198, 5.1406503, 7.352893, -0.030390546, 0.018325957, 0.005192355, 1],
                              [-3.7493198, 5.1406503, 7.352893, -0.01450805, 0.08246681, -0.08582657, 1],
                              [-3.8283286, 5.1286793, 7.522881, -0.01450805, 0.08246681, -0.08582657, 1],
                              [-3.8283286, 5.1286793, 7.522881, -0.05971208, 0.06291912, -0.07727446, 1],
                              [-3.5913026, 5.219659, 7.3121915, -0.05971208, 0.06291912, -0.07727446, 1],
                              [-3.5913026, 5.219659, 7.3121915, 0.03863723, -0.045814894, 0.070554934, 1],
                              [-3.7205894, 5.2172647, 7.307403, 0.03863723, -0.045814894, 0.070554934, 1],
                              [-3.7205894, 5.2172647, 7.307403, -0.002290745, -0.020158553, 0.010079277, 1],
                              [-3.7660792, 5.0831895, 7.3337393, -0.002290745, -0.020158553, 0.010079277, 1],
                              [-3.7660792, 5.0831895, 7.3337393, 0.013591751, -0.10567969, 0.008246681, 1],
                              [-3.359065, 5.3226094, 7.51091, 0.013591751, -0.10567969, 0.008246681, 1],
                              [3.359065, 5.3226094, 7.51091, 0.05085453, 0.006108652, 0.04795292, 1],
                              [3.53863, 5.1885343, 7.472603, 0.05085453, 0.006108652, 0.04795292, 1],
                              [-3.53863, 5.1885343, 7.472603, -0.017562376, -0.01282817, -0.03634648, 1],
                              [-3.5913026, 5.171775, 7.4821796, -0.017562376, -0.01282817, -0.03634648, 1],
                              [-3.5913026, 5.171775, 7.4821796, 0.016646078, -0.069027774, 0.020463986, 1],
                              [-3.5913026, 5.1526213, 7.307403, 0.016646078, -0.069027774, 0.020463986, 1],
                              [-3.5913026, 5.1526213, 7.307403, 0.01786781, -0.009162978, -0.003970624, 1],
                              [-3.6056678, 5.0951605, 7.3816233, 0.01786781, -0.009162978, -0.003970624, 1]]], dtype=np.float32)
    print(input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print("UKNOWN", round(output_data[0][0] * 100), "%")
    print("SIT", round(output_data[0][1] * 100), "%")
    print("DOWN", round(output_data[0][2] * 100), "%")
    print("WALK", round(output_data[0][3] * 100), "%")
    print("RUN", round(output_data[0][4] * 100), "%")
