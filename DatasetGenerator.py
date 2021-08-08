import numpy as np
import pandas as pd
from tqdm import tqdm


class DatasetGenerator:
    def __init__(self, filename, input_shape, label_dict):
        self._dataset = self._get_dataset(filename, label_dict)
        self._dataset = self._create_window(input_shape)
        self._split_dataset()

    def _get_dataset(self, filename, label_dict, label_name="state"):
        dataframe = pd.read_csv(filename)
        dataframe[label_name] = [label_dict[x] for x in dataframe.state]
        label = dataframe.pop(label_name)
        return dataframe.values, label.values

    def _create_window(self, input_shape):
        time_frame, input_size = input_shape
        x, y = self._dataset
        dataset_x = np.empty(shape=((len(x) - time_frame), time_frame, input_size))
        dataset_y = np.empty(shape=(len(x) - time_frame))

        for index in tqdm(range(len(dataset_x))):
            data = np.empty(shape=(time_frame, input_size))
            for t in range(time_frame):
                if t == time_frame:
                    data[t] = [*x[index + t], y[index + t]]
                else:
                    data[t] = [*x[index + t], 0]
            dataset_x[index] = data
            dataset_y[index] = y[index + time_frame]
        return dataset_x, dataset_y

    def _split_dataset(self, train=.7, val=.2, shuffle=True):
        x, y = self._dataset
        size = len(x)

        if shuffle:
            indices = np.arange(size)
            x, y = x[indices], y[indices]

        train_x = x[:int(size * train)]
        train_y = y[:int(size * train)]
        val_x = x[int(size * train):int(size * (train + val))]
        val_y = y[int(size * train):int(size * (train + val))]
        test_x = x[int(size * (train + val)):]
        test_y = y[int(size * (train + val)):]
        self.train = (train_x, train_y)
        self.val = (val_x, val_y)
        self.test = (test_x, test_y)
