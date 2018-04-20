import numpy as np
import pandas as pd


class TruncatedSplitTrainTest:
    def __init__(self, truncated_backprop_length, train_test_proportion=0.8):
        self.truncated_backprop_length = truncated_backprop_length
        self.train_test_proportion = train_test_proportion

    def get_X(self, data_X_city1, data_X_city2, print_stats=False):
        values_city1 = data_X_city1.values
        values_city2 = data_X_city2.values

        divide_in_city_1 = int(values_city1.shape[0] * self.train_test_proportion)
        divide_in_city_2 = int(values_city2.shape[0] * self.train_test_proportion)

        train_city1 = values_city1[:divide_in_city_1, :]
        test_city1  = values_city1[divide_in_city_1:, :]
        train_city2 = values_city2[:divide_in_city_2, :]
        test_city2  = values_city2[divide_in_city_2:, :]

        train_X = np.concatenate((train_city1, train_city2))
        test_X = np.concatenate((test_city1, test_city2))

        if print_stats:
            print('X values:')
            print('\tCity1:', train_city1.shape, test_city1.shape)
            print('\tCity2:', train_city2.shape, test_city2.shape)
            print('\tC1&C2:', train_X.shape, test_X.shape)

        return train_X, test_X

    def get_y(self, y_city1, y_city2, print_stats=False):
        divide_in_1city = int(y_city1.shape[0] * self.train_test_proportion)
        divide_in_2city = int(y_city2.shape[0] * self.train_test_proportion)

        train_y_city1 = y_city1[:divide_in_1city]
        test_y_city1 = y_city1[divide_in_1city:]
        train_y_city2 = y_city2[:divide_in_2city]
        test_y_city2 = y_city2[divide_in_2city:]

        train_y = np.concatenate((train_y_city1, train_y_city2))
        test_y = np.concatenate((test_y_city1, test_y_city2))

        if print_stats:
            print('Y values:')
            print('\tCity1:', train_y_city1.shape, test_y_city1.shape)
            print('\tCity2:', train_y_city2.shape, test_y_city2.shape)
            print('\tC1&C2:', train_y.shape, test_y.shape)

        return train_y, test_y
