import numpy as np
import pandas as pd
import numbers

import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


class PreprocessingDenga:
    def __init__(self, data_X, data_y, nan_fill_method='mean'):
        self.raw_data = data_X

        processing_data = self.raw_data.copy()
        processing_data.set_index('week_start_date', inplace=True)
        processing_data.drop('year', axis=1, inplace=True)          

        # Drop redundant columns
        processing_data.drop('reanalysis_sat_precip_amt_mm', axis=1, inplace=True)
        processing_data.drop('reanalysis_specific_humidity_g_per_kg', axis=1, inplace=True)

        # Fill nan values
        self.values_to_nan_fill = None
        if nan_fill_method == 'mean':
            self.values_to_nan_fill = processing_data.mean()
        if self.values_to_nan_fill is not None:
            processing_data.fillna(self.values_to_nan_fill, inplace=True)

        # Convert city name to numerical, and week_of_year to circle
        self.train_city_index = processing_data.keys().get_loc('city')
        values = processing_data.values
        self.encoder = LabelBinarizer()
        city = self.encoder.fit_transform(values[:, self.train_city_index])
        city_encoded = np.hstack((city, 1 - city))  # convert 1 column to 2 column each for one city

        self.train_week_index = processing_data.keys().get_loc('weekofyear')
        week_values = values[:, self.train_week_index]
        self.encoder_of_weeks = CircleTransform(week_values)
        week_sin, week_cos = self.encoder_of_weeks.transform(week_values)
        week_sin_cos = np.column_stack((week_sin, week_cos))

        values = np.delete(values, np.s_[self.train_city_index, self.train_week_index], axis=1)
        values = np.hstack((city_encoded, week_sin_cos, values))
        values = values.astype('float32')

        # Get final count of features
        self.features_count = values.shape[1]  # Warning: Shape may be dependent of transformations before, so check it.

        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(values)
        self.scaled = self.scaler.transform(values)  # TODO: Scaled to normal_version 

        # data_y preprocessing:
        self.data_y = data_y['total_cases']
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler.fit(self.data_y.values.reshape(-1, 1))

    def basic_X_preprocessing(self, given_data):
        """Prepare the same preprocessing process just like in init to new data."""
        processing_data = given_data.copy()
        processing_data.set_index('week_start_date', inplace=True)
        processing_data.drop('year', axis=1, inplace=True)
        processing_data.drop('reanalysis_sat_precip_amt_mm', axis=1, inplace=True)
        processing_data.drop('reanalysis_specific_humidity_g_per_kg', axis=1, inplace=True)
        if self.values_to_nan_fill is not None:
            processing_data.fillna(self.values_to_nan_fill, inplace=True)  

        # Transform of a city and a week attribute:
        city_index = processing_data.keys().get_loc('city')
        values = processing_data.values
        city = self.encoder.transform(values[:, city_index])
        city_encoded = np.hstack((city, 1 - city))

        week_index = processing_data.keys().get_loc('weekofyear')
        week_values = values[:, self.train_week_index]
        week_sin, week_cos = self.encoder_of_weeks.transform(week_values)
        week_sin_cos = np.column_stack((week_sin, week_cos))

        values = np.delete(values, np.s_[city_index, week_index], axis=1)
        values = np.hstack((city_encoded, week_sin_cos, values))

        scaled = self.scaler.transform(values)  
        return scaled

    def get_truncated_data_X(self, data_X_prepared, truncated_backprop_length):
        data_X_truncated = PreprocessingSeries.prepare_for_truncated_backpropagation(
            data_X_prepared, truncated_backprop_length - 1, 0)
        return data_X_truncated

    def get_split_data_X_by_city(self, data_X):
        index_of_first_occurrence_next_city = self.get_city_split_index(data_X, self.train_city_index)
        city1 = data_X[:index_of_first_occurrence_next_city, :]
        city2 = data_X[index_of_first_occurrence_next_city:, :]
        return city1, city2

    def get_split_and_truncated_data_y_by_city(self, data_y, truncated_backprop_length):
        index_of_first_occurrence_next_city = self.get_city_split_index(data_y)
        data_y = data_y['total_cases']
        y_city_1 = data_y[:index_of_first_occurrence_next_city]
        y_city_2 = data_y[index_of_first_occurrence_next_city:]

        # Delete rows with not enough history
        y_city_1 = y_city_1[truncated_backprop_length - 1:]
        y_city_2 = y_city_2[truncated_backprop_length - 1:]
        return y_city_1, y_city_2

    def get_normalized_y(self, given_data_y):
        data_y_norm = self.y_scaler.transform(given_data_y.reshape(-1, 1))
        data_y_norm_and_shape = data_y_norm.reshape(-1)
        return data_y_norm_and_shape

    def get_inverse_y(self, given_y):
        y_result_not_shaped = self.y_scaler.inverse_transform(given_y)
        y_result = y_result_not_shaped.astype(int).reshape(-1)
        return y_result

    # TODO: Refactor
    def get_city_split_index(self, data, index=float('nan')):
        if math.isnan(index) and not isinstance(data, pd.DataFrame):
            raise AttributeError("Wrong arguments to split_data")
        if math.isnan(index):
            index = data.keys().get_loc('city')
        if isinstance(data, pd.DataFrame):
            data = data.values
        values_column = data[:, index]

        if values_column.dtype not in [np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)]:
            values_column = self.encoder.transform(values_column)

        index = np.argmax(values_column == 0)
        return index


class PreprocessingSeries:
    @staticmethod
    def prepare_for_truncated_backpropagation(data, past_steps, future_steps=0):
        """ Prepare data to use it in the supervised time-series problem"""
        if not isinstance(past_steps, numbers.Number) and past_steps >= 0 and \
                not isinstance(future_steps, numbers.Number) and future_steps >= 0:
            raise ValueError("past_steps and future_steps should be a number greater than one")

        data = pd.DataFrame(data)
        # Stack past examples, and their column names
        columns_names = []
        shifted_dfs = []
        for i in range(past_steps, -(future_steps + 1), -1):
            shifted_dfs.append(data.shift(i))
            columns_names.extend([str(name) + "_t({})".format(-i) for name in data.columns.tolist()])
        result_df = pd.concat(shifted_dfs, axis=1)
        result_df.columns = columns_names

        # Remove examples with a incomplete history
        if future_steps == 0:
            result_df = result_df[past_steps:]
        else:
            result_df = result_df[past_steps:-future_steps]
        return result_df


class CircleTransform:
    """
    Transform values in range [min; max] to get version where a distance from 'min to max'
    is the same as from 'min +1 to min + 2'. It transforms one column values to 2 column
    values lay on circle with the middle in (0.5, 0.5), and the radius 0.5.
    """

    def __init__(self, values):
        self.min_val = np.min(values)
        self.max_val = np.max(values)+1

    def transform(self, values):
        values = values.astype('float32')
        values_scaled_0_2pi = (values-self.min_val)/(self.max_val-self.min_val) * 2 * np.pi
        return np.sin(values_scaled_0_2pi)*.5+.5, np.cos(values_scaled_0_2pi)*.5+.5
