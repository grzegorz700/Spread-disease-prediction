# Spread-disease-prediction
Predicition of spread disease (dengue fever) by a reccurent neural network.

Project consists off a few steps:
* analysis of the data
* preprocessing
* training the model
* prediction

Model based on a reccurent neural network (LSTM or GRU).

## Files structure:
* "Data properties and analysis.ipynb" - Shows analysis of the dataset/
* "Train neural network.ipynb" - Shows the schemat of preprocesing, training, and evaluating the model
* "preprocesing.py" - It contains classes used in preprocessing.
* "split_train_test.py" - It contains class for dividing the set to the train and dev sets.
* "utis.py" - It contains functions for ploting and playing sound after long processes.
* "data\\" - It should contain data from competition (https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). Data is not included due to the website rules. Data will be avaible after the agreement.
* "results\\" - It will be contain results after the training process.

## Data:
Data for the competition should be downloaded from the source website and put in the 'data\\' directory. Data is not included due to the website rules. Data will be avaible after the agreement.

## Actual best result based on code:
26.0673  mean absolute error

## Competition source:
https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

