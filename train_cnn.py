"""
A script to train and test a 1D CNN ML model on capstone pulse time series data. Predicts whether or not a pulse is present in the signal.

To do:
- Data augmentation
- Understand data preprocessing
- Model tuning
- Pulse detection (BPM)

Duncan Boyd
duncan@wapta.ca
Feb 24, 2025
"""

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
from numpy.polynomial.polynomial import Polynomial
from sklearn import preprocessing
from matplotlib import pyplot as plt

from load_data import build_dataset


NUM_FEATURES = 1
SEQUENCE_LENGTH = 240
NUM_SAMPLES = 385


def preprocess_data(X, y):

    for ii in range(X.shape[0]):
        # set the lowest value in the series to 0
        ppg_data = X[ii]
        min_val = min(ppg_data)
        ppg_data = [x - min_val for x in ppg_data]

        # apply a band pass filter to the data
        nyquist = 0.5 * 60
        low = 0.5 / nyquist # lowest heartrate of 0.5Hz/30BPM
        high = 5 / nyquist # highest heartrate of 5Hz/300BPM
        b, a = signal.butter(1, [low, high], btype='band')
        ppg_data = signal.filtfilt(b, a, ppg_data)

        # remove baseline drift with a polynomial fit
        p = Polynomial.fit(range(len(ppg_data)), ppg_data, 4)
        midline = p(range(len(ppg_data)))
        ppg_data = ppg_data - midline

        # normalize data
        X[ii] = preprocessing.normalize([ppg_data])

    return X, y


def train_model(X_train, y_train):

    # Build a 1D CNN model
    model = Sequential()

    # 1D Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(MaxPooling1D(pool_size=2))

    # Additional convolutional and pooling layers can be added here
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flattening the output from convolution layers before feeding to Dense layers
    model.add(Flatten())

    # Fully connected layers (Dense)
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model


if __name__ == "__main__":

    X, y = build_dataset()

    # check to make sure data formatting has not changed
    assert(NUM_SAMPLES == X.shape[0])
    assert(SEQUENCE_LENGTH == X.shape[1])

    X, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    model = train_model(X_train, y_train)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n\nFinal test accuracy: {test_acc:.4f}\nFinal test loss: {test_loss:.4f}")
