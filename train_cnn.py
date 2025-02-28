"""
A script to train and test a 1D CNN ML model on capstone pulse time series data. Predicts whether or not a pulse is present in the signal.

To do:
- Model tuning
- Make sure data is labelled well
- Modify code to automatically detect num samples and sequence length
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
import random
from scipy.interpolate import interp1d

from load_data import build_dataset


NUM_FEATURES = 1
SEQUENCE_LENGTH = 240
NUM_SAMPLES = 701


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

    model = Sequential()

    # 1D convolutional layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(MaxPooling1D(pool_size=2))

    # additional convolutional and pooling layers can be added here
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # compile and fit the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=15, batch_size=32)

    return model


def augment_data(X, y):

    augmented_X = []
    augmented_y = []
    
    for series, label in zip(X, y):
        augmented_X.append(series)
        augmented_y.append(label)
        
        # randomly scale the data
        scale_factor = np.random.uniform(0.5, 2.0)
        augmented_X.append(series * scale_factor)
        augmented_y.append(label)

        # randomly stretch data along the x axis
        stretch_factor = np.random.uniform(1.0, 2.0)
        x_stretched = np.linspace(0, len(series), int(len(series) * stretch_factor))
        interpolator = interp1d(list(range(0, len(series))), series, kind='linear', fill_value="extrapolate")
        stretched_series = interpolator(x_stretched)
        offset = random.randint(0, len(stretched_series) - len(series)) # pick a random segment of the stretched section
        augmented_X.append(stretched_series[offset:offset+SEQUENCE_LENGTH])
        augmented_y.append(label)

    augmented_X = np.asarray(augmented_X)
    augmented_y = np.asarray(augmented_y)

    return augmented_X, augmented_y


if __name__ == "__main__":

    X, y = build_dataset()

    # check to make sure data formatting has not changed
    assert(NUM_SAMPLES == X.shape[0])
    assert(SEQUENCE_LENGTH == X.shape[1])

    X, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    X, y = augment_data(X_train, y_train)

    model = train_model(X_train, y_train)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n\nFinal test accuracy: {test_acc:.4f}\nFinal test loss: {test_loss:.4f}")

    y_pred = model.predict(X_test)

    while (1):
        rand_sample = random.randint(0, X_test.shape[0] - 1)
        plt.figure()
        plt.plot(X_test[rand_sample])
        plt.title(f"Prediction: {y_pred[rand_sample]}\nActual: {y_test[rand_sample]}\n{'Correct' if abs(y_pred[rand_sample] - y_test[rand_sample]) < 0.5 else "Incorrect"}")
        plt.show()
