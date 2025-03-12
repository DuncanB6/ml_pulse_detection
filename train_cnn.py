"""
A script to train and test a 1D CNN ML model on capstone pulse time series data. Predicts whether or not a pulse is present in the signal.

To do:
- Model tuning
    - sweep params:
        - # conv layers (1 - 7, 1 increments)
        - dense layer points (32 - 1024, power of 2 increments)
        - dropout rate (0.2 - 0.5, 0.05 increments)
        - l2 decay rate (0 - 0.1, 0.001 increments)
- Add cross validation
- Make sure data is labelled well
- Modify code to automatically detect num samples and sequence length

To explore:
- skip connections
- BPM prediction
- https://www.hackster.io/news/easy-tinyml-on-esp32-and-arduino-a9dbc509f26c

Duncan Boyd
duncan@wapta.ca
Feb 24, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress a TF warning about CPU use

import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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


def train_model(X_train, y_train, X_val, y_val):

    model = Sequential()

    model.add(Input((SEQUENCE_LENGTH, 1)))

    # 1D convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=1024, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # flatten and dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))  # output layer for binary classification

    # compile and fit the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    callback = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=callback)

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

    print(f"\n{25*'-'} 1D CNN for Pulse Detection {25*'-'}\n")

    X_data, y_data = build_dataset()

    print(f"{100 * np.count_nonzero(y_data == 1) / np.count_nonzero(y_data == 0):.2f}% of data has a pulse")

    # check to make sure data formatting has not changed
    #assert(NUM_SAMPLES == X.shape[0])
    #assert(SEQUENCE_LENGTH == X.shape[1])

    X_data, y_data = preprocess_data(X_data, y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    X_train, y_train = augment_data(X_train, y_train)

    model = train_model(X_train, y_train, X_val, y_val)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n\nFinal test accuracy: {test_acc:.4f}\nFinal test loss: {test_loss:.4f}")

    while (1):
        rand_sample = random.randint(0, X_test.shape[0] - 1)
        X_test_sample = np.asarray([X_test[rand_sample]])
        y_test_sample = y_test[rand_sample]

        y_pred = model.predict(X_test_sample)[0]

        plt.figure()
        plt.plot(X_test_sample[0])
        plt.title(f"Prediction: {y_pred[0]:.2f}\nActual: {y_test_sample}\n{'Correct' if abs(y_pred - y_test_sample) < 0.5 else "Incorrect"}")
        plt.show()
