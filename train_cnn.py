"""
A script to train and test a 1D CNN ML model on capstone pulse time series data. Predicts whether or not a pulse is present in a 4 second signal.

To do:
- Model tuning
    - sweep params:
        - # conv layers (1 - 7, 1 increments)
        - dense layer points (32 - 1024, power of 2 increments)
        - dropout rate (0.2 - 0.5, 0.05 increments)
        - l2 decay rate (0 - 0.1, 0.001 increments)
- Add cross validation

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
import logging
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import scipy.signal as signal
from numpy.polynomial.polynomial import Polynomial
from sklearn import preprocessing
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import random
from scipy.interpolate import interp1d
from datetime import datetime

from load_data import build_dataset

SPEED_MODE = True # train with a single epoch for debugging
LOGGING_DIR = 'logging'
MODELS_DIR = 'models'
SAVE_MODEL = False

def preprocess_data(X, y):
    """
    Preprocessing borrowed from our capstone projects preprocessing, reduces noise and normalizes.
    """

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

    model.add(Input((X_train.shape[1], 1)))

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
    history = model.fit(X_train, y_train, epochs=(1 if SPEED_MODE else 200), batch_size=32, validation_data=(X_val, y_val), callbacks=callback)
    logger.info(f"Model trained for {len(history.epoch)} epochs")

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
        augmented_X.append(stretched_series[offset:offset+X.shape[1]])
        augmented_y.append(label)

    augmented_X = np.asarray(augmented_X)
    augmented_y = np.asarray(augmented_y)

    return augmented_X, augmented_y


if __name__ == "__main__":

    log_filename = os.path.join(LOGGING_DIR, f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logger = logging.getLogger()

    logger.info(f"1D CNN for Pulse Detection")

    X_data, y_data = build_dataset()
    logger.info(f"Data loaded, {100 * np.count_nonzero(y_data == 1) / np.count_nonzero(y_data == 0):.2f}% of data has a pulse")

    X_data, y_data = preprocess_data(X_data, y_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3) # split test data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2) # split val data
    X_train, y_train = augment_data(X_train, y_train)
    logger.info(f"Data processed, divided, and augmented: train: {len(X_train)} | test: {len(X_test)} | val: {len(X_val)}")

    model = train_model(X_train, y_train, X_val, y_val)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    logger.info(f"Final test accuracy is {100*test_acc:.2f}% and final test loss is {test_loss:.4f}")

    if SAVE_MODEL:
        model_filename = os.path.join(MODELS_DIR, f'model_{int(100*test_acc)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.keras')
        model.save(model_filename)
        logging.info(f"Model saved as: {model_filename}")

    logger.info(f"Done!")
