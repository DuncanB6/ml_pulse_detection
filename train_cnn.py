"""
A script to train and test a 1D CNN ML model on capstone pulse time series data. Predicts whether or not a pulse is present in a 4 second signal.

Duncan Boyd
duncan@wapta.ca
Feb 24, 2025
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress a TF warning about CPU use

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
import statistics as stats
import csv

from load_data import build_dataset

SPEED_MODE = False  # train with a single epoch for debugging
LOGGING_DIR = "logging"
MODELS_DIR = "models"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
SAVE_MODEL = False


class ModelConfig:
    def __init__(
        self, conv_layers=3, dense_points=128, dropout_rate=0.3, l2_decay=0.01
    ):
        self.conv_layers = conv_layers
        self.dense_points = dense_points
        self.dropout_rate = dropout_rate
        self.l2_decay = l2_decay


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
        low = 0.5 / nyquist  # lowest heartrate of 0.5Hz/30BPM
        high = 5 / nyquist  # highest heartrate of 5Hz/300BPM
        b, a = signal.butter(1, [low, high], btype="band")
        ppg_data = signal.filtfilt(b, a, ppg_data)

        # remove baseline drift with a polynomial fit
        p = Polynomial.fit(range(len(ppg_data)), ppg_data, 4)
        midline = p(range(len(ppg_data)))
        ppg_data = ppg_data - midline

        # normalize data
        X[ii] = preprocessing.normalize([ppg_data])

    return X, y


def train_model(X_train, y_train, X_val, y_val, model_cfg: ModelConfig):

    model = Sequential()

    filters = 64
    # Define the first Conv1D layer with input shape
    model.add(
        Conv1D(
            filters=filters,
            kernel_size=3,
            activation="relu",
            input_shape=(X_train.shape[1], 1),
        )
    )
    model.add(MaxPooling1D(pool_size=2))

    # Add additional Conv1D and MaxPooling1D layers as defined in model_cfg
    for ii in range(1, model_cfg.conv_layers):
        filters *= 2
        model.add(Conv1D(filters=filters, kernel_size=3, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(
        Dense(
            2 * model_cfg.dense_points,
            activation="relu",
            kernel_regularizer=l2(model_cfg.l2_decay),
        )
    )
    model.add(Dropout(model_cfg.dropout_rate))
    model.add(
        Dense(
            model_cfg.dense_points,
            activation="relu",
            kernel_regularizer=l2(model_cfg.l2_decay),
        )
    )

    # Output layer for binary classification
    model.add(Dense(1, activation="sigmoid"))

    # Compile and fit the model
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    # Early stopping callback
    callback = EarlyStopping(patience=5, restore_best_weights=True)

    # Fit the model with training and validation data
    history = model.fit(
        X_train,
        y_train,
        epochs=(1 if SPEED_MODE else 200),
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[callback],
        verbose=0,
    )

    return model, history


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
        interpolator = interp1d(
            list(range(0, len(series))), series, kind="linear", fill_value="extrapolate"
        )
        stretched_series = interpolator(x_stretched)
        offset = random.randint(
            0, len(stretched_series) - len(series)
        )  # pick a random segment of the stretched section
        augmented_X.append(stretched_series[offset : offset + X.shape[1]])
        augmented_y.append(label)

    augmented_X = np.asarray(augmented_X)
    augmented_y = np.asarray(augmented_y)

    return augmented_X, augmented_y


def model_trial(model_cfg, logger):

    logger.info(f"1D CNN for Pulse Detection")

    X_data, y_data = build_dataset()
    logger.info(
        f"Data loaded, {100 * np.count_nonzero(y_data == 1) / np.count_nonzero(y_data == 0):.2f}% of data has a pulse"
    )

    X_data, y_data = preprocess_data(X_data, y_data)

    model_accuracies = []
    kf = KFold(n_splits=2 if SPEED_MODE else 10)
    for i, (train_index, test_index) in enumerate(kf.split(X_data)):
        logging.info(f"Fold {i}")
        X_train = X_data[train_index]
        y_train = y_data[train_index]
        X_test = X_data[test_index]
        y_test = y_data[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2
        )  # split val data
        X_train, y_train = augment_data(X_train, y_train)
        logger.info(
            f"Data processed, divided, and augmented: train: {len(X_train)} | test: {len(X_test)} | val: {len(X_val)}"
        )

        # add an axis for proper input shape
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        model, history = train_model(X_train, y_train, X_val, y_val, model_cfg)
        logger.info(f"Model trained for {len(history.epoch)} epochs")

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(
            f"Test accuracy is {100*test_acc:.2f}% and final test loss is {test_loss:.4f}"
        )

        model_accuracies.append(test_acc)

    mean_acc = stats.mean(model_accuracies)
    logger.info(f"Mean accuracy is {100*mean_acc:.2f}%")

    if SAVE_MODEL:
        model_filename = os.path.join(
            MODELS_DIR,
            f'model_{int(100*test_acc)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.keras',
        )
        model.save(model_filename)
        logging.info(f"Model saved as: {model_filename}")

    logger.info(f"Done!\n\n")

    return mean_acc


def sweep_param(param, sweep, logger, csv_path):

    model_cfg = ModelConfig()

    logging.info(f"Sweeping {param}")
    sweep_accs = {}
    for val in sweep:

        if param == "conv_layers":
            model_cfg.conv_layers = val
        elif param == "dense_points":
            model_cfg.dense_points = val
        elif param == "dropout_rate":
            model_cfg.dropout_rate = val
        elif param == "l2_rate":
            model_cfg.l2_decay = val

        logging.info(f"Trying {val}")

        try:
            mean_acc = model_trial(model_cfg, logger)
        except KeyboardInterrupt as e:
            logging.exception("Interrupted by user!")
            exit()
        except Exception as e:
            logging.exception("Exception occured!")
            break

        sweep_accs[val] = mean_acc

        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    datetime.now(),
                    model_cfg.conv_layers,
                    model_cfg.dense_points,
                    model_cfg.dropout_rate,
                    model_cfg.l2_decay,
                    mean_acc,
                ]
            )

    logging.info(f"{param} Sweep Results:\n{sweep_accs}\n\n")

    plt.clf()
    plt.plot(list(sweep_accs.keys()), list(sweep_accs.values()))
    plt.title(f"Accuracy vs. # {param}")
    fig_path = os.path.join(FIGURES_DIR, f"{param}_sweep.png")
    plt.savefig(fig_path)

    return


if __name__ == "__main__":

    log_filename = os.path.join(
        LOGGING_DIR, f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    model_cfg = ModelConfig()

    # configure sweeps
    conv_sweep = list(range(1, 6 + 1))  # 1 - 6, 1 increments
    dense_sweep = [32 * (2**i) for i in range(6)]  # 32 - 1024, power of 2 increments
    dropout_sweep = list(np.arange(0.2, 0.500001, 0.05))
    dropout_sweep = np.round(dropout_sweep, 2)  # 0.2 - 0.5, 0.05 increments
    l2_sweep = list(np.arange(0, 0.100001, 0.01))
    l2_sweep = np.round(l2_sweep, 2)  # 0 - 0.1, 0.01 increments

    headers = [
        "Time",
        "Conv Layers",
        "Dense_layers",
        "Dropout Rate",
        "L2 Rate",
        "Accuracy",
    ]

    csv_path = os.path.join(
        RESULTS_DIR, f'results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    )
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    sweep_param("conv_layers", conv_sweep, logger, csv_path)
    sweep_param("dense_points", dense_sweep, logger, csv_path)
    sweep_param("dropout_rate", dropout_sweep, logger, csv_path)
    sweep_param("l2_rate", l2_sweep, logger, csv_path)

    logger.info(f"Extra Done!\n\n")
