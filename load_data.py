"""
Loads pulse data from my capstone project and formats it for machine learning.

Duncan Boyd
duncan@wapta.ca
Feb 24, 2025
"""

import os
import h5py
import numpy as np
import scipy.signal as signal
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from sklearn import preprocessing

DATA_FOLDER = "data"


def load_data_from_file(file_path, X_data: list, y_data: list):

    with h5py.File(file_path, "r") as file:

        for series_name in list(file.keys()):
            ppg_data = file[f"{series_name}/ppg_data"][:]
            X_data.append(ppg_data)

            pulse_label = file[f"{series_name}/pulse_label"][()]
            y_data.append(1 if pulse_label == "T" else 0)

            # hr_label = file[f"{series_name}/hr_label"][()] # unused for now

    return X_data, y_data


def build_dataset():
    file_paths = [
        os.path.join(DATA_FOLDER, file)
        for file in os.listdir(DATA_FOLDER)
        if os.path.isfile(os.path.join(DATA_FOLDER, file))
    ]

    X_data = []
    y_data = []

    for file_path in file_paths:
        X_data, y_data = load_data_from_file(file_path, X_data, y_data)

    X_data = np.asarray(X_data).astype(np.float64)
    y_data = np.asarray(y_data).astype(np.float64)

    # randomly shuffle data
    np.random.seed(33)
    indices = np.random.permutation(len(X_data))
    X_data = X_data[indices]
    y_data = y_data[indices]

    return X_data, y_data


if __name__ == "__main__":

    X, y = build_dataset()

    ii = 0
    while(1):
        raw_ppg = X[ii]

        # Step 1: Remove min value (baseline shift)
        min_val = min(raw_ppg)
        ppg_data = [x - min_val for x in raw_ppg]

        # Step 2: Apply bandpass filter
        nyquist = 0.5 * 60
        low = 0.5 / nyquist  # 0.5 Hz
        high = 5 / nyquist   # 5 Hz
        b, a = signal.butter(1, [low, high], btype="band")
        ppg_data = signal.filtfilt(b, a, ppg_data)

        # Step 3: Remove baseline drift using polynomial fit
        x_vals = range(len(ppg_data))
        p = Polynomial.fit(x_vals, ppg_data, 4)
        midline = p(x_vals)
        ppg_data = ppg_data - midline

        # Step 4: Normalize
        ppg_data = preprocessing.normalize([ppg_data])[0]  # flatten from 2D to 1D

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        # Before processing
        axes[0].plot(raw_ppg, color='skyblue')
        axes[0].set_title("Before Processing")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)
        axes[0].set_ylim([min(raw_ppg), max(raw_ppg)])  # Adjust y-axis based on original data range

        # After processing
        axes[1].plot(ppg_data, color='seagreen')
        axes[1].set_title("After Processing")
        axes[1].set_xlabel("Sample Index")
        axes[1].grid(True)
        axes[1].set_ylim([min(ppg_data), max(ppg_data)])  # Adjust y-axis based on processed data range

        plt.suptitle("PPG Signal Comparison")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        ii += 1


        