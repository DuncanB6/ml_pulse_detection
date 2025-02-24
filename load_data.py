"""
Loads pulse data from my capstone project and formats it for machine learning.

Duncan Boyd 
duncan@wapta.ca
Feb 24, 2025
"""

import os
import h5py
import numpy as np

DATA_FOLDER = 'data'
DATA_LENGTH = 400

def load_data_from_file(file_path, X, y):

    with h5py.File(file_path, 'r') as file:

        for series_name in list(file.keys()):
            ppg_data = file[f"{series_name}/ppg_data"][:]
            assert(len(ppg_data) == DATA_LENGTH)
            X = np.vstack([X, np.array([ppg_data])])

            pulse_label = file[f"{series_name}/pulse_label"][()].decode()
            y = np.append(y, 1 if pulse_label == 'T' else 0)

            #hr_label = file[f"{series_name}/hr_label"][()] # unused for now

    return X, y
    
def build_dataset():
    file_paths = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, file))]

    X = np.empty((0, DATA_LENGTH))
    y = np.array([])

    for file_path in file_paths:
        X, y = load_data_from_file(file_path, X, y)

    return X, y


if __name__ == "__main__":

    X, y = build_dataset()


    

