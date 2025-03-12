"""
Loads pulse data from my capstone project and formats it for machine learning.

Duncan Boyd 
duncan@wapta.ca
Feb 24, 2025
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

DATA_FOLDER = 'data'

def load_data_from_file(file_path, X_data:list, y_data:list):

    with h5py.File(file_path, 'r') as file:

        for series_name in list(file.keys()):
            ppg_data = file[f"{series_name}/ppg_data"][:]
            X_data.append(ppg_data)

            pulse_label = file[f"{series_name}/pulse_label"][()].decode()
            y_data.append(1 if pulse_label == 'T' else 0)

            #hr_label = file[f"{series_name}/hr_label"][()] # unused for now

    return X_data, y_data
    
def build_dataset():
    file_paths = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, file))]

    X_data = []
    y_data = []

    for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
        X_data, y_data = load_data_from_file(file_path, X_data, y_data)

    X_data = np.asarray(X_data).astype(np.float64)
    y_data = np.asarray(y_data).astype(np.float64)

    return X_data, y_data


if __name__ == "__main__":

    X, y = build_dataset()


    

