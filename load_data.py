import os
import h5py
#import pandas as pd

DATA_FOLDER = 'data'

def load_data_from_file(file_path):

    with h5py.File(file_path, 'r') as file:
        print(list(file.keys()))

        for series_name in list(file.keys()):
            ppg_data = file[f"{series_name}/ppg_data"][:]
            pulse_label = file[f"{series_name}/pulse_label"][()].decode()
            hr_label = file[f"{series_name}/hr_label"][()]

    return
    
def create_dataset():
    file_paths = [os.path.join(DATA_FOLDER, file) for file in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, file))]
    print(file_paths)

    for file_path in file_paths:
        load_data_from_file(file_path)

    return


if __name__ == "__main__":

    file_paths = create_dataset()
    

