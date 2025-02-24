import os
import h5py
import pandas as pd

DATA_FOLDER = 'data'

def load_data_from_file(file_path):

    with h5py.File(file_path, 'r') as file:
        # Inspect the structure of the file if you want to explore the dataset
        print(list(file.keys()))
        
        # Assuming 'entries' is a group in the HDF5 file, and each entry contains data and two labels
        entries = file['entries']  # Replace 'entries' with the correct group name
        
        # List to store the data rows
        data_rows = []
        
        for entry in entries:
            # Assuming each entry contains a dataset of 'data' and labels 'label1' and 'label2'
            data = entries[entry]['data'][:]  # Replace 'data' with the actual dataset name
            label1 = entries[entry]['label1'][()]  # Replace 'label1' with the correct label name
            label2 = entries[entry]['label2'][()]  # Replace 'label2' with the correct label name
            
            # Append data row to the list
            data_rows.append([data, label1, label2])
        
    # Create a DataFrame from the collected rows
    df = pd.DataFrame(data_rows, columns=['Data', 'Label1', 'Label2'])

    # Display the DataFrame
    print(df)

    return
    
def create_dataset():
    file_paths = [file for file in os.listdir(DATA_FOLDER) if os.path.isfile(os.path.join(DATA_FOLDER, file))]

    return file_paths


if __name__ == "__main__":
    print('beginning...')
    file_paths = create_dataset()
    load_data_from_file(file_paths[0])
    print(file_paths)

