
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from load_data import build_dataset
from train_cnn import augment_data, train_model, preprocess_data, ModelConfig

if __name__ == "__main__":

    X_data, y_data = build_dataset()
    X_data_proc, y_data_proc = preprocess_data(X_data, y_data)

    for i in range(len(X_data)):
        plt.figure(figsize=(12, 6))
        
        # Plot original sequence
        plt.subplot(1, 2, 1)
        plt.plot(X_data[i])
        plt.title(f"Original Sequence {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # Plot processed sequence
        plt.subplot(1, 2, 2)
        plt.plot(X_data_proc[i])
        plt.title(f"Processed Sequence {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # Show the plots
        plt.tight_layout()
        plt.show()

