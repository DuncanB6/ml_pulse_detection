"""
Demo script for testing out the newest parameters and looking at some predictions. Trains a single model
using mostly functions from the other scripts. Plots random plots and their predictions.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from load_data import build_dataset
from train_cnn import augment_data, train_model, preprocess_data, ModelConfig


def model_eval(model_cfg, X_data, y_data):

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2
    )  # split val data

    X_train, y_train = augment_data(X_train, y_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2
    )  # split val data

    # add an axis for proper input shape
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    model, history = train_model(X_train, y_train, X_val, y_val, model_cfg)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy: {test_acc:.2f} | Loss: {test_loss:.2f}")

    while (1):
        random_index = np.random.randint(0, len(X_test))

        sample = X_test[random_index]
        true_label = y_test[random_index]

        y_pred = model.predict(np.expand_dims(sample, axis=0))

        pred_label = int(y_pred >= 0.5)

        # set background according to correctness
        if int(pred_label[0][0]) == int(true_label):
            background_color = 'lightgreen'
        else:
            background_color = 'lightcoral'

        plt.figure(facecolor=background_color)
        plt.plot(sample, color='k')
        plt.title(f'True: {int(true_label)}, Pred: {int(pred_label[0][0])}')
        plt.axis('off')
        plt.show()

    return

if __name__ == "__main__":

    X_data, y_data = build_dataset()
    X_data, y_data = preprocess_data(X_data, y_data)

    model_cfg = ModelConfig(conv_layers=4, dense_points=32, dropout_rate=0.44, l2_decay=0.045)

    model_eval(model_cfg, X_data, y_data)