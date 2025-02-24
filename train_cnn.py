import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_data import build_dataset

def preprocess_data(X, y):

    num_samples = 1000
    sequence_length = 50
    num_features = 1

    # Generate random time series data
    X = np.random.randn(num_samples, sequence_length, num_features)
    y = np.random.randint(0, 2, size=(num_samples,))  # Binary classification labels

    # Data normalization (standardization)
    scaler = StandardScaler()
    X = X.reshape(-1, num_features)  # Reshaping for fitting scaler
    X = scaler.fit_transform(X)  # Standardize each feature
    X = X.reshape(num_samples, sequence_length, num_features)  # Reshaping back to (num_samples, sequence_length, num_features)

    return X, y

def train_model(X_train, y_train, X_val, y_val):

    # Build a 1D CNN model
    model = Sequential()

    # 1D Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(400, 1)))
    model.add(MaxPooling1D(pool_size=2))

    # Additional convolutional and pooling layers can be added here
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flattening the output from convolution layers before feeding to Dense layers
    model.add(Flatten())

    # Fully connected layers (Dense)
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    return model

def test_model(model):

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    return

if __name__ == "__main__":

    X, y = build_dataset()

    X, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, X_test, y_test)

    test_model(model, X_test, y_test)
