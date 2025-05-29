import os
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

time_steps = 100


def clean_decimal(value):
    """Ensures a valid decimal number by keeping only the first dot."""
    if isinstance(value, str):
        value = re.sub(r'(?<=\d)\.(?=.*\.)', '', value)  
    return value


def load_data(folder):
    data = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, dtype=str) 
            
            df["Signal"] = df["Signal"].apply(clean_decimal)

            df["Signal"] = pd.to_numeric(df["Signal"], errors='coerce')

            if "Label" not in df.columns:
                print(f"⚠ Warning: No 'Label' column found in {file}")
                continue

            df["Label"] = pd.to_numeric(df["Label"], errors='coerce')

            df = df.dropna()

            df["Signal"] = MinMaxScaler().fit_transform(df[["Signal"]])

            for i in range(len(df) - time_steps):
                data.append(df["Signal"].values[i:i+time_steps])
                labels.append(df["Label"].values[i+time_steps])

    return np.array(data), np.array(labels)


filtered_data, filtered_labels = load_data(
    "C:/INFORMATICA/TFG/ModelV2CNN/Datasets/Train_Data/filtered")
original_data, original_labels = load_data(
    "C:/INFORMATICA/TFG/ModelV2CNN/Datasets/Train_Data/original")


X = np.concatenate((filtered_data, original_data), axis=0)
y = np.concatenate((filtered_labels, original_labels), axis=0)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train = X_train.reshape(-1, time_steps, 1)
X_test = X_test.reshape(-1, time_steps, 1)


model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu',
           input_shape=(time_steps, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))


model.save("spasticity_cnn_model.h5")


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.2f}")
