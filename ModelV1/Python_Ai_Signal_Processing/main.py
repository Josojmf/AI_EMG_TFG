import os
import pandas as pd
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

WINDOW_SIZE = 127
TARGET_NUM_CHANNELS = 9
MODEL_PATH = "./Model/ModelV1.h5"
HEALTHY_FOLDER = r"C:\INFORMATICA\TFG\ModelV1\Python_Ai_Signal_Processing\Datasets\Healthy-Dataset"
SPASTIC_FOLDER = r"C:\INFORMATICA\TFG\ModelV1\Python_Ai_Signal_Processing\Datasets\Spasticity-Dataset"

def load_data(folder_path):
    csv_files = glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
    list_df = []
    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        # Convert all columns except the last one to float, handling non-numeric issues
        df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
        list_df.append(df)
    return pd.concat(list_df) if list_df else pd.DataFrame()

def preprocess_data(data):
    # Assuming the last column is the label
    features = data.iloc[:, :-1]  # all columns except last as features
    labels = data.iloc[:, -1]  # last column as labels
    # Compute RMS or another statistic as a single 'Signal' value
    features['Signal'] = np.sqrt((features**2).mean(axis=1))
    features = features[['Signal']]  # Keep only the 'Signal' column
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

def build_model():
    model = Sequential([
        Dense(64, input_dim=1, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    healthy_data = load_data(HEALTHY_FOLDER)
    spastic_data = load_data(SPASTIC_FOLDER)
    if not healthy_data.empty and not spastic_data.empty:
        data = pd.concat([healthy_data, spastic_data])
        X_train, X_val, y_train, y_val = preprocess_data(data)
        model = build_model()
        model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
        model.save(MODEL_PATH)
        print("Model trained and saved at:", MODEL_PATH)
    else:
        print("Failed to load data. Please check your data files and directory paths.")

if __name__ == "__main__":
    main()
