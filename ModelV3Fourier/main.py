import numpy as np
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from CNNModel import CNNModel
from FFTTransformer import FFTTransformer
import logging
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base dataset directories
FILTERED_FOLDER = r"C:\INFORMATICA\TFG\ModelV3Fourier\Datasets\Train_Data\filtered"
ORIGINAL_FOLDER = r"C:\INFORMATICA\TFG\ModelV3Fourier\Datasets\Train_Data\original"

# Function to load all CSV files from a directory
def load_csv_from_folder(folder):
    dataframes = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            logging.info(f"Loading dataset: {file_path}")
            df = pd.read_csv(file_path, dtype={"Signal": str, "Label": str})  # Ensure strings to handle issues
            df["Signal"] = pd.to_numeric(df["Signal"], errors='coerce')  # Convert to numeric
            df["Label"] = pd.to_numeric(df["Label"], errors='coerce')
            df.dropna(subset=["Signal", "Label"], inplace=True)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else None

# Load and preprocess dataset
def load_and_preprocess_data():
    logging.info("Starting dataset loading and preprocessing")
    start_time = time.time()
    
    df_filtered = load_csv_from_folder(FILTERED_FOLDER)
    df_original = load_csv_from_folder(ORIGINAL_FOLDER)
    
    if df_filtered is None or df_original is None:
        raise ValueError("Error: One or both dataset folders are empty or missing valid CSV files.")
    
    df = pd.concat([df_filtered, df_original]).sample(frac=1).reset_index(drop=True)
    logging.info(f"Total dataset size before FFT: {df.shape}")
    
    X = df['Signal'].values
    y = df['Label'].values
    
    fft_transformer = FFTTransformer()
    X_transformed = fft_transformer.apply_fft(X)
    logging.info(f"Transformed dataset shape: {X_transformed.shape}")
    
    y = y[:len(X_transformed)]  # Ensure consistency
    logging.info(f"Label distribution: {np.bincount(y.astype(int))}")
    
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_transformed)
    
    logging.info(f"Dataset preprocessing completed in {time.time() - start_time:.2f} seconds")
    return train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train and evaluate the CNN model
logging.info("Starting model training")
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Compute class weights to handle imbalanced labels
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Check dataset sizes
logging.info(f"Training set size: {X_train.shape}, {y_train.shape}")
logging.info(f"Test set size: {X_test.shape}, {y_test.shape}")

cnn_model = CNNModel((X_train.shape[1], 1))
history = cnn_model.train(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, class_weight=class_weight_dict)
logging.info("Model training complete")

loss, accuracy = cnn_model.evaluate(X_test, y_test)
logging.info(f"Test Accuracy: {accuracy:.2f}")
cnn_model.save_model("cnn_fft_model.keras")  # Save in recommended Keras format
logging.info("Model saved successfully")
