import numpy as np
import os
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from UltimateCNNModelGPU import UltimateCNNModelGPU
from FFTTransformerGPU import FFTTransformerGPU
import logging
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# 🛠 CONFIGURE LOGGING
# ===============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===============================
# 📊 DATASET PATHS (ADAPTS FOR DOCKER)
# ===============================
if os.path.exists("/workspace"):  # Running inside Docker
    BASE_PATH = "/workspace/Datasets/Train_Data"
else:  # Running on Windows
    BASE_PATH = r"C:\\INFORMATICA\\TFG\\ModelV3.5Fourier\\Datasets\\Train_Data"

FILTERED_FOLDER = os.path.join(BASE_PATH, "filtered")
ORIGINAL_FOLDER = os.path.join(BASE_PATH, "original")

# ===============================
# 👍 FUNCTION: LOAD CSV FILES FAST
# ===============================
def load_csv_from_folder(folder, chunk_size=50000):
    """Loads and concatenates CSV files from a given folder."""
    dataframes = []
    if not os.path.exists(folder):
        logging.error(f"🚨 Dataset folder not found: {folder}")
        return None
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            try:
                df = pd.read_csv(file_path, dtype=str, chunksize=chunk_size)  # Load as string first
                for chunk in df:
                    chunk = chunk.apply(pd.to_numeric, errors='coerce')  # Convert, forcing NaNs on errors
                    chunk.dropna(inplace=True)  # Remove corrupted rows
                    dataframes.append(chunk)
            except Exception as e:
                logging.warning(f"⚠️ Error reading {file_path}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else None

# ===============================
# 📜 PARALLEL FFT TRANSFORMATION
# ===============================
def parallel_fft(signals):
    """Applies FFT in parallel using multiprocessing."""
    transformer = FFTTransformerGPU()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        transformed = pool.map(transformer.apply_fft, np.array_split(signals, mp.cpu_count()))
    return np.vstack(transformed)

# ===============================
# 🏢 LOAD & PREPROCESS DATASET
# ===============================
def load_and_preprocess_data():
    """Loads, preprocesses, and normalizes the ECG dataset."""
    df_filtered = load_csv_from_folder(FILTERED_FOLDER)
    df_original = load_csv_from_folder(ORIGINAL_FOLDER)
    
    if df_filtered is None or df_original is None:
        logging.error("❌ Dataset could not be loaded. Exiting...")
        exit(1)

    df = pd.concat([df_filtered, df_original]).sample(frac=1).reset_index(drop=True)
    
    X = df["Signal"].values.astype(np.float32)
    y = df["Label"].values.astype(np.int8)

    # Ensure dataset size is a multiple of 101 (for FFT windowing)
    if len(X) % 101 != 0:
        trim_size = len(X) % 101
        logging.warning(f"⚠️ Dataset size {len(X)} is not a multiple of 101. Trimming last {trim_size} samples.")
        X = X[:-trim_size]
        y = y[:-trim_size]  # 🔹 Trim `y` to match

    # 🛠 **Parallel FFT Transformation**
    X_transformed = parallel_fft(X)
    
    # Ensure `y` matches `X_transformed.shape[0]`
    y = y[:X_transformed.shape[0]]

    # 🔹 **Corrección de MinMaxScaler**
    scaler = MinMaxScaler()
    X_flattened = X_transformed.reshape(X_transformed.shape[0], -1)  # Convert to 2D
    X_scaled = scaler.fit_transform(X_flattened)  # Normalize
    X_normalized = X_scaled.reshape(X_scaled.shape[0], 101, 1)  # Convert back to 3D

    logging.info(f"✅ Final X shape: {X_normalized.shape}, Final y shape: {y.shape}")

    return train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# ===============================
# 💪 MODEL TRAINING PIPELINE
# ===============================
if __name__ == "__main__":
    logging.info("📥 Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    logging.info("⚖️ Computing class weights...")
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    
    logging.info("🧠 Initializing CNN model...")
    cnn_model = UltimateCNNModelGPU((X_train.shape[1], 1))
    
    logging.info("🚀 Training model on GPU...")
    with tf.device('/GPU:0'):
        cnn_model.train(X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, class_weight=dict(enumerate(class_weights)))
    
    logging.info("💾 Saving trained model...")
    cnn_model.save_model("/workspace/ultimate_cnn_fft_model.keras")
    
    logging.info("✅ Training complete! Model saved successfully.")