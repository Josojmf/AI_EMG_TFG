import numpy as np
import os
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from CNNModel import UltimateCNNModel
from FFTTransformer import FFTTransformer
import logging

tf.keras.mixed_precision.set_global_policy('mixed_float16')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FILTERED_FOLDER = r"C:\INFORMATICA\TFG\ModelV3.5Fourier\Datasets\Train_Data\filtered"
ORIGINAL_FOLDER = r"C:\INFORMATICA\TFG\ModelV3.5Fourier\Datasets\Train_Data\original"

def load_csv_from_folder(folder, chunk_size=50000):
    dataframes = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk['Signal'] = pd.to_numeric(chunk['Signal'], errors='coerce')
                chunk['Label'] = pd.to_numeric(chunk['Label'], errors='coerce', downcast='integer')
                chunk.dropna(subset=['Signal', 'Label'], inplace=True)
                chunk['Signal'] = chunk['Signal'].astype('float32')
                chunk['Label'] = chunk['Label'].astype('int8')
                dataframes.append(chunk)

    return pd.concat(dataframes, ignore_index=True) if dataframes else None

def parallel_fft(signals):
    transformer = FFTTransformer()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        transformed = pool.map(transformer.apply_fft, np.array_split(signals, mp.cpu_count()))
    return np.vstack(transformed)

def load_and_preprocess_data():
    df_filtered = load_csv_from_folder(FILTERED_FOLDER)
    df_original = load_csv_from_folder(ORIGINAL_FOLDER)
    df = pd.concat([df_filtered, df_original]).sample(frac=1).reset_index(drop=True)

    X = df["Signal"].values.astype(np.float32)
    y = df["Label"].values.astype(np.int8)

    X_transformed = parallel_fft(X)
    y = y[:X_transformed.shape[0]]

    X_normalized = MinMaxScaler().fit_transform(X_transformed)
    return train_test_split(X_normalized, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cnn_model = UltimateCNNModel((X_train.shape[1], 1))
    cnn_model.train(X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, class_weight=dict(enumerate(class_weights)))
    cnn_model.save_model("ultimate_cnn_fft_model.keras")
