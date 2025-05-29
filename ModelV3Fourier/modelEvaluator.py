import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    jaccard_score, log_loss, hamming_loss, fowlkes_mallows_score, classification_report
)
from FFTTransformer import FFTTransformer
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "cnn_FFT.keras"  
REPORT_FILE = "evaluation_report.txt"
CONF_MATRIX_FILE = "confusion_matrix.png"
WINDOW_SIZE = 100  
FFT_SIZE = 704  
TEST_DATA_DIR = "C:\\INFORMATICA\\TFG\\ModelV3\\Datasets\\Test_Data\\Labeled_Test"

def load_and_process_signals(directory, window_size=WINDOW_SIZE):
    """Loads and segments ECG signals from labeled test CSV files."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    all_signals, all_labels = [], []

    if not files:
        print("❌ No test files found in:", directory)
        return np.array([]), np.array([])

    print(f"📂 Found {len(files)} test files. Loading...")

    for file in files:
        print(f"🔍 Processing file: {file}")
        df = pd.read_csv(file)

        # Debugging: Show column names
        print(f"📝 Columns found: {df.columns.tolist()}")

        if 'Signal' not in df.columns or 'Label' not in df.columns:
            print(f"⚠️ Skipping {file}: Missing 'Signal' or 'Label' column.")
            continue

        # Debugging: Show first few rows
        print(df.head())

        # Segment data into windows
        num_segments = len(df) // window_size
        for i in range(num_segments):
            segment = df.iloc[i * window_size:(i + 1) * window_size]
            signal = segment['Signal'].values
            label = segment['Label'].mode()[0]  
            all_signals.append(signal)
            all_labels.append(label)

    print(f"✅ Loaded {len(all_signals)} signal segments.")
    return np.array(all_signals), np.array(all_labels)

def extract_fft_features(signals):
    """Extracts FFT features and ensures they match CNN's expected shape."""
    transformer = FFTTransformer(lowcut=20, highcut=450, fs=1000)
    fft_results = [transformer.apply_fft(signal) for signal in signals if len(signal) >= transformer.order]

    if not fft_results:
        print("❌ No valid FFT results obtained.")
        return None

    frequencies, powers = zip(*fft_results)
    features = np.column_stack([frequencies, powers])

    # Ensure FFT output matches expected size
    if features.shape[1] > FFT_SIZE:
        features = features[:, :FFT_SIZE]  
    elif features.shape[1] < FFT_SIZE:
        pad_width = FFT_SIZE - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), 'constant')

    print(f"✅ Extracted FFT features with shape: {features.shape}")
    return features.reshape(features.shape[0], FFT_SIZE, 1)

def evaluate_model():
    """Loads the trained model and evaluates it on labeled test data."""
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found. Train the model first!")
        return

    print("📂 Loading test data...")
    signals, labels = load_and_process_signals(TEST_DATA_DIR)

    if len(signals) == 0:
        print("❌ No valid test signals found. Check dataset format.")
        return

    features = extract_fft_features(signals)

    if features is None or len(features) == 0:
        print("❌ No valid test features found. Check your dataset.")
        return

    print("📂 Loading trained model...")
    model = load_model(MODEL_PATH)

    print("🔍 Evaluating the model...")
    y_pred = (model.predict(features) > 0.7).astype(int).ravel()  # Flatten predictions to 1D

    # Compute metrics
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    roc_auc = roc_auc_score(labels, y_pred)
    mcc = matthews_corrcoef(labels, y_pred)
    balanced_acc = balanced_accuracy_score(labels, y_pred)
    cohen_kappa = cohen_kappa_score(labels, y_pred)
    jaccard = jaccard_score(labels, y_pred)
    fmi = fowlkes_mallows_score(labels, y_pred)
    hamming = hamming_loss(labels, y_pred)
    logloss = log_loss(labels, y_pred)

    conf_matrix = confusion_matrix(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True)

    with open(REPORT_FILE, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Cohen's Kappa: {cohen_kappa:.4f}\n")
        f.write(f"Jaccard Similarity: {jaccard:.4f}\n")
        f.write(f"Fowlkes-Mallows Index: {fmi:.4f}\n")
        f.write(f"Hamming Loss: {hamming:.6f}\n")
        f.write(f"Log Loss: {logloss:.4f}\n\n")

    print(f"✅ Evaluation report saved: {REPORT_FILE}")
    plot_confusion_matrix(conf_matrix)

def plot_confusion_matrix(conf_matrix):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Normal", "Spastic"])
    plt.yticks([0, 1], ["Normal", "Spastic"])
    
    # Add values inside the confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="red")

    plt.savefig(CONF_MATRIX_FILE)
    print(f"📊 Confusion matrix saved: {CONF_MATRIX_FILE}")

if __name__ == "__main__":
    evaluate_model()
