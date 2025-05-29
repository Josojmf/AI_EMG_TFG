import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

def load_data(csv_path, segment_length=101):
    df = pd.read_csv(csv_path)
    signals = df["Signal"].values
    labels = df["Label"].values

    num_segments = len(signals) // segment_length
    X = []
    y = []

    for i in range(num_segments):
        segment = signals[i * segment_length : (i + 1) * segment_length]
        label_segment = labels[i * segment_length : (i + 1) * segment_length]
        if len(segment) == segment_length:
            X.append(segment.reshape(-1, 1))
            # Se toma la etiqueta más común en el segmento (puedes ajustar esto si quieres otra lógica)
            y.append(int(round(np.mean(label_segment))))

    X = np.array(X)
    y = np.array(y)

    return X, y

def evaluate_model(model_path, clean_data_path, noisy_data_path):
    model = tf.keras.models.load_model(model_path)
    
    for label, path in [("CLEAN", clean_data_path), ("NOISY", noisy_data_path)]:
        X, y_true = load_data(path)
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print(f"\n=== RESULTS FOR {label} DATASET ===")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1 Score:", f1_score(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        try:
            print("ROC AUC Score:", roc_auc_score(y_true, y_pred_prob))
        except:
            print("ROC AUC Score: Not computable for this case.")
        print("Classification Report:\n", classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model(
        model_path=r"C:\INFORMATICA\TFG\ModelV2CNN\spasticity_cnn_model.h5",
        clean_data_path=r"C:\INFORMATICA\TFG\ModelV3.5Fourier\Datasets\Test_Data\Labeled_Test\Clean_labeled.csv",
        noisy_data_path=r"C:\INFORMATICA\TFG\ModelV3.5Fourier\Datasets\Test_Data\Labeled_Test\Noisy_labeled.csv"
    )
