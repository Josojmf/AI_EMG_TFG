import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from reportlab.pdfgen import canvas

MODEL_PATH = "spasticity_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_data(file):
    df = pd.read_csv(file)
    df["Signal"] = MinMaxScaler().fit_transform(df[["Signal"]])
    
    time_steps = 100
    sequences = [df["Signal"].values[i:i+time_steps] for i in range(len(df)-time_steps)]
    
    return np.array(sequences).reshape(-1, time_steps, 1), df

def generate_report(accuracy, precision, recall, f1):
    pdf_path = "model_metrics_report.pdf"
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, f"Model Accuracy: {accuracy:.2f}")
    c.drawString(100, 730, f"Precision: {precision:.2f}")
    c.drawString(100, 710, f"Recall: {recall:.2f}")
    c.drawString(100, 690, f"F1 Score: {f1:.2f}")
    c.save()
    return pdf_path

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Predict", "Model Metrics"])
    
    if page == "Upload & Predict":
        st.title("Spasticity Detection in ECG Signals")
        
        uploaded_file = st.file_uploader("Upload an ECG CSV file", type=["csv"])
        
        if uploaded_file is not None:
            data, df = preprocess_data(uploaded_file)
            predictions = model.predict(data)
            
            predicted_labels = (predictions > 0.5).astype(int).flatten()

            st.write("## Prediction Results")
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(predictions, label="Predictions", color='blue', linewidth=1)

            spasticity_mask = predictions.flatten() > 0.5
            ax.fill_between(range(len(predictions)), 0, 1, where=spasticity_mask, 
                            color='red', alpha=0.3, label="Spasticity Detected")

            ax.set_ylim(0, 1)
            ax.invert_yaxis()

            # Add labels
            ax.set_title("Spasticity Detection")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Prediction Probability")
            ax.legend()

            st.pyplot(fig)

            st.write("## Original Signal")
            st.line_chart(df["Signal"])
    
    elif page == "Model Metrics":
        st.title("Model Metrics")
        
       
        test_data_path = "test_data.csv"  
        try:
            df_test = pd.read_csv(test_data_path)
            df_test["Signal"] = MinMaxScaler().fit_transform(df_test[["Signal"]])
            test_sequences = [df_test["Signal"].values[i:i+100] for i in range(len(df_test)-100)]
            X_test = np.array(test_sequences).reshape(-1, 100, 1)
            y_test = df_test["Label"].values[-len(X_test):]
            
            # Get Predictions
            predictions = model.predict(X_test)
            predicted_labels = (predictions > 0.5).astype(int).flatten()

            # Calculate Metrics
            accuracy = accuracy_score(y_test, predicted_labels)
            precision = precision_score(y_test, predicted_labels)
            recall = recall_score(y_test, predicted_labels)
            f1 = f1_score(y_test, predicted_labels)
            
            st.write(f"### Model Metrics:")
            st.write(f"- **Accuracy**: {accuracy:.2f}")
            st.write(f"- **Precision**: {precision:.2f}")
            st.write(f"- **Recall**: {recall:.2f}")
            st.write(f"- **F1 Score**: {f1:.2f}")

            # Generate and provide PDF report
            report_path = generate_report(accuracy, precision, recall, f1)
            with open(report_path, "rb") as f:
                st.download_button("Download Report", f, file_name="model_metrics_report.pdf")
        except Exception as e:
            st.write("⚠ Error loading test data. Make sure a valid test CSV file exists.")

if __name__ == "__main__":
    main()
