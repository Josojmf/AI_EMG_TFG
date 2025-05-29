import os
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import csv
import re
import io
from tensorflow.keras.models import load_model
from FFTTransformer import FFTTransformer
from CNNModel import MultiHeadSelfAttention  # <-- IMPORTANTE

# ==========================
# Configuración UI
# ==========================
st.set_page_config(page_title="EMG Spasm Analyzer", layout="wide")
WINDOW_SIZE = 200 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSS personalizado para mejorar la apariencia
st.markdown("""
    <style>
        .stSidebar {
            background-color: #1e1e1e;
        }
        .css-1d391kg {
            padding-top: 0 !important;
        }
        .sidebar-content {
            padding: 20px;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
        }
        .spinner {
            font-size: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image(os.path.join(BASE_DIR, "logo.png"), width=200)
    selected_page = st.selectbox("Navigation", 
        ["🏠 Home", "📊 Predict EMG", "📈 Model Metrics", "🖼 Confusion Matrix", "ℹ️ Model Info"]
    )

 
def fix_decimals_str(x):
    x = re.sub(r'[^0-9.\-]', '', x)
    parts = x.split('.')
    return parts[0] + "." + "".join(parts[1:]) if len(parts) > 2 else x

def sanitize_and_load_csv(uploaded_file):
    text_io = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='replace')
    reader = csv.DictReader(text_io)
    if not reader.fieldnames or 'Time' not in reader.fieldnames or 'Signal' not in reader.fieldnames:
        st.error("CSV debe contener encabezados 'Time' y 'Signal'.")
        return None, None

    times, signals = [], []
    for row in reader:
        try:
            t = float(fix_decimals_str(row['Time']))
            s = float(fix_decimals_str(row['Signal']))
            times.append(t)
            signals.append(s)
        except ValueError:
            continue

    return np.array(times, dtype=float), np.array(signals, dtype=float)


def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "ultimate_cnn_fft_model.keras")
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path, compile=False, custom_objects={"MultiHeadSelfAttention": MultiHeadSelfAttention})
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def extract_fft_features(signals, transformer):
    if len(signals) == 0:
        return np.array([])

    window_size = transformer.window_size
    pad_size = (window_size - (len(signals) % window_size)) % window_size
    padded_signals = np.pad(signals, (0, pad_size), mode='constant')

    fft_features = transformer.apply_fft(padded_signals)
    return fft_features[..., np.newaxis] if fft_features.size > 0 else np.array([])

def predict_signal(model, X):
    if X.shape[0] == 0:
        return np.array([], dtype=int)
    preds = model.predict(X, batch_size=min(32, len(X)))
    avg, std = np.mean(preds), np.std(preds)
    threshold = max(0.3, avg + 0.01 * std)
    return (preds > threshold).astype(int).flatten()



def plot_ecg_signal_interactive(time, signals, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signals, mode='lines', name='EMG Signal', line=dict(color='blue')))

    expanded_predictions = np.repeat(predictions, WINDOW_SIZE)[:len(time)]
    event_indices = np.where(expanded_predictions == 1)[0]

    if len(event_indices) > 0:
        fig.add_trace(go.Scatter(
            x=time[event_indices], y=signals[event_indices], mode='markers',
            name='Detected Events', marker=dict(color='red', size=6)
        ))

    fig.update_layout(title="EMG Signal with Detected Events",
                      xaxis_title="Time (s)", yaxis_title="Signal Amplitude")
    st.plotly_chart(fig, use_container_width=True)


def show_home():
    st.title("EMG Analyzer")
    st.markdown("""
    ## **📑 Informe del Proyecto**
    ---
    **Objetivo:**  
    Aplicación de **inteligencia artificial** para mejorar la evaluación y tratamiento de pacientes con **espasticidad** derivada de alteraciones neuromusculares tras un **daño cerebral adquirido (DCA).**
    ...
    """)

def show_model_metrics():
    st.title("📊 Métricas del Modelo")
    st.metric(label="Accuracy", value="98.5%")
    st.metric(label="Loss", value="0.02")

def show_confusion_matrix():
    st.title("🖼 Matriz de Confusión")
    st.image(os.path.join(BASE_DIR, "confusion_matrix.png"), caption="Matriz de Confusión del Modelo")

def show_model_info():
    st.title("ℹ️ Información del Modelo")
    st.markdown("""
    - **Arquitectura:** CNN con 3 capas convolucionales y 2 densas.
    - **Optimización:** Adam con `learning rate = 0.001`
    """)


def main():
    if selected_page == "🏠 Home":
        show_home()
    elif selected_page == "📊 Predict EMG":
        st.title("EMG Signal Classification UI")
        uploaded_file = st.file_uploader("Upload EMG CSV File", type=["csv"])
        if uploaded_file:
            with st.spinner("🔄 Procesando..."):
                time, signals = sanitize_and_load_csv(uploaded_file)
                if time is None or signals is None:
                    st.error("⚠️ Archivo inválido.")
                else:
                    transformer = FFTTransformer()
                    features = extract_fft_features(signals, transformer)
                    if features.size == 0:
                        st.error("⚠️ No se extrajeron características.")
                    else:
                        model = load_trained_model()
                        predictions = predict_signal(model, features)
                        plot_ecg_signal_interactive(time, signals, predictions)
    elif selected_page == "📈 Model Metrics":
        show_model_metrics()
    elif selected_page == "🖼 Confusion Matrix":
        show_confusion_matrix()
    elif selected_page == "ℹ️ Model Info":
        show_model_info()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))
    import streamlit.web.bootstrap as bootstrap
    bootstrap.run('ui.py', f'ui.py', [], port=port)

