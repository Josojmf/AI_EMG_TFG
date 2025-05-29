import os
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import csv
import re
import io
from tensorflow.keras.models import load_model
from FFTTransformer import FFTTransformer

# ==========================
# Configuración UI
# ==========================
st.set_page_config(page_title="ECG Analyzer", layout="wide")
WINDOW_SIZE = 200 

# Base directory where script is executed (Ensures compatibility in Docker & Local)
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

# Sidebar con navegación mejorada
with st.sidebar:
    st.image(os.path.join(BASE_DIR, "logo.png"), width=200)
    selected_page = st.selectbox("Navigation", 
        ["🏠 Home", "📊 Predict ECG", "📈 Model Metrics", "🖼 Confusion Matrix", "ℹ️ Model Info"]
    )

# ==========================
# Funciones Utilitarias
# ==========================
def fix_decimals_str(x):
    """Limpia múltiples decimales y elimina caracteres no numéricos."""
    x = re.sub(r'[^0-9.\-]', '', x)
    parts = x.split('.')
    return parts[0] + "." + "".join(parts[1:]) if len(parts) > 2 else x

def sanitize_and_load_csv(uploaded_file):
    """Carga y sanitiza los datos del CSV."""
    text_io = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='replace')
    reader = csv.DictReader(text_io)

    if not reader.fieldnames or 'Time' not in reader.fieldnames or 'Signal' not in reader.fieldnames:
        st.error("CSV debe contener encabezados 'Time' y 'Signal'.")
        return None, None

    times, signals = [], []
    for row in reader:
        try:
            clean_time = fix_decimals_str(row.get('Time', '').strip())
            clean_signal = fix_decimals_str(row.get('Signal', '').strip())
            t, s = float(clean_time), float(clean_signal)
            times.append(t)
            signals.append(s)
        except ValueError:
            continue

    return np.array(times, dtype=float), np.array(signals, dtype=float)

# ==========================
# Modelo y Predicción
# ==========================
def load_trained_model():
    """Carga el modelo CNN para la clasificación de ECG desde una ruta relativa."""
    model_path = os.path.join(os.path.dirname(__file__), "cnn_fft_model.keras")
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def extract_fft_features(signals, transformer):
    """Extrae características FFT con padding adecuado."""
    if len(signals) == 0:
        return np.array([])

    window_size = transformer.window_size
    pad_size = (window_size - (len(signals) % window_size)) % window_size
    padded_signals = np.pad(signals, (0, pad_size), mode='constant')

    fft_features = transformer.apply_fft(padded_signals)
    return fft_features[..., np.newaxis] if fft_features.size > 0 else np.array([])

def predict_signal(model, X):
    """Realiza predicciones usando umbral adaptativo."""
    if X.shape[0] == 0:
        return np.array([], dtype=int)

    preds = model.predict(X, batch_size=min(32, len(X)))
    avg, std = np.mean(preds), np.std(preds)
    threshold = max(0.3, avg + 0.01 * std)
    predictions = (preds > threshold).astype(int).flatten()

    return predictions

# ==========================
# Visualización de ECG
# ==========================
def plot_ecg_signal_interactive(time, signals, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signals, mode='lines', name='ECG Signal', line=dict(color='blue')))

    expanded_predictions = np.repeat(predictions, WINDOW_SIZE)[:len(time)]
    event_indices = np.where(expanded_predictions == 1)[0]

    if len(event_indices) > 0:
        fig.add_trace(go.Scatter(
            x=time[event_indices], y=signals[event_indices], mode='markers',
            name='Detected Events', marker=dict(color='red', size=6)
        ))

    fig.update_layout(title="ECG Signal with Detected Events", xaxis_title="Time (s)", yaxis_title="Signal Amplitude")
    st.plotly_chart(fig, use_container_width=True)

# ==========================
# Pantallas del Proyecto
# ==========================
def show_home():
    """Pantalla principal con informe del proyecto."""
    st.title("ECG Analyzer")
    st.markdown("""
    ## **📑 Informe del Proyecto**
    ---
    **Objetivo:**  
    Aplicación de **inteligencia artificial** para mejorar la evaluación y tratamiento de pacientes con **espasticidad** derivada de alteraciones neuromusculares tras un **daño cerebral adquirido (DCA).**

    ---
    ## **🔬 Aplicaciones**
    - Monitoreo en tiempo real con **sensores portátiles**.  
    - Implementación de **modelos predictivos** para adaptar terapias.  
    - Optimización de **recursos sanitarios** con rehabilitación eficiente.

    ---
    ## **🌍 Alcance del Proyecto**
    Aunque inicialmente centrado en **DCA**, el sistema también puede aplicarse a **otros trastornos neurológicos** con espasticidad muscular.

    ---
    ## **🔍 Abstract (English Version)**
    *Using machine learning algorithms and advanced signal processing techniques, this system aims to:*
    - Detect **muscle activation patterns**.  
    - Improve **neuromuscular function assessment**.  
    - Optimize **rehabilitation protocols** through personalized therapies.
    """)

def show_model_metrics():
    """Pantalla de métricas del modelo."""
    st.title("📊 Métricas del Modelo")
    st.write("### Precisión del Modelo:")
    st.metric(label="Accuracy", value="98.5%")
    st.write("### Pérdida del Modelo:")
    st.metric(label="Loss", value="0.02")

def show_confusion_matrix():
    """Pantalla de matriz de confusión."""
    st.title("🖼 Matriz de Confusión")
    st.image(os.path.join(BASE_DIR, "confusion_matrix.png"), caption="Matriz de Confusión del Modelo")

def show_model_info():
    """Pantalla de información del modelo."""
    st.title("ℹ️ Información del Modelo")
    st.markdown("""
    - **Arquitectura:** CNN con 3 capas convolucionales y 2 densas.
    - **Optimización:** Adam con `learning rate = 0.001`
    """)

# ==========================
# Función principal
# ==========================
def main():
    if selected_page == "🏠 Home":
        show_home()

    elif selected_page == "📊 Predict ECG":
        st.title("ECG Signal Classification UI")
        uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])

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

# ==========================
# Ejecución de la UI
# ==========================
if __name__ == "__main__":
    main()
