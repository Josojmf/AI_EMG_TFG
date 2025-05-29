# Use Python 3.9 as base image
FROM python:3.9-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libxml2-dev \
    libxslt-dev \
    dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (for caching efficiency)
COPY requirements.txt ./ 

# Convert requirements file to Unix format (fix potential CRLF issues)
RUN dos2unix requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 🔥 Install TensorFlow 2.19.0 (to match local version)
RUN pip install --no-cache-dir tensorflow==2.19.0

RUN pip install --no-cache-dir --upgrade --force-reinstall streamlit==1.42.0


# Install other dependencies
RUN pip install --no-cache-dir \
    protobuf==4.25.3 \
    plotly==5.15.0 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    numpy==1.24.3 \
    Flask==2.2.3 \
    Flask-SocketIO==5.4.1 \
    tensorflow-io-gcs-filesystem==0.31.0

# 🔥 Copy Model, Assets, and Utility Files
COPY cnn_fft_model.keras /app/
COPY logo.png /app/
COPY confusion_matrix.png /app/
COPY FFTTransformer.py /app/

# Copy the rest of the application files
COPY . .

# Ensure Streamlit and other tools are available in PATH
ENV PATH="/root/.local/bin:$PATH"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Expose the correct port
EXPOSE 8501

# Run the Streamlit UI
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
