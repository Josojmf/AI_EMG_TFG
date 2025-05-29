# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make sure the model file is copied into the container
COPY spasticity_cnn_model.h5 /app/spasticity_cnn_model.h5

# Expose the port the Streamlit app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
