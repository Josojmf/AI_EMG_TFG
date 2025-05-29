# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Exposición del puerto dinámico asignado por Heroku
ENV PORT $PORT

# Comando de inicio (Streamlit tomará el puerto de la variable de entorno PORT)
CMD streamlit run ui.py --server.port=$PORT --server.address=0.0.0.0
