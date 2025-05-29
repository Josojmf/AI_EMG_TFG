FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=${PORT:-8501}

EXPOSE $PORT

CMD streamlit run ui.py --server.port=$PORT --server.address=0.0.0.0