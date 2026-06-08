FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist data, models and logs outside the container via volumes
VOLUME ["/app/data", "/app/models", "/app/logs"]

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
