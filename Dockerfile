FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which slim images lack
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/model.joblib \
    LOKY_MAX_CPU_COUNT=2 \
    PYTHONUNBUFFERED=1

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data

# Train inside the image so the model always matches the library versions.
# Uses data/train.csv if present, otherwise the synthetic demo data.
RUN python scripts/train_final.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
