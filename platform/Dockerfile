FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CW_DATA_DIR=/var/lib/centrewatch CW_INCOMING_DIR=/data/incoming CENTREWATCH_MODELS_DIR=/models YOLO_CONFIG_DIR=/var/lib/centrewatch/yolo
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY platform/requirements.txt platform/requirements-vision.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-vision.txt
COPY platform/server ./server
COPY platform/scripts ./scripts
COPY platform/web ./web
COPY data/annotations/manifest.example.json /data/annotations/manifest.example.json
RUN useradd --create-home --uid 10001 centrewatch && mkdir -p /var/lib/centrewatch /data/incoming /models && chown -R centrewatch:centrewatch /var/lib/centrewatch /models
USER centrewatch
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8787"]
