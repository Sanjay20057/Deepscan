#!/bin/bash
set -e

echo "==> Installing TensorFlow..."
pip install tensorflow-cpu==2.14.0 --quiet

echo "==> Starting FastAPI on port 8000..."
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

echo "==> Starting Flask on port $PORT..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"
