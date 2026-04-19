#!/bin/bash
set -e

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing TensorFlow..."
pip install tensorflow-cpu==2.14.0 --only-binary=:all: --quiet

echo "==> Starting FastAPI..."
# Use 0.0.0.0 so it's reachable internally
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

# CRITICAL: Give the CNN model time to load (TensorFlow is slow)
echo "==> Waiting 15 seconds for CNN Model to initialize..."
sleep 15

echo "==> Starting Flask..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"
