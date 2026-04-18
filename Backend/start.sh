#!/bin/bash
set -e

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing TensorFlow..."
pip install tensorflow-cpu==2.14.0 --only-binary=:all: --quiet

echo "==> Starting FastAPI on port 8000..."
uvicorn video_api:app --host 0.0.0.0 --port 8000 --workers 1 &

echo "==> Waiting for FastAPI to start..."
sleep 5

echo "==> Checking FastAPI health..."
curl -s http://localhost:8000/health || echo "FastAPI not ready yet, continuing..."

echo "==> Starting Flask..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"
