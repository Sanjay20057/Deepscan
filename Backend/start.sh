#!/bin/bash
set -e

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing Pillow..."
pip install Pillow==9.5.0 --only-binary=:all:

echo "==> Installing OpenCV..."
pip install opencv-python-headless==4.8.0.76 --only-binary=:all:

echo "==> Installing TensorFlow..."
pip install tensorflow-cpu==2.14.0 --only-binary=:all:

echo "==> Starting FastAPI on port 8000..."
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

echo "==> Starting Flask on port $PORT..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"
