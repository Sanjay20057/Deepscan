#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "==> Upgrading pip and essential tools..."
pip install --upgrade pip setuptools wheel

echo "==> Installing compatible NumPy and TensorFlow..."
# We force numpy<2.0.0 here to fix the 'AttributeError: _ARRAY_API not found'
pip install "numpy<2.0.0" tensorflow-cpu==2.14.0 --only-binary=:all: --quiet

echo "==> Starting FastAPI Backend Logic..."
# Running in the background (&) so the script can continue to Flask
# uvicorn handles the CNN model loading in video_api.py
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

# CRITICAL: Give the CNN model time to load. TensorFlow is heavy and slow on Render's free tier.
# Without this, Flask might try to connect to an 'offline' FastAPI
echo "==> Waiting 20 seconds for CNN Model to initialize..."
sleep 20

echo "==> Starting Flask Web Gateway..."
# Using -w 1 (1 worker) instead of 2 to save RAM on Render's free tier
# This binds to the PORT variable (defaulting to 10000)
exec gunicorn -w 1 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"
