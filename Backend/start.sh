#!/bin/bash
set -e

echo "==> Starting FastAPI Backend Logic..."
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

echo "==> Waiting 20 seconds for CNN Model to initialize..."
sleep 20

echo "==> Starting Flask Web Gateway..."
gunicorn -w 2 -b 0.0.0.0:$PORT "flask_app:create_app()"
