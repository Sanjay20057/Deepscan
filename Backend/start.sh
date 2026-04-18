#!/bin/bash
set -e

# Start FastAPI on port 8000 in the background
uvicorn video_api:app --host 0.0.0.0 --port 8000 &

# Start Flask (Gunicorn) on the PORT Render assigns (default 10000)
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} "flask_app:create_app()"