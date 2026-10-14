FROM python:3.11-slim

# HF Spaces requires port 7860
EXPOSE 7860

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs as non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user

ENV PORT=7860
ENV FASTAPI_BASE_URL=http://localhost:8000

CMD ["bash", "start.sh"]
