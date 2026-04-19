FROM python:3.11-slim

WORKDIR /app

# Hugging Face Spaces runs on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY Backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install "numpy<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY Backend/ .

# Create data directory
RUN mkdir -p data

# Hugging Face requires port 7860
EXPOSE 7860

# Start both services
CMD ["bash", "start.sh"]
