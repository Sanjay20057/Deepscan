FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install "numpy<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt
# Copy all backend files
COPY . .

# Create data directory
RUN mkdir -p data

# Expose ports
EXPOSE 10000
EXPOSE 8000

# Start both services
CMD ["bash", "start.sh"]
