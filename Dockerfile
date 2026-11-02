# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies needed for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn WSGI server
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app

