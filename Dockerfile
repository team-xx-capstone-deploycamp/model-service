# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install uv package manager
RUN pip install --no-cache-dir uv

# Install Python dependencies with uv
RUN uv pip install --no-cache-dir -r requirements.txt \
    && uv pip install --no-cache-dir -r requirements-dev.txt

# Copy the project code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models

# Set up entrypoint to use the wrapper script
ENTRYPOINT ["python", "web/app.py"]
