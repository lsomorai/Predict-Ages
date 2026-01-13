# Dockerfile for Age Prediction Application
# Supports both CPU and GPU inference

# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (with extended timeout for large packages like PyTorch)
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY scripts/ ./scripts/

# Create directories for checkpoints and outputs
RUN mkdir -p /app/checkpoints /app/outputs /app/exports

# Expose Gradio port
EXPOSE 7860

# Set the default command to run the Gradio app
CMD ["python", "app.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1
