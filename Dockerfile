FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libpq-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies in two stages to leverage Docker cache
COPY requirements.txt .

# First install core dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir gunicorn==21.2.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "4", "--worker-class", "gthread", "--threads", "2", "--timeout", "120", "finance_rag_app:app"]
