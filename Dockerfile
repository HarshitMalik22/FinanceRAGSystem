FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

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

# Install Python dependencies first to leverage Docker cache
COPY requirements.txt .

# Install pip and core dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY finance_rag_app.py .
COPY config.py .
COPY document_processor.py .
COPY rag_system.py .

# Create necessary directories
RUN mkdir -p /app/chroma_db

# Expose the port the app runs on
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:10000/health || exit 1

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "2", \
     "--worker-class", "gthread", \
     "--threads", "4", \
     "--timeout", "120", \
     "--preload", \
     "finance_rag_app:app"]
