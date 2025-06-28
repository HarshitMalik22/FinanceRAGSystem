# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with pip's new resolver
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    FLASK_ENV=production \
    VECTOR_STORE_DIR="/app/chroma_db"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Verify Python and pip are working
RUN python --version && \
    pip --version

# Install NLTK data and spaCy model
RUN python -m nltk.downloader popular && \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db

# Set the working directory
WORKDIR /app

# Verify the installation
RUN echo "PATH: $PATH" && \
    echo "Python path: $(which python)" && \
    echo "Pip path: $(which pip)" && \
    echo "Installed packages:" && \
    pip list

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
