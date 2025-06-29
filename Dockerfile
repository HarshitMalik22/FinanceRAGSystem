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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user langchain-chroma chromadb

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    FLASK_ENV=production \
    VECTOR_STORE_DIR="/app/chroma_db" \
    CHROMA_SERVER_HOST="" \
    CHROMA_SERVER_PORT="8000" \
    CHROMA_SERVER_SSL="false"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy frontend build
COPY --from=builder /app/frontend/build /app/frontend_build

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db

# Install NLTK data and spaCy model
RUN python -m nltk.downloader popular && \
    python -m spacy download en_core_web_sm

# Set the working directory
WORKDIR /app

# Expose the port the app runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--worker-class", "gthread", \
     "--threads", "4", \
     "--timeout", "120", \
     "--preload", \
     "finance_rag_app:app"]
