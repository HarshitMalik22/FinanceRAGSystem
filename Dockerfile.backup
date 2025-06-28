# ===== BUILD STAGE =====
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only the files needed for installing dependencies
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# ===== RUNTIME STAGE =====
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_MAX_REQUESTS=1000 \
    GUNICORN_MAX_REQUESTS_JITTER=50 \
    GUNICORN_GRACEFUL_TIMEOUT=30 \
    GUNICORN_KEEPALIVE=5

# Create a non-root user
RUN addgroup --system appgroup && \
    adduser --system --no-create-home --disabled-password --disabled-login --shell /bin/false --gecos "" --ingroup appgroup appuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories with correct permissions
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/data/cache \
    && chown -R appuser:appgroup /app/data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
CMD ["sh", "-c", "gunicorn \
  --bind 0.0.0.0:$PORT \
  --workers $GUNICORN_WORKERS \
  --threads $GUNICORN_THREADS \
  --timeout $GUNICORN_TIMEOUT \
  --max-requests $GUNICORN_MAX_REQUESTS \
  --max-requests-jitter $GUNICORN_MAX_REQUESTS_JITTER \
  --graceful-timeout $GUNICORN_GRACEFUL_TIMEOUT \
  --keep-alive $GUNICORN_KEEPALIVE \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  --chdir /app \
  finance_rag_app:server"
