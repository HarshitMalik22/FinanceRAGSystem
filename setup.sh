#!/bin/bash

# Setup script for Finance RAG System deployment

echo "🚀 Starting Finance RAG System deployment..."

# Create necessary directories
echo "📂 Creating necessary directories..."
mkdir -p /opt/render/project/src/chroma_db

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=finance_rag_app.py
export FLASK_ENV=production
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Initialize the database
echo "💾 Initializing database..."
python -c "from finance_rag_app import app; app.app_context().push(); from config import Config; print('✅ Configuration loaded successfully')"

# Start the application
echo "🚀 Starting application..."
exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 finance_rag_app:app
