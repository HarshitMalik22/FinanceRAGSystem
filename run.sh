#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a  # Export all variables in the file
    source .env
    set +a
else
    echo "Error: .env file not found"
    exit 1
fi

# Run the application
python finance_rag_app.py
