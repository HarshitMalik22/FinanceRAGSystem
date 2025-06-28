web: gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 300 --worker-class gthread --log-level=info --worker-tmp-dir /dev/shm finance_rag_app:app
