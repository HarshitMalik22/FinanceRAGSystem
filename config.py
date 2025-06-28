"""Configuration settings for the Finance RAG System."""
import os
from typing import Optional

class Config:
    """Application configuration."""
    
    # Document processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    
    # API Keys (should be set via environment variables)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model configuration
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Vector store configuration
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "/app/chroma_db")
    VECTOR_STORE_COLLECTION: str = os.getenv("VECTOR_STORE_COLLECTION", "financial_docs")
    
    # Performance settings
    MAX_WORKERS: int = min(8, (os.cpu_count() or 4) * 2)
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 60  # seconds
    
    # Application settings
    DEBUG: bool = os.getenv("FLASK_ENV", "production").lower() != "production"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        required_vars = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Validate configuration on import
try:
    Config.validate()
    print("✅ Configuration validated successfully")
except ValueError as e:
    print(f"⚠️  Configuration warning: {e}")
