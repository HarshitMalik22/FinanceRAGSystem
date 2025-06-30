"""Configuration settings for the Finance RAG System."""
import os
from typing import Optional

class Config:
    """Application configuration with rate limiting and retry settings."""
    
    # Document processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GOOGLE_EMBEDDING_MODEL: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    
    # Vector store configuration
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./chroma_db")
    VECTOR_STORE_COLLECTION: str = os.getenv("VECTOR_STORE_COLLECTION", "financial_docs")
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))  # Number of documents to retrieve during RAG
    
    # Rate limiting and retry settings
    MAX_WORKERS: int = min(4, (os.cpu_count() or 2) * 2)  # Reduced workers to avoid rate limits
    MAX_RETRIES: int = 3  # Maximum number of retry attempts
    REQUEST_TIMEOUT: int = 30  # Request timeout in seconds
    RATE_LIMIT_DELAY: float = 1.0  # Initial delay in seconds
    MAX_RATE_LIMIT_DELAY: float = 60.0  # Maximum delay in seconds
    RATE_LIMIT_BACKOFF_FACTOR: float = 2.0  # Exponential backoff factor
    
    # Application settings
    DEBUG: bool = os.getenv("FLASK_ENV", "production").lower() != "production"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Cache settings
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./document_cache")
    
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
