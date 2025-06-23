#!/usr/bin/env python3
"""
Test script to verify Chroma DB integration and document processing.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from finance_rag_app import Config, EnhancedRAGSystem, Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_chroma.log')
    ]
)
logger = logging.getLogger(__name__)

def test_chroma_integration():
    """Test Chroma DB integration and document processing."""
    try:
        # Initialize configuration
        config = Config()
        
        # Override vector store directory for testing
        test_dir = "./test_chroma_db"
        config.VECTOR_STORE_DIR = test_dir
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = EnhancedRAGSystem(config)
        
        # Create a test document
        test_docs = [
            Document(
                page_content="This is a test document about financial analysis.",
                metadata={"source": "test", "page": 1}
            ),
            Document(
                page_content="This is another test document about investment strategies.",
                metadata={"source": "test", "page": 2}
            )
        ]
        
        # Test adding documents
        logger.info("Testing document addition...")
        rag_system._add_chunks_to_vector_store(test_docs)
        
        # Verify documents were added
        doc_count = rag_system.vector_store._collection.count()
        logger.info(f"Documents in collection: {doc_count}")
        
        # Test query
        logger.info("Testing query...")
        response = rag_system.query("What is this document about?")
        logger.info(f"Query response: {response}")
        
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")
            
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    load_dotenv()
    success = test_chroma_integration()
    sys.exit(0 if success else 1)
