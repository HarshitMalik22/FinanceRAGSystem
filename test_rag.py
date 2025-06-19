#!/usr/bin/env python3
"""
Test script for the Finance RAG System.
Run this to verify your setup and test basic functionality.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    load_dotenv()
    
    required_vars = [
        "LLM_PROVIDER",
        "GOOGLE_API_KEY" if os.getenv("LLM_PROVIDER") == "google" else "OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please check your .env file and make sure all required variables are set.")
        return False
    
    logger.info("✓ Environment variables are properly configured")
    return True

def test_rag_system():
    """Test the RAG system with a sample document and query."""
    from finance_rag_app import Config, EnhancedRAGSystem
    import tempfile
    import requests
    
    try:
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Download a sample PDF
            response = requests.get("https://www.africau.edu/images/default/sample.pdf")
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        logger.info("Testing RAG system with sample document...")
        
        # Initialize the RAG system
        config = Config()
        rag_system = EnhancedRAGSystem(config)
        
        # Add the test document
        rag_system.add_documents_batch([temp_path], ["test_document.pdf"])
        
        # Test a query
        query = "What is this document about?"
        logger.info(f"Query: {query}")
        response = rag_system.query(query)
        
        logger.info(f"Response: {response}")
        logger.info("✓ RAG system test completed successfully")
        
        # Clean up
        os.unlink(temp_path)
        return True
        
    except Exception as e:
        logger.error(f"Error testing RAG system: {str(e)}", exc_info=True)
        return False
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == "__main__":
    logger.info("Starting Finance RAG System Test...")
    
    if not check_environment():
        sys.exit(1)
    
    if not test_rag_system():
        logger.error("❌ RAG system test failed")
        sys.exit(1)
    
    logger.info("✅ All tests completed successfully")
