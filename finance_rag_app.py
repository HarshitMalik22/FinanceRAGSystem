#!/usr/bin/env python3
"""
Enhanced Finance RAG System - A scalable financial document analysis system
using LangChain with proper LLM API integration and optimized for large documents.
"""

# Load environment variables first
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get the absolute path to the directory containing this script
script_dir = Path(__file__).parent.resolve()

# Load .env file from the same directory as the script
env_path = script_dir / '.env'
print(f"Loading environment from: {env_path}")
load_dotenv(env_path, override=True)

# Verify the API keys are loaded and valid
groq_key = os.getenv('GROQ_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')

print(f"GROQ_API_KEY loaded: {'Yes' if groq_key else 'No'}")
print(f"GOOGLE_API_KEY loaded: {'Yes' if google_key else 'No'}")

# Validate API keys before proceeding
if not groq_key or groq_key.strip() == '':
    print("❌ ERROR: GROQ_API_KEY is not set or empty in .env file")
    print("Please add a valid GROQ API key to the .env file")
    sys.exit(1)

if not google_key or google_key.strip() == '':
    print("❌ ERROR: GOOGLE_API_KEY is not set or empty in .env file")
    print("Please add a valid Google API key to the .env file")
    sys.exit(1)

print("✅ API keys validation passed")

import uuid
import json
import time
import logging
import tempfile
import threading
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Flask imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# File processing
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

# Additional imports
import PyPDF2
import pandas as pd
from langchain_core.runnables import RunnableLambda
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    """Configuration class for the RAG system."""
    
    def __init__(self):
        # API Keys
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # LLM Configuration
        self.LLM_PROVIDER = "groq"
        self.GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.GOOGLE_EMBEDDING_MODEL = os.getenv('GOOGLE_EMBEDDING_MODEL', 'models/embedding-001')
        self.GOOGLE_API_BASE = os.getenv('GOOGLE_API_BASE', 'https://generativelanguage.googleapis.com')
        
        # Document Processing
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', '5'))
        
        # Vector Store
        self.VECTOR_STORE_DIR = os.getenv('VECTOR_STORE_DIR', './vector_store')
        self.VECTOR_STORE_COLLECTION = os.getenv('VECTOR_STORE_COLLECTION', 'finance_docs')
        
        # Caching
        self.ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
        self.CACHE_DIR = os.getenv('CACHE_DIR', './cache')
        
        # Rate Limiting
        self.RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '1.0'))
        self.RATE_LIMIT_BACKOFF_FACTOR = float(os.getenv('RATE_LIMIT_BACKOFF_FACTOR', '2.0'))
        self.MAX_RATE_LIMIT_DELAY = float(os.getenv('MAX_RATE_LIMIT_DELAY', '60.0'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
        self.RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))
        self.REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
        
        # Processing
        self.EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '10'))

# DocumentProcessor class
class DocumentProcessor:
    """Enhanced document processor for handling large volumes of documents."""
    
    def __init__(self, config: Config):
        """Initialize the DocumentProcessor with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self._cache = {}  # In-memory document cache
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Create cache directory if caching is enabled
        if self.config.ENABLE_CACHING:
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)
            logger.info(f"Document caching enabled. Cache directory: {self.config.CACHE_DIR}")
        else:
            logger.info("Document caching is disabled")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_cached_documents(self, file_hash: str) -> Optional[List[Document]]:
        """Retrieve cached processed documents."""
        if not self.config.ENABLE_CACHING:
            return None
            
        cache_file = os.path.join(self.config.CACHE_DIR, f"{file_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached documents: {e}")
        return None
    
    def _cache_documents(self, file_hash: str, documents: List[Document]):
        """Cache processed documents."""
        if not self.config.ENABLE_CACHING:
            return
            
        cache_file = os.path.join(self.config.CACHE_DIR, f"{file_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(documents, f)
        except Exception as e:
            logger.warning(f"Failed to cache documents: {e}")
    
    def load_document(self, file_path: str, file_name: str = None) -> List[Document]:
        """
        Load and process a single document with optimized performance.
        
        Args:
            file_path: Path to the document file
            file_name: Optional custom file name
            
        Returns:
            List of Document objects or None if loading fails
        """
        file_name = file_name or os.path.basename(file_path)
        logger.info(f"Loading document: {file_name}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        # Check cache first
        file_hash = self._get_file_hash(file_path)
        if self.config.ENABLE_CACHING and file_hash in self._cache:
            logger.debug(f"Loading from cache: {file_name}")
            return self._cache[file_hash]
            
        try:
            # Get file size for optimization
            file_size = os.path.getsize(file_path)
            
            # Map file extensions to loaders with optimized settings
            loader_map = {
                '.pdf': lambda p: PyPDFLoader(p),
                '.docx': lambda p: Docx2txtLoader(p),
                '.doc': lambda p: Docx2txtLoader(p),
                '.xls': lambda p: UnstructuredExcelLoader(p, mode="elements"),
                '.xlsx': lambda p: UnstructuredExcelLoader(p, mode="elements"),
                '.csv': lambda p: CSVLoader(p, csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': None
                }),
                '.txt': lambda p: TextLoader(p, encoding='utf-8', autodetect_encoding=True),
                '.md': lambda p: TextLoader(p, encoding='utf-8')
            }
            
            # Get file extension and validate
            _, ext = os.path.splitext(file_path.lower())
            if ext not in loader_map:
                logger.warning(f"Unsupported file type: {file_path}")
                return []
                
            # Load documents with progress tracking
            start_time = time.time()
            loader = loader_map[ext](file_path)
            
            # Process in chunks for large files
            if file_size > 10 * 1024 * 1024:  # > 10MB
                logger.info(f"Processing large file in chunks: {file_name} ({file_size/1024/1024:.1f}MB)")
                docs = []
                chunk_num = 0
                while True:
                    try:
                        chunk_docs = loader.load()
                        if not chunk_docs:
                            break
                        docs.extend(chunk_docs)
                        chunk_num += 1
                        if chunk_num % 5 == 0:
                            logger.debug(f"Processed {chunk_num} chunks, {len(docs)} documents so far...")
                    except Exception as e:
                        logger.warning(f"Error processing chunk {chunk_num} of {file_name}: {str(e)}")
                        break
            else:
                docs = loader.load()
            
            # Add metadata to all documents
            processed_docs = []
            for doc in docs:
                try:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata.update({
                        'source': file_name,
                        'file_hash': file_hash,
                        'processed_at': datetime.now().isoformat()
                    })
                    processed_docs.append(doc)
                except Exception as meta_error:
                    logger.warning(f"Error adding metadata to document: {str(meta_error)}")
                    processed_docs.append(doc)  # Still add the document
            
            if not processed_docs:
                logger.warning(f"No valid documents found in {file_name}")
                return []
            
            # Cache the loaded documents if enabled
            if self.config.ENABLE_CACHING:
                self._cache[file_hash] = processed_docs
                
            load_time = time.time() - start_time
            logger.info(
                f"Loaded {len(processed_docs)} documents from {file_name} "
                f"in {load_time:.2f}s ({len(processed_docs)/max(load_time, 0.1):.1f} docs/s)"
            )
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error loading document {file_name}: {str(e)}", exc_info=True)
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with progress tracking."""
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
            
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        all_splits = []
        for doc in tqdm(documents, desc="Splitting documents"):
            try:
                splits = self.text_splitter.split_documents([doc])
                all_splits.extend(splits)
            except Exception as e:
                logger.warning(f"Error splitting document: {e}")
                continue
        
        logger.info(f"Created {len(all_splits)} document chunks")
        return all_splits

class EnhancedRAGSystem:
    """Enhanced RAG system with proper LLM API integration and scalability."""
    
    def __init__(self, config: Config):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        
        # Initialize LLM and embeddings first with optimized settings
        self._initialize_llm_and_embeddings()
        
        # Initialize vector store with optimized settings
        self._initialize_vector_store()
        
        # Initialize QA chain with optimized settings
        self._initialize_qa_chain()
        
        # Optimized thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=min(4, (os.cpu_count() or 2) * 2),  # Limit max workers
            thread_name_prefix='rag_worker'
        )
        
        # Cache for embeddings to avoid reprocessing
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
    
    def _initialize_llm_and_embeddings(self):
        """Initialize LLM and embeddings with proper error handling."""
        try:
            # Initialize Google embeddings
            logger.info(f"Initializing Google embeddings with model: {self.config.GOOGLE_EMBEDDING_MODEL}")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.GOOGLE_EMBEDDING_MODEL,
                google_api_key=self.config.GOOGLE_API_KEY
            )
            
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embeddings test successful. Vector dimension: {len(test_embedding)}")
            
            # Initialize Google LLM
            logger.info("Initializing Google Generative AI LLM")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.1,
                max_tokens=1024
            )
            
            # Test LLM
            test_response = self.llm.invoke("Hello, this is a test.")
            logger.info(f"LLM test successful. Response: {test_response.content[:50]}...")
            
            logger.info("Successfully initialized LLM and embeddings")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM and embeddings: {str(e)}")
            raise ValueError("Failed to initialize LLM and embeddings. Please check your API keys.")
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store in either local or server mode based on configuration."""
        try:
            # Check if we should use ChromaDB server
            chroma_server_host = os.getenv("CHROMA_SERVER_HOST")
            
            if chroma_server_host:
                # Server mode
                from chromadb.config import Settings
                chroma_server_port = os.getenv("CHROMA_SERVER_PORT", "8000")
                chroma_server_ssl = os.getenv("CHROMA_SERVER_SSL", "false").lower() == "true"
                
                logger.info(f"Connecting to ChromaDB server at {chroma_server_host}:{chroma_server_port}")
                
                chroma_server_settings = Settings(
                    chroma_api_impl="rest",
                    chroma_server_host=chroma_server_host,
                    chroma_server_http_port=chroma_server_port,
                    chroma_server_ssl_enabled=chroma_server_ssl,
                )
                
                # Add authentication if provided
                auth_creds = os.getenv("CHROMA_SERVER_AUTH_CREDENTIALS")
                if auth_creds:
                    chroma_server_settings.chroma_client_auth_credentials = auth_creds
                
                self.vector_store = Chroma(
                    collection_name=self.config.VECTOR_STORE_COLLECTION,
                    embedding_function=self.embeddings,
                    client_settings=chroma_server_settings
                )
                logger.info("Successfully connected to ChromaDB server")
                
            else:
                # Local persistence mode (for development)
                logger.info(f"Initializing local Chroma vector store in: {self.config.VECTOR_STORE_DIR}")
                os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True)
                
                self.vector_store = Chroma(
                    persist_directory=self.config.VECTOR_STORE_DIR,
                    embedding_function=self.embeddings,
                    collection_name=self.config.VECTOR_STORE_COLLECTION
                )
                logger.info("Successfully initialized local Chroma vector store")
            
            # Verify the collection is accessible
            try:
                collection = self.vector_store._collection
                if collection:
                    count = collection.count()
                    logger.info(f"Vector store contains {count} documents")
                else:
                    logger.warning("Failed to verify collection in vector store")
            except Exception as e:
                logger.warning(f"Could not get document count: {str(e)}")
            
        except Exception as e:
            error_msg = f"Failed to initialize vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(f"Vector store initialization failed: {str(e)}")
    
    def _initialize_qa_chain(self):
        """Initialize QA chain with enhanced prompt for structured responses."""
        template = """You are an expert financial analyst. Your task is to analyze financial documents and provide clear, structured responses. 

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Analyze the provided context thoroughly before responding
2. Structure your response with clear sections using markdown formatting
3. Start with a concise 1-2 sentence summary of the key points
4. Follow with detailed information in a well-organized format
5. For numerical data, be precise and include units/currency
6. If the information is not in the context, clearly state: "I don't have enough information to answer this question based on the provided documents"
7. When referencing specific details, include the source document name and page number if available

FORMAT YOUR RESPONSE AS FOLLOWS:
```markdown
## Summary
[Provide a 1-2 sentence summary of the key points]

## Key Details
- [List the most important points with clear formatting]
- [Use bullet points for better readability]
- [Include specific numbers, dates, and terms]

## Additional Information
[Provide any additional relevant details]
[Use subsections if needed]

## Sources
- [Document 1] - [Page X]
- [Document 2] - [Page Y]
```

YOUR RESPONSE:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever with MMR for diversity
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.TOP_K_RETRIEVAL,
                "fetch_k": self.config.TOP_K_RETRIEVAL * 2
            }
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        logger.info("QA chain initialized successfully")

    def add_documents_batch(self, file_paths: List[str], file_names: List[str] = None) -> Dict[str, Any]:
        """Process and add multiple documents to the vector store.
        
        Args:
            file_paths: List of file paths to process
            file_names: Optional list of original file names
            
        Returns:
            Dict containing processing results
        """
        if not file_paths:
            return {"status": "error", "message": "No files provided"}
            
        if not file_names:
            file_names = [os.path.basename(p) for p in file_paths]
            
        results = {
            "status": "success",
            "processed_files": [],
            "total_documents": 0,
            "errors": []
        }
        
        all_chunks = []
        
        for file_path, file_name in zip(file_paths, file_names):
            try:
                logger.info(f"Processing file: {file_name}")
                
                # Process the document using your document processor
                documents = self.document_processor.load_document(file_path, file_name)
                
                if not documents:
                    error_msg = f"No valid content found in {file_name}"
                    logger.warning(error_msg)
                    results["errors"].append({"file": file_name, "error": error_msg})
                    continue
                
                # Split documents into chunks
                chunks = self.document_processor.split_documents(documents)
                
                if not chunks:
                    error_msg = f"No chunks generated from {file_name}"
                    logger.warning(error_msg)
                    results["errors"].append({"file": file_name, "error": error_msg})
                    continue
                
                # Add chunks to the batch
                all_chunks.extend(chunks)
                
                # Update results
                doc_count = len(chunks)
                results["processed_files"].append({
                    "file_name": file_name,
                    "document_count": doc_count
                })
                results["total_documents"] += doc_count
                
                logger.info(f"Successfully processed {file_name}: {doc_count} document chunks prepared")
                
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results["errors"].append({"file": file_name, "error": str(e)})
                results["status"] = "partial_success"
        
        # Add all chunks to vector store in one batch
        if all_chunks:
            try:
                logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
                
                # Get initial count
                try:
                    initial_count = self.vector_store._collection.count()
                    logger.info(f"Vector store initial count: {initial_count}")
                except:
                    initial_count = 0
                    logger.warning("Could not get initial vector store count")
                
                # Add documents to vector store
                self.vector_store.add_documents(all_chunks)
                
                # Verify documents were added
                try:
                    final_count = self.vector_store._collection.count()
                    added_count = final_count - initial_count
                    logger.info(f"Successfully added {added_count} documents to vector store. Total: {final_count}")
                except:
                    logger.warning("Could not verify document count after adding")
                
                # Force persistence if using local storage
                if hasattr(self.vector_store, 'persist'):
                    try:
                        self.vector_store.persist()
                        logger.info("Vector store persisted successfully")
                    except Exception as persist_error:
                        logger.warning(f"Could not persist vector store: {persist_error}")
                
            except Exception as e:
                error_msg = f"Error adding documents to vector store: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results["status"] = "error"
                results["errors"].append({"general": error_msg})
        
        return results

    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask the RAG system
            
        Returns:
            dict: A dictionary containing the answer and source documents
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Check if vector store has any documents
            try:
                doc_count = self.vector_store._collection.count()
                if doc_count == 0:
                    return {
                        "result": "No documents have been uploaded yet. Please upload some financial documents first to ask questions about them.",
                        "source_documents": []
                    }
                logger.info(f"Vector store has {doc_count} documents available for querying")
            except Exception as e:
                logger.warning(f"Could not check document count: {e}")
            
            # Use the QA chain to get the answer
            result = self.qa_chain({"query": question})
            
            # Format the response
            response = {
                "result": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
            logger.info(f"Query processed successfully. Answer length: {len(response['result'])}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {"error": f"Failed to process query: {str(e)}"}

    def get_statistics(self):
        try:
            doc_count = self.vector_store._collection.count()
        except:
            doc_count = "Unknown"
        
        return {
            "total_documents": doc_count,
            "llm_provider": self.config.LLM_PROVIDER,
            "chunk_size": self.config.CHUNK_SIZE,
            "vector_store_dir": self.config.VECTOR_STORE_DIR
        }

# Initialize configuration
config = Config()

# Ensure vector store directory exists
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

# Ensure the vector store and cache directories exist
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# Initialize RAG system
try:
    print("🔧 Initializing Enhanced RAG system...")
    rag_system = EnhancedRAGSystem(config)
    print("✅ Enhanced RAG system initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize RAG system: {e}")
    logger.error(f"Failed to initialize RAG system: {e}")
    sys.exit(1)

# Initialize Flask server with correct paths
frontend_path = os.path.abspath('frontend_build')
app = Flask(__name__, 
            static_folder=os.path.join(frontend_path, 'static'), 
            template_folder=frontend_path)

app.config['CORS_SUPPORTS_CREDENTIALS'] = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['TIMEOUT'] = 300  # 5 minutes timeout for requests

# Increase the timeout for gunicorn workers
timeout = 300  # 5 minutes
worker_class = 'gthread'
workers = 2
threads = 4

# Allow both local development and production URLs
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://localhost:3000",
    "https://financeragsystem.onrender.com"
]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "expose_headers": ["Content-Type"]
        }
    },
    supports_credentials=True
)

# Track uploaded documents
uploaded_documents = {}

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the backend is running."""
    try:
        stats = rag_system.get_statistics()
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# API endpoint to query documents
@app.route('/api/query', methods=['POST', 'OPTIONS'])
def query_documents():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    start_time = time.time()
    
    # Log request
    logger.info("\n" + "=" * 80)
    logger.info(f"Incoming query request at {time.ctime()}")
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            error_msg = "No question provided in request body"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        question = data.get('question', '').strip()
        if not question:
            error_msg = "Question cannot be empty"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        logger.info(f"Processing question: {question}")
        
        # Process the query
        result = rag_system.query(question)
        
        # Add processing time
        result['processing_time_seconds'] = round(time.time() - start_time, 2)
        
        logger.info(f"Query processed successfully in {result['processing_time_seconds']}s")
        logger.info("-" * 80 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

# API endpoint to delete a document
@app.route('/api/documents/<document_id>', methods=['DELETE', 'OPTIONS'])
def delete_document(document_id):
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        if document_id not in uploaded_documents:
            return jsonify({"error": f"Document with ID {document_id} not found"}), 404
            
        # Remove document from tracking
        del uploaded_documents[document_id]
        
        # Note: In a real implementation, you would also remove the document from the vector store
        # This would require implementing a method in your RAG system to remove documents by ID
        
        return jsonify({
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500

# API endpoint to upload a document
@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_document():
    start_time = time.time()
    
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    logger.info("Upload document request received")
    
    if 'file' not in request.files:
        error_msg = "No file part in the request"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        error_msg = "No file selected"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    
    # Validate file type
    allowed_extensions = {'pdf', 'docx', 'txt', 'csv', 'xlsx'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        error_msg = f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    
    # Ensure file has a valid name
    if not file.filename or not file.filename.strip():
        error_msg = "Invalid file name"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    
    # Create a temp directory in the system's temp location
    temp_dir = os.path.join(tempfile.gettempdir(), 'finance_rag_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, secure_filename(file.filename))
    
    try:
        # Save the file temporarily
        logger.info(f"Saving uploaded file to temporary location: {file_path}")
        file.save(file_path)
        
        # Verify file was saved and has content
        if not os.path.exists(file_path):
            error_msg = f"Failed to save file to {file_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            error_msg = "Uploaded file is empty"
            logger.error(error_msg)
            os.remove(file_path)
            return jsonify({"error": error_msg}), 400
            
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        # No longer clearing the vector store to prevent data loss
        logger.info("Adding new document to existing vector store")
        
        # Process the file
        logger.info(f"Processing file: {file.filename}")
        result = rag_system.add_documents_batch([file_path], [file.filename])
        
        if result.get('status') == 'error' or (result.get('status') == 'partial_success' and not result.get('processed_files')):
            error_msg = "Failed to process document: "
            if result.get('errors'):
                error_msg += "; ".join([e.get('error', 'Unknown error') for e in result['errors']])
            else:
                error_msg += "Unknown error"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"File processed successfully in {processing_time} seconds")
        
        response = {
            "message": "File processed successfully",
            "document_id": str(hash(file.filename)),
            "filename": file.filename,
            "file_size": file_size,
            "chunks_processed": result.get('total_documents', 0),
            "upload_time": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "status": result.get('status', 'success')
        }
        
        if result.get('errors'):
            response["warnings"] = [e.get('error') for e in result['errors'] if e.get('error')]
        
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up temporary file
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {str(e)}")

# API endpoint to list documents
@app.route('/api/documents', methods=['GET', 'OPTIONS'])
def list_documents():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        # Convert the dictionary of documents to a list for the response
        documents_list = [
            {"id": doc_id, "filename": doc_info["filename"], "uploaded_at": doc_info["uploaded_at"]}
            for doc_id, doc_info in uploaded_documents.items()
        ]
        
        return jsonify({
            "documents": documents_list,
            "total_documents": len(documents_list),
            "stats": rag_system.get_statistics()
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to list documents: {str(e)}"}), 500

# Helper function for OPTIONS preflight requests
def _build_cors_preflight_response():
    response = jsonify({"message": "Preflight request successful"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
    response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
    return response

# Run the application
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Enhanced Finance RAG System Starting...")
    print("="*60)
    print(f"📊 LLM Provider: {config.LLM_PROVIDER}")
    print(f"🔧 Chunk Size: {config.CHUNK_SIZE}")
    print(f"📁 Vector Store: {config.VECTOR_STORE_DIR}")
    
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"🔗 URL: http://localhost:{port}")
        print(f"🏥 Health Check: http://localhost:{port}/api/health")
        print("="*60)
        
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',
            port=port,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)
    
    print("\n👋 Application stopped")