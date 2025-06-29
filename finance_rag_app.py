#!/usr/bin/env python3
"""
Enhanced Finance RAG System - A scalable financial document analysis system
using LangChain with proper LLM API integration and optimized for large documents.
"""

# Load environment variables first
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the absolute path to the directory containing this script
script_dir = Path(__file__).parent.resolve()

# Load .env file from the same directory as the script
env_path = script_dir / '.env'
print(f"Loading environment from: {env_path}")
load_dotenv(env_path, override=True)

# Verify the API key is loaded
print(f"GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
print(f"GOOGLE_API_KEY loaded: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")

import os
import uuid
import json
import time
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Flask imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# File processing with better error handling
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    WebBaseLoader,
)

# Additional imports for better document handling
import PyPDF2
import pandas as pd
from pathlib import Path
import hashlib
import pickle
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Configuration
class Config:
    # Document processing - optimized for speed
    CHUNK_SIZE = 2000  # Increased chunk size to reduce total chunks
    CHUNK_OVERLAP = 50  # Reduced overlap for faster processing
    MAX_BATCH_SIZE = 100  # Smaller batch size for more responsive processing
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required")
        
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Google Configuration (for embeddings)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required for embeddings")
    
    GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
    
    # LLM Provider (for compatibility with existing code)
    LLM_PROVIDER = "Groq"
    
    # Vector store configuration
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./chroma_db")
    VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION", "financial_documents")
    
    # Performance settings - optimized for speed
    MAX_WORKERS = min(16, (os.cpu_count() or 4) * 2)  # Use 2x CPU cores, max 16
    TEMPERATURE = 0.1
    MAX_TOKENS = 1000
    TOP_K_RETRIEVAL = 6
    
    # Cache settings - enable aggressive caching
    ENABLE_CACHING = True
    CACHE_DIR = "./document_cache"
    
    # Performance tuning
    EMBEDDING_BATCH_SIZE = 64  # Batch size for embedding operations
    MAX_RETRIES = 3  # Number of retries for failed operations
    RETRY_DELAY = 1.0  # Initial retry delay in seconds

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
            return None
            
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
                return None
                
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
                    doc.metadata.update({
                        'source': file_name,
                        'file_hash': file_hash,
                        'processed_at': datetime.now().isoformat()
                    })
                    processed_docs.append(doc)
                except Exception as meta_error:
                    logger.warning(f"Error adding metadata to document: {str(meta_error)}")
            
            if not processed_docs:
                logger.warning(f"No valid documents found in {file_name}")
                return None
            
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
            return None
    
    def _load_pdf_with_fallback(self, file_path: str, file_name: str) -> List[Document]:
        """Load PDF with optimized settings and fast fallback methods."""
        try:
            # First try PyMuPDF which is generally faster than PyPDF
            loader = UnstructuredFileLoader(
                file_path,
                mode="elements",  # Process in elements mode for better performance
                strategy="fast"     # Use fast strategy
            )
            docs = loader.load()
            
            # Filter out small text elements that are likely not useful
            return [doc for doc in docs if len(doc.page_content.strip()) > 50]
            
        except Exception as e:
            logger.warning(f"Fast PDF loading failed: {str(e)}. Trying fallback...")
            try:
                # Fallback to PyPDF with optimized settings
                loader = PyPDFLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.error(f"PDF loading failed: {str(e)}")
                return []
    
    def _load_csv_enhanced(self, file_path: str, file_name: str) -> List[Document]:
        """Enhanced CSV loading with better handling."""
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            # Create a summary document
            summary = f"CSV File: {file_name}\n"
            summary += f"Columns: {', '.join(df.columns)}\n"
            summary += f"Rows: {len(df)}\n"
            summary += f"Data types:\n{df.dtypes.to_string()}\n"
            
            if len(df) > 0:
                summary += f"\nFirst few rows:\n{df.head().to_string()}\n"
                summary += f"\nSummary statistics:\n{df.describe().to_string()}"
            
            documents.append(Document(
                page_content=summary,
                metadata={'source': file_name, 'type': 'csv_summary'}
            ))
            
            # For smaller datasets, include row-by-row data
            if len(df) <= 1000:
                for idx, row in df.iterrows():
                    row_text = f"Row {idx + 1}:\n" + "\n".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(Document(
                        page_content=row_text,
                        metadata={'source': file_name, 'row': idx + 1, 'type': 'csv_row'}
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_name}: {e}")
            raise
    
    def _load_excel_enhanced(self, file_path: str, file_name: str) -> List[Document]:
        """Enhanced Excel loading."""
        try:
            # Load all sheets
            excel_file = pd.ExcelFile(file_path)
            documents = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                summary = f"Excel File: {file_name}, Sheet: {sheet_name}\n"
                summary += f"Columns: {', '.join(df.columns)}\n"
                summary += f"Rows: {len(df)}\n"
                
                if len(df) > 0:
                    summary += f"\nFirst few rows:\n{df.head().to_string()}\n"
                    summary += f"\nSummary statistics:\n{df.describe().to_string()}"
                
                documents.append(Document(
                    page_content=summary,
                    metadata={'source': file_name, 'sheet': sheet_name, 'type': 'excel_summary'}
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Excel {file_name}: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with progress tracking."""
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
    
    def _list_available_models(self):
        """List all available models from the API."""
        import google.generativeai as genai
        from google.api_core import client_options as client_options_lib
        
        try:
            client = genai.Client(
                api_key=self.config.GOOGLE_API_KEY,
                client_options=client_options_lib.ClientOptions(
                    api_endpoint=self.config.GOOGLE_API_BASE
                )
            )
            
            models = client.list_models()
            available_models = [model.name for model in models]
            logger.info(f"Available models: {available_models}")
            return available_models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def _initialize_llm_and_embeddings(self):
        """Initialize Groq LLM and Google embeddings with proper configuration and error handling."""
        logger.info("Initializing Groq model and Google embeddings...")
        
        if not self.config.GROQ_API_KEY:
            error_msg = "GROQ_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Import required modules
            from langchain_groq import ChatGroq
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from groq import Groq
            import google.generativeai as genai
            
            # Initialize embeddings with error handling and timeout
            logger.info("Initializing embeddings...")
            try:
                # Use Google's embeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    request_timeout=30,  # seconds
                    max_retries=3
                )
                
                # Test the embeddings
                test_embedding = self.embeddings.embed_query("test")
                logger.info(f"Successfully initialized embeddings. Vector dimension: {len(test_embedding)}")
                
            except Exception as e:
                error_msg = f"Failed to initialize embeddings: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError("Failed to initialize embeddings. Please check your Google API key and network connection.")
            
            # Initialize LLM with error handling
            logger.info("Initializing LLM...")
            try:
                # Get API key directly from environment variables with debug logging
                groq_api_key = os.getenv("GROQ_API_KEY")
                logger.info(f"API Key length: {len(groq_api_key) if groq_api_key else 0}")
                logger.info(f"API Key starts with: {groq_api_key[:8] if groq_api_key else 'N/A'}")
                
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                
                # Initialize Groq client directly to test the key
                from groq import Groq
                test_client = Groq(api_key=groq_api_key)
                
                # Test the client with a simple request
                test_response = test_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Say this is a test"}],
                    model=self.config.GROQ_MODEL,
                )
                logger.info(f"Successfully tested Groq API. Response: {test_response.choices[0].message.content[:100]}...")
                
                # Now initialize the ChatGroq instance
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=self.config.GROQ_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    streaming=False  # Disable streaming for simpler error handling
                )
                
                # Test the LLM with a simple query
                test_response = self.llm.invoke("Hello, are you working?")
                logger.info(f"Successfully initialized LLM. Test response: {test_response.content[:100]}...")
                
            except Exception as e:
                error_msg = f"Failed to initialize LLM: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError("Failed to initialize the language model. Please check your Groq API key and model name.")
            
            logger.info("Successfully initialized Groq model and Google embeddings")
            
        except ImportError as e:
            error_msg = f"Required packages not found: {str(e)}. Please install with: pip install groq langchain-groq langchain-google-genai google-generativeai"
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to initialize models: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _initialize_vector_store(self):
        """Initialize vector store with proper configuration and error handling."""
        try:
            # Ensure the vector store directory exists and is writable
            os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True, mode=0o755)
            
            # Test directory permissions
            test_file = os.path.join(self.config.VECTOR_STORE_DIR, '.permission_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                error_msg = f"Cannot write to vector store directory {self.config.VECTOR_STORE_DIR}: {str(e)}"
                logger.error(error_msg)
                raise PermissionError(error_msg)
            
            logger.info(f"Initializing Chroma vector store in {self.config.VECTOR_STORE_DIR}")
            logger.info(f"Collection name: {self.config.VECTOR_STORE_COLLECTION}")
                
            logger.info(f"Initializing Chroma vector store in: {os.path.abspath(self.config.VECTOR_STORE_DIR)}")
            
            # Initialize Chroma with error handling
            try:
                # For ChromaDB 1.0.0+ with the new client API
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                
                # Initialize the Chroma client
                client = chromadb.PersistentClient(
                    path=self.config.VECTOR_STORE_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                
                # Get or create the collection
                try:
                    collection = client.get_collection(name=self.config.VECTOR_STORE_COLLECTION)
                    logger.info(f"Using existing collection: {self.config.VECTOR_STORE_COLLECTION}")
                except Exception:
                    # Collection doesn't exist, create it
                    collection = client.create_collection(
                        name=self.config.VECTOR_STORE_COLLECTION,
                        metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
                    )
                    logger.info(f"Created new collection: {self.config.VECTOR_STORE_COLLECTION}")
                
                # Initialize the LangChain Chroma wrapper
                self.vector_store = Chroma(
                    client=client,
                    collection_name=self.config.VECTOR_STORE_COLLECTION,
                    embedding_function=self.embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                
                logger.info("Chroma vector store initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize Chroma: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)
            
            # Verify the collection was created/loaded
            if not hasattr(self.vector_store, '_collection'):
                error_msg = "Failed to initialize Chroma collection"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Check document count
            try:
                doc_count = self.vector_store._collection.count()
                logger.info(f"Vector store initialized with {doc_count} documents in collection '{self.config.VECTOR_STORE_COLLECTION}'")
                
                # Log collection metadata for debugging
                try:
                    collection_metadata = self.vector_store._collection.get()
                    if collection_metadata and 'metadatas' in collection_metadata:
                        logger.debug(f"Collection metadata keys: {collection_metadata.keys()}")
                        logger.debug(f"Document IDs: {collection_metadata.get('ids', [])}")
                        logger.debug(f"Document count: {len(collection_metadata.get('ids', []))}")
                except Exception as meta_error:
                    logger.warning(f"Could not retrieve collection metadata: {str(meta_error)}")
                    
            except Exception as count_error:
                logger.warning(f"Could not get document count: {str(count_error)}")
                logger.info("Vector store initialized (document count unavailable)")
            
            # Skip test document addition in production
            logger.debug("Skipping test document addition in production")
                
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

def _get_cached_embeddings(self, text: str) -> Optional[List[float]]:
    """Get cached embeddings if available."""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    with self.cache_lock:
        return self.embedding_cache.get(text_hash)

def _cache_embeddings(self, text: str, embedding: List[float]) -> None:
    """Cache embeddings for future use."""
    if not text.strip():
        return
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    with self.cache_lock:
        self.embedding_cache[text_hash] = embedding

def _process_batch_with_cache(self, batch: List[Document]) -> List[Document]:
    """Process a batch of documents with embedding caching."""
    # Check cache first
    texts_to_embed = []
    cached_embeddings = {}

    for doc in batch:
        text = doc.page_content
        cached = self._get_cached_embeddings(text)
        if cached is not None:
            cached_embeddings[text] = cached
        else:
            texts_to_embed.append(text)

    # Get embeddings for non-cached texts
    if texts_to_embed:
        try:
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            for text, embedding in zip(texts_to_embed, new_embeddings):
                self._cache_embeddings(text, embedding)
                cached_embeddings[text] = embedding
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return []

    # Update documents with embeddings
    for doc in batch:

def add_documents_batch(self, file_paths: List[str], file_names: List[str] = None) -> Dict[str, Any]:
    """Process documents with optimized performance and error handling."""
    if not file_names:
        file_names = [os.path.basename(p) for p in file_paths]
        
    if len(file_paths) != len(file_names):
        raise ValueError("file_paths and file_names must have the same length")
    
    results = {
        'processed': 0,
        'failed': 0,
        'total_chunks': 0,
        'errors': []
    }
    
    logger.info(f"Processing {len(file_paths)} documents...")
    
    # Process documents in parallel with a fixed number of workers
    with ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as executor:
        # Process each document
        future_to_name = {
            executor.submit(self._process_single_document, path, name): name
            for path, name in zip(file_paths, file_names)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(chunks))
                batch = chunks[start_idx:end_idx]
                
                # Submit batch processing task
                future = executor.submit(
                    self._process_single_batch,
                    batch,
                    batch_num + 1,
                    total_batches
                )
                futures.append(future)
            
            # Process completed futures
            for future in as_completed(futures):
                try:
                    batch_added = future.result()
                    added_count += batch_added
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    batch_errors += 1
                    if batch_errors >= 3:  # If we fail 3 times in a row, give up
                        raise RuntimeError("Too many batch processing errors") from e
        
        success_ratio = (added_count / len(chunks)) * 100 if chunks else 0
        logger.info(
            f"Added {added_count}/{len(chunks)} documents to vector store "
            f"({success_ratio:.1f}% success rate)"
        )
        
        # Persistence is handled automatically by Chroma 0.4.x+
        return added_count
        
    def _process_single_batch(self, batch: List[Document], batch_num: int, total_batches: int) -> int:
        """
        Process a single batch of documents with retries.
        
        Args:
            batch: List of documents in the batch
            batch_num: Current batch number
            total_batches: Total number of batches
            
        Returns:
            int: Number of successfully processed documents in this batch
        """
        retry_count = 0
        max_retries = self.config.MAX_RETRIES
        
        while retry_count <= max_retries:
            try:
                # Filter out any invalid documents
                valid_docs = [doc for doc in batch if doc.page_content and doc.page_content.strip()]
                if not valid_docs:
                    logger.warning(f"No valid documents in batch {batch_num}")
                    return 0
                    
                # Add documents to the vector store
                self.vector_store.add_documents(valid_docs)
                logger.info(f"Successfully added batch {batch_num}/{total_batches} with {len(valid_docs)} documents")
                return len(valid_docs)
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(
                        f"Failed to add batch {batch_num} after {max_retries} attempts: {str(e)}",
                        exc_info=True
                    )
                    return 0
                    
                wait_time = self.config.RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(
                    f"Error adding batch {batch_num} (attempt {retry_count}/{max_retries}): {str(e)}"
                    f" - Retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)
        
        return 0
            
    def query(self, question, max_retries=3):
        if not question or not question.strip():
            error_msg = "Empty question provided"
            logger.warning(error_msg)
            return {
                "result": "Please provide a valid question.",
                "source_documents": [],
                "error": error_msg
            }
        
        logger.info(f"Processing query: '{question}'")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} - Sending to QA chain...")
                
                # Check if QA chain is properly initialized
                if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                    error_msg = "QA chain not properly initialized"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Execute the query
                result = self.qa_chain({"query": question})
                logger.info(f"Received response from QA chain")
                
                if not result:
                    error_msg = "Empty response from QA chain"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Process source documents
                source_docs = []
                source_documents = result.get("source_documents", [])
                logger.info(f"Found {len(source_documents)} source documents")
                
                for i, doc in enumerate(source_documents, 1):
                    try:
                        content = doc.page_content
                        metadata = getattr(doc, 'metadata', {})
                        source_docs.append({
                            "content": content[:500] + ("..." if len(content) > 500 else ""),
                            "metadata": metadata,
                            "source": metadata.get('source', 'Unknown'),
                            "page": metadata.get('page', 'N/A')
                        })
                        logger.debug(f"Source doc {i}: {metadata.get('source', 'Unknown')} (page {metadata.get('page', 'N/A')})")
                    except Exception as doc_error:
                        logger.warning(f"Error processing source document {i}: {str(doc_error)}")
                
                response = {
                    "result": result.get("result", "No answer generated."),
                    "source_documents": source_docs,
                    "query": question
                }
                
                logger.info("Query processed successfully")
                return response
                
            except Exception as e:
                error_msg = f"Query attempt {attempt + 1} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for query: {question}")
                    return {
                        "result": "I encountered an error while processing your query. Please try rephrasing your question or try again later.",
                        "source_documents": [],
                        "error": str(e)
                    }
                
                # Exponential backoff before retry
                wait_time = (2 ** attempt) * 0.5
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return {
            "result": "Maximum retry attempts reached. Please try again later.",
            "source_documents": [],
            "error": "Max retries exceeded"
        }
        
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

# Ensure vector store directory exists and is writable
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
try:
    os.chmod(config.VECTOR_STORE_DIR, 0o777)  # Make writable
except Exception as e:
    logger.warning(f"Warning: Could not set permissions for {config.VECTOR_STORE_DIR}: {str(e)}")

# Validate configuration
if not config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set or empty")

# Ensure the vector store and cache directories exist
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# Initialize RAG system
try:
    rag_system = EnhancedRAGSystem(config)
    logger.info(f"Using Groq model: {config.GROQ_MODEL}")
    logger.info("Enhanced RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    raise

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
    os.chmod(temp_dir, 0o777)  # Ensure write permissions
    
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
        
        if result.get('failed', 0) > 0 or result.get('processed', 0) == 0:
            error_msg = f"Failed to process document: {', '.join(result.get('errors', ['Unknown error']))}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"File processed successfully in {processing_time} seconds")
        
        response = {
            "message": "File processed successfully",
            "document_id": str(hash(file.filename)),
            "filename": file.filename,
            "file_size": file_size,
            "chunks_processed": result.get('total_chunks', 0),
            "upload_time": datetime.now().isoformat(),
            "processing_time_seconds": processing_time
        }
        
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
    
    return jsonify({"message": "Preflight request successful"}), 200

# Run the application
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Enhanced Finance RAG System Starting...")
    print("="*60)
    print(f"📊 LLM Provider: {config.LLM_PROVIDER}")
    print(f"🔧 Chunk Size: {config.CHUNK_SIZE}")
    print(f"📁 Vector Store: {config.VECTOR_STORE_DIR}")
    print(f"🔗 URL: http://localhost:8080")
    print("="*60)
    
    try:
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',
            port=8080,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Application failed to start: {e}")
    
    print("\n👋 Application stopped")