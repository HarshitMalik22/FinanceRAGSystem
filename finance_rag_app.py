#!/usr/bin/env python3
"""
Enhanced Finance RAG System - A scalable financial document analysis system
using LangChain with proper LLM API integration and optimized for large documents.
"""

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
    # Document processing
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    MAX_BATCH_SIZE = 50
    
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
    
    # Performance settings
    MAX_WORKERS = 4
    TEMPERATURE = 0.1
    MAX_TOKENS = 1000
    TOP_K_RETRIEVAL = 6
    
    # Cache settings
    ENABLE_CACHING = True
    CACHE_DIR = "./document_cache"

class DocumentProcessor:
    """Enhanced document processor for handling large volumes of documents."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Create cache directory
        os.makedirs(config.CACHE_DIR, exist_ok=True)
    
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
        """Load and process a single document with caching."""
        file_name = file_name or os.path.basename(file_path)
        logger.info(f"Starting to load document: {file_name} from {file_path}")
        
        try:
            file_hash = self._get_file_hash(file_path)
            logger.debug(f"Generated file hash: {file_hash}")
            
            # Check cache first if enabled
            if self.config.ENABLE_CACHING:
                logger.debug("Checking for cached documents...")
                cached_docs = self._get_cached_documents(file_hash)
                if cached_docs:
                    logger.info(f"Loaded {len(cached_docs)} cached documents for {file_name}")
                    return cached_docs
            
            # Ensure file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"No read permissions for file: {file_path}")
            
            logger.info(f"Loading document: {file_path}")
            documents = []
            
            # Load document based on file type
            try:
                if file_path.lower().endswith('.pdf'):
                    logger.debug("Processing as PDF file")
                    documents = self._load_pdf_with_fallback(file_path, file_name)
                elif file_path.lower().endswith('.docx'):
                    logger.debug("Processing as DOCX file")
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                elif file_path.lower().endswith('.txt'):
                    logger.debug("Processing as TXT file")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                elif file_path.lower().endswith('.csv'):
                    logger.debug("Processing as CSV file")
                    documents = self._load_csv_enhanced(file_path, file_name)
                elif file_path.lower().endswith(('.xls', '.xlsx')):
                    logger.debug("Processing as Excel file")
                    documents = self._load_excel_enhanced(file_path, file_name)
                else:
                    error_msg = f"Unsupported file type: {file_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                if not documents:
                    warning_msg = f"No content could be extracted from {file_name}"
                    logger.warning(warning_msg)
                    return []
                    
                logger.info(f"Successfully loaded {len(documents)} document(s) from {file_name}")
                
                # Cache the loaded documents
                if self.config.ENABLE_CACHING:
                    logger.debug("Caching loaded documents")
                    try:
                        self._cache_documents(file_hash, documents)
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache documents: {str(cache_error)}")
                
                # Add metadata to each document
                processed_docs = []
                for doc in documents:
                    try:
                        doc.metadata.update({
                            'source': file_name,
                            'file_hash': file_hash,
                            'processed_at': datetime.now().isoformat()
                        })
                        processed_docs.append(doc)
                    except Exception as meta_error:
                        logger.warning(f"Error adding metadata to document: {str(meta_error)}")
                        processed_docs.append(doc)  # Still add the document even if metadata fails
                
                if not processed_docs:
                    logger.warning(f"No valid documents processed from {file_name}")
                    return []
                    
                return processed_docs
                
            except Exception as e:
                error_msg = f"Error processing document {file_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(f"Failed to process document: {error_msg}") from e
            
        except Exception as e:
            logger.error(f"Error loading {file_name}: {str(e)}")
            raise
    
    def _load_pdf_with_fallback(self, file_path: str, file_name: str) -> List[Document]:
        """Load PDF with multiple fallback methods."""
        try:
            # Try LangChain PyPDFLoader first
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.warning(f"PyPDFLoader failed for {file_name}: {e}")
            
            try:
                # Fallback to PyPDF2
                documents = []
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={'source': file_name, 'page': page_num + 1}
                            )
                            documents.append(doc)
                return documents
            except Exception as e2:
                logger.error(f"All PDF loading methods failed for {file_name}: {e2}")
                raise
    
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
        
        # Initialize LLM and embeddings based on provider
        self._initialize_llm_and_embeddings()
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Initialize QA chain
        self._initialize_qa_chain()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    
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
                self.llm = ChatGroq(
                    groq_api_key=self.config.GROQ_API_KEY,
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
            os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True)
            
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
            
            # Ensure vector store directory exists and is writable
            os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True, mode=0o755)
            if not os.access(self.config.VECTOR_STORE_DIR, os.W_OK):
                error_msg = f"Vector store directory is not writable: {self.config.VECTOR_STORE_DIR}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            logger.info(f"Initializing Chroma vector store in: {os.path.abspath(self.config.VECTOR_STORE_DIR)}")
            
            # Initialize Chroma with error handling
            try:
                self.vector_store = Chroma(
                    collection_name=self.config.VECTOR_STORE_COLLECTION,
                    embedding_function=self.embeddings,
                    persist_directory=self.config.VECTOR_STORE_DIR,
                    collection_metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
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
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("QA chain initialized successfully")
    
    def add_documents_batch(self, file_paths: List[str], file_names: List[str] = None) -> Dict[str, Any]:
        """Add multiple documents in batches with progress tracking."""
        if not file_names:
            file_names = [os.path.basename(path) for path in file_paths]
        
        results = {
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        logger.info(f"Processing {len(file_paths)} documents...")
        
        # Process documents in parallel
        future_to_file = {}
        for file_path, file_name in zip(file_paths, file_names):
            future = self.executor.submit(self._process_single_document, file_path, file_name)
            future_to_file[future] = (file_path, file_name)
        
        all_chunks = []
        for future in as_completed(future_to_file):
            file_path, file_name = future_to_file[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                    results['processed'] += 1
                    results['total_chunks'] += len(chunks)
                    logger.info(f"Processed {file_name}: {len(chunks)} chunks")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"No content extracted from {file_name}")
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Error processing {file_name}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Add chunks to vector store in batches and verify persistence
        if all_chunks:
            try:
                # Store the initial document count
                initial_count = self.vector_store._collection.count()
                logger.info(f"Initial document count in vector store: {initial_count}")
                
                # Add documents to the vector store
                added_count = self._add_chunks_to_vector_store(all_chunks)
                
                # Verify documents were added
                final_count = self.vector_store._collection.count()
                expected_count = initial_count + added_count
                
                if final_count < expected_count:
                    logger.warning(f"Document count mismatch. Expected {expected_count}, got {final_count}")
                else:
                    logger.info(f"Successfully added {added_count} documents. New total: {final_count}")
                    
                # Force persistence and verify
                self.vector_store.persist()
                if not os.path.exists(os.path.join(self.config.VECTOR_STORE_DIR, 'chroma.sqlite3')):
                    logger.error("Failed to persist vector store: chroma.sqlite3 not found")
                else:
                    logger.info("Vector store persistence verified")
                    
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
                raise
        
        logger.info(f"Batch processing complete: {results['processed']} successful, {results['failed']} failed")
        return results
    
    def _process_single_document(self, file_path: str, file_name: str) -> List[Document]:
        """Process a single document and return chunks."""
        try:
            logger.info(f"Loading document: {file_name}")
            documents = self.document_processor.load_document(file_path, file_name)
            if not documents:
                logger.error(f"No content extracted from {file_name}")
                return []
                
            logger.info(f"Splitting document: {file_name} into chunks")
            chunks = self.document_processor.split_documents(documents)
            
            if not chunks:
                logger.error(f"No chunks generated from {file_name}")
                return []
                
            logger.info(f"Generated {len(chunks)} chunks from {file_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}", exc_info=True)
            raise
    
    def _add_chunks_to_vector_store(self, chunks: List[Document]) -> int:
        """
        Add document chunks to vector store in batches.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            int: Number of successfully added documents
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return 0
            
        batch_size = self.config.MAX_BATCH_SIZE
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        logger.info(f"Adding {len(chunks)} chunks to vector store in {total_batches} batches...")
        
        # Ensure vector store directory exists and is writable
        os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True, mode=0o755)
        if not os.access(self.config.VECTOR_STORE_DIR, os.W_OK):
            error_msg = f"Vector store directory is not writable: {self.config.VECTOR_STORE_DIR}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        added_count = 0
        batch_errors = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Verify documents in batch
                valid_docs = []
                for doc in batch:
                    if not doc.page_content or not doc.page_content.strip():
                        logger.warning(f"Empty document content in batch {batch_num}")
                        continue
                    valid_docs.append(doc)
                
                if not valid_docs:
                    logger.warning(f"No valid documents in batch {batch_num}")
                    continue
                
                # Add documents
                try:
                    self.vector_store.add_documents(valid_docs)
                    added_count += len(valid_docs)
                    logger.info(f"Successfully added batch {batch_num}/{total_batches} with {len(valid_docs)} documents")
                    
                    # Persist after each successful batch
                    self.vector_store.persist()
                    logger.debug(f"Persisted vector store after batch {batch_num}")
                    
                except Exception as add_error:
                    logger.error(f"Failed to add batch {batch_num}: {str(add_error)}")
                    batch_errors += 1
                    if batch_errors >= 3:  # If we fail 3 times in a row, give up
                        raise
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error processing batch {batch_num}: {str(e)}", exc_info=True)
                batch_errors += 1
                if batch_errors >= 3:  # If we fail 3 times in a row, give up
                    raise
                continue
        
        success_ratio = (added_count / len(chunks)) * 100 if chunks else 0
        logger.info(
            f"Added {added_count}/{len(chunks)} documents to vector store "
            f"({success_ratio:.1f}% success rate)"
        )
        
        if added_count > 0:
            try:
                self.vector_store.persist()
                logger.info("Final vector store persistence completed")
            except Exception as e:
                logger.error(f"Failed final persistence: {str(e)}")
        
        return added_count
            
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

# Initialize configuration and RAG system
config = Config()

# Validate configuration
if not config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

# Ensure the vector store directory exists
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

# Initialize Flask server
app = Flask(__name__)

# Configure CORS for all routes under /api/
app.config['CORS_SUPPORTS_CREDENTIALS'] = True
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "http://localhost:3000",
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
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
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
        
        # Clear existing vector store before adding new documents
        logger.info("Clearing existing vector store")
        try:
            if rag_system.vector_store and hasattr(rag_system.vector_store, 'delete'):
                # Get all document IDs and delete them
                all_docs = rag_system.vector_store.get()
                if all_docs and 'ids' in all_docs and all_docs['ids']:
                    rag_system.vector_store.delete(ids=all_docs['ids'])
                    logger.info(f"Deleted {len(all_docs['ids'])} existing documents from vector store")
        except Exception as e:
            logger.warning(f"Warning: Could not clear vector store: {str(e)}")
        
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
            if os.path.exists(file_path):
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