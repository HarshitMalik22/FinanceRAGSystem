#!/usr/bin/env python3
"""
Enhanced Finance RAG System - A scalable financial document analysis system
using LangChain with proper LLM API integration and optimized for large documents.
"""

import os
import dash
import uuid
import base64
import tempfile
import json
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
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
    VECTOR_STORE_DIR = "./enhanced_chroma_db"
    VECTOR_STORE_COLLECTION = "financial_documents"
    
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
        file_hash = self._get_file_hash(file_path)
        
        # Check cache first
        cached_docs = self._get_cached_documents(file_hash)
        if cached_docs:
            logger.info(f"Loaded {len(cached_docs)} cached documents for {file_name}")
            return cached_docs
        
        try:
            # Load document based on file type
            if file_path.lower().endswith('.pdf'):
                documents = self._load_pdf_with_fallback(file_path, file_name)
            elif file_path.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif file_path.lower().endswith('.csv'):
                documents = self._load_csv_enhanced(file_path, file_name)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                documents = self._load_excel_enhanced(file_path, file_name)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_name,
                    'file_hash': file_hash,
                    'processed_at': datetime.now().isoformat()
                })
            
            # Cache the processed documents
            self._cache_documents(file_hash, documents)
            
            logger.info(f"Loaded {len(documents)} documents from {file_name}")
            return documents
            
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
        """Initialize vector store with proper configuration."""
        os.makedirs(self.config.VECTOR_STORE_DIR, exist_ok=True)
        
        try:
            self.vector_store = Chroma(
                collection_name=self.config.VECTOR_STORE_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=self.config.VECTOR_STORE_DIR
            )
            
            # Check document count
            try:
                doc_count = self.vector_store._collection.count()
                logger.info(f"Vector store initialized with {doc_count} documents")
            except:
                logger.info("Vector store initialized (document count unavailable)")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _initialize_qa_chain(self):
        """Initialize QA chain with enhanced prompt."""
        template = """You are an expert financial analyst. Use the following context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer strictly on the provided context
2. If the information is not in the context, clearly state "I don't have enough information to answer this question based on the provided documents"
3. For numerical data, be precise and include units/currency
4. Provide specific references to document sources when possible
5. Structure your response clearly with relevant headings if the answer is complex

Answer:"""
        
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
        
        # Add chunks to vector store in batches
        if all_chunks:
            self._add_chunks_to_vector_store(all_chunks)
        
        logger.info(f"Batch processing complete: {results['processed']} successful, {results['failed']} failed")
        return results
    
    def _process_single_document(self, file_path: str, file_name: str) -> List[Document]:
        """Process a single document and return chunks."""
        try:
            documents = self.document_processor.load_document(file_path, file_name)
            chunks = self.document_processor.split_documents(documents)
            return chunks
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            raise
    
    def _add_chunks_to_vector_store(self, chunks: List[Document]):
        """Add document chunks to vector store in batches."""
        batch_size = self.config.MAX_BATCH_SIZE
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        logger.info(f"Adding {len(chunks)} chunks to vector store in {total_batches} batches...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                self.vector_store.add_documents(batch)
                logger.info(f"Added batch {(i // batch_size) + 1}/{total_batches}")
            except Exception as e:
                logger.error(f"Error adding batch {(i // batch_size) + 1}: {e}")
                raise
        
        # Persist the vector store
        try:
            self.vector_store.persist()
            logger.info("Vector store persisted successfully")
        except Exception as e:
            logger.warning(f"Failed to persist vector store: {e}")
    
    def query(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query the RAG system with retry logic and enhanced error handling."""
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
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

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

app.title = "Enhanced Finance RAG System"

# Enhanced app layout
app.layout = dbc.Container([
    dcc.Store(id='document-store', data={'documents': [], 'stats': {}}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="bi bi-graph-up-arrow me-3"),
                "Enhanced Finance Document Analyzer"
            ], className="text-center mb-4 text-primary"),
            html.P(
                "Upload financial documents and ask questions using advanced AI analysis",
                className="text-center text-muted mb-4"
            )
        ])
    ]),
    
    # Stats row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("0", id="doc-count", className="text-primary mb-0"),
                    html.P("Documents", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Groq", className="text-success mb-0"),
                    html.P("LLM Provider", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Ready", id="system-status", className="text-info mb-0"),
                    html.P("System Status", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{config.CHUNK_SIZE}", className="text-warning mb-0"),
                    html.P("Chunk Size", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=3),
    ], className="mb-4"),
    
    # Main content
    dbc.Row([
        # Left sidebar - Document upload
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-cloud-upload me-2"),
                    "Document Upload"
                ]),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-documents',
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload fs-1 text-muted"),
                            html.Br(),
                            html.Span("Drag & Drop or Click to Upload"),
                            html.Br(),
                            html.Small("PDF, DOCX, TXT, CSV, Excel files", className="text-muted")
                        ]),
                        style={
                            'width': '100%',
                            'height': '120px',
                            'lineHeight': '120px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'cursor': 'pointer',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=True
                    ),
                    
                    html.Div(id='upload-status'),
                    html.Hr(),
                    html.Div(id='document-list')
                ])
            ])
        ], md=4),
        
        # Right side - Query interface
        dbc.Col([
            # Query input
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-chat-dots me-2"),
                    "Ask Questions"
                ]),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.Input(
                            id="question-input",
                            placeholder="Ask a question about your financial documents...",
                            type="text",
                            size="lg"
                        ),
                        dbc.Button([
                            html.I(className="bi bi-send me-2"),
                            "Ask"
                        ], id="ask-button", color="primary", size="lg")
                    ], className="mb-3"),
                    
                    html.Div(id='query-status'),
                ])
            ], className="mb-4"),
            
            # Results
            html.Div(id='query-results')
        ], md=8)
    ])
], fluid=True, className="p-4", style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# Callbacks
@app.callback(
    [Output('document-store', 'data'),
     Output('upload-status', 'children'),
     Output('doc-count', 'children'),
     Output('system-status', 'children')],
    [Input('upload-documents', 'contents'),
     Input('upload-documents', 'filename')],
    [State('document-store', 'data')]
)
def handle_document_upload(contents, filenames, stored_data):
    """Handle document uploads with enhanced feedback."""
    if not contents:
        return stored_data, "", len(stored_data.get('documents', [])), "Ready"
    
    try:
        # Show processing status
        processing_alert = dbc.Alert([
            html.Span(dbc.Spinner(size="sm"), className="me-2"),
            f"Processing {len(contents)} document(s)..."
        ], color="info", className="mb-3")
        
        # Save uploaded files temporarily
        temp_files = []
        file_names = []
        
        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create temp file with proper extension
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(decoded)
                temp_files.append(tmp.name)
                file_names.append(filename)
        
        # Process documents
        results = rag_system.add_documents_batch(temp_files, file_names)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Update stored data
        new_documents = stored_data.get('documents', [])
        for filename in file_names:
            if filename not in [doc['name'] for doc in new_documents]:
                new_documents.append({
                    'name': filename,
                    'uploaded_at': datetime.now().isoformat(),
                    'id': str(uuid.uuid4())
                })
        
        stored_data['documents'] = new_documents
        stored_data['stats'] = rag_system.get_statistics()
        
        # Create status message
        if results['processed'] > 0:
            status = dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                f"Successfully processed {results['processed']} document(s) into {results['total_chunks']} chunks"
            ], color="success", dismissable=True)
            system_status = "Active"
        else:
            status = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Failed to process documents. Please check file formats."
            ], color="danger", dismissable=True)
            system_status = "Error"
        
        return stored_data, status, len(new_documents), system_status
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        error_status = dbc.Alert([
            html.I(className="bi bi-exclamation-circle me-2"),
            f"Upload failed: {str(e)}"
        ], color="danger", dismissable=True)
        
        return stored_data, error_status, len(stored_data.get('documents', [])), "Error"

@app.callback(
    Output('document-list', 'children'),
    [Input('document-store', 'data')]
)
def update_document_list(stored_data):
    """Update the document list display."""
    documents = stored_data.get('documents', [])
    
    if not documents:
        return dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "No documents uploaded yet"
        ], color="light")
    
    items = []
    for doc in documents:
        items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.I(className="bi bi-file-earmark-text me-2 text-primary"),
                    html.Strong(doc['name']),
                ]),
                html.Small([
                    html.I(className="bi bi-calendar3 me-1"),
                    doc['uploaded_at'].split('T')[0]
                ], className="text-muted")
            ])
        )
    
    return dbc.ListGroup(items, flush=True)

@app.callback(
    [Output('query-results', 'children'),
     Output('query-status', 'children')],
    [Input('ask-button', 'n_clicks')],
    [State('question-input', 'value'),
     State('document-store', 'data')],
    prevent_initial_call=True
)
def process_query(n_clicks, question, stored_data):
    """Process user queries with enhanced UI feedback and error handling."""
    logger.info(f"Processing query: {question}")
    
    if not n_clicks or not question:
        logger.warning("No question provided or button not clicked")
        return "", ""
    
    documents = stored_data.get('documents', [])
    logger.info(f"Found {len(documents)} documents in store")
    
    if not documents:
        logger.warning("No documents found in the store")
        no_docs_alert = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            "Please upload documents before asking questions."
        ], color="warning")
        return "", no_docs_alert
    
    try:
        # Show loading status
        loading_status = dbc.Alert([
            html.Span(dbc.Spinner(size="sm"), className="me-2"),
            "Processing your question..."
        ], color="info")
        
        logger.info("Sending query to RAG system...")
        result = rag_system.query(question)
        logger.info(f"Received response from RAG system: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if not result or (isinstance(result, dict) and "error" in result):
            error_msg = result.get('error', 'No response from the model') if isinstance(result, dict) else 'Invalid response format'
            logger.error(f"Query failed: {error_msg}")
            error_alert = dbc.Alert([
                html.I(className="bi bi-exclamation-circle me-2"),
                f"Query failed: {error_msg}"
            ], color="danger")
            return "", error_alert
        
        # Create response layout
        response_content = []
        
        # Main answer
        answer_card = dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-lightbulb me-2"),
                "Answer"
            ]),
            dbc.CardBody([
                html.P(result['result'], className="mb-0", style={'whiteSpace': 'pre-wrap'})
            ])
        ], className="mb-4")
        
        response_content.append(answer_card)
        
        # Source documents
        if result.get('source_documents'):
            sources_content = []
            for i, doc in enumerate(result['source_documents'][:4]):  # Limit to 4 sources
                source_card = dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-file-text me-2"),
                        f"Source {i+1}: {doc.get('source', 'Unknown')}"
                    ]),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Page: "),
                            str(doc.get('page', 'N/A'))
                        ], className="mb-2"),
                        html.P(doc['content'], className="text-muted small mb-0")
                    ])
                ], className="mb-2")
                sources_content.append(source_card)
            
            if sources_content:
                sources_card = dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-journals me-2"),
                        "Source Documents"
                    ]),
                    dbc.CardBody(sources_content)
                ], className="mb-4")
                response_content.append(sources_card)
        
        success_status = dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            "Query processed successfully"
        ], color="success", dismissable=True)
        
        return response_content, success_status
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        error_alert = dbc.Alert([
            html.I(className="bi bi-exclamation-circle me-2"),
            f"An error occurred: {str(e)}"
        ], color="danger")
        return "", error_alert

# Make server available for Gunicorn
server = app.server

# Update the server port if provided via command line
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the Finance RAG System')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    # Update the URL in the startup message
    print(f"\n🔗 URL: http://localhost:{args.port}")
    print("=" * 60)
    
    # Run the app with the specified port
    app.run(host='0.0.0.0', port=args.port, debug=False)

# Keyboard shortcut for query input
app.clientside_callback(
    """
    function(id) {
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && event.target.id === 'question-input') {
                document.getElementById('ask-button').click();
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output('question-input', 'id'),
    [Input('question-input', 'id')]
)

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