# Finance RAG System

A Finance-focused Retrieval-Augmented Generation (RAG) system built with LangChain, Google's Gemini, and Dash.

## 🚨 Important Security Notice 🚨

If you've previously used this repository, please rotate your API keys as older versions may have exposed them in the commit history.

## Features

- Upload and process various document types (PDF, DOCX, TXT, CSV, XLSX)
- Natural language question answering over documents
- Web-based interface with Dash
- Persistent vector storage with ChromaDB
- Document caching for improved performance
- Support for multiple LLM providers (Google Gemini, OpenAI)

## Prerequisites

- Python 3.10+
- Google Cloud account (for Gemini API)
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/FinanceRAGSystem.git
   cd FinanceRAGSystem
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   - Copy `.env.example` to `.env`
   - Update with your API keys:

     ```env
     # Required for Google Gemini
     LLM_PROVIDER=google
     GOOGLE_API_KEY=your_google_api_key_here
     
     # Optional: For OpenAI
     # LLM_PROVIDER=openai
     # OPENAI_API_KEY=your_openai_api_key
     
     # Optional: Tweak these settings as needed
     CHUNK_SIZE=1000
     CHUNK_OVERLAP=200
     TEMPERATURE=0.1
     ```

5. **Run the application**

   ```bash
   python finance_rag_app.py
   ```

   The application will be available at [http://localhost:8050](http://localhost:8050).

## 🧪 Testing the System

Run the test script to verify your setup:

```bash
python test_rag.py
```

This will:

1. Check your environment configuration
2. Download a sample PDF
3. Process it through the RAG system
4. Run a test query

## 🐳 Docker Deployment

1. Build the Docker image:

   ```bash
   docker build -t finance-rag .
   ```

2. Run the container:

   ```bash
   docker run -p 8050:8080 --env-file .env finance-rag
   ```

## 🔒 Security Best Practices

1. **Never commit API keys** - The `.env` file is in `.gitignore`
2. **Use environment variables** for all sensitive configuration
3. **Rotate API keys** regularly
4. **Monitor usage** through your cloud provider's dashboard

## 🛠 Troubleshooting

### Common Issues

1. **Missing Dependencies**

   - Run `pip install -r requirements.txt --upgrade`
   - Make sure you have system dependencies (like `python3-dev` and `build-essential`)

2. **API Connection Issues**
   - Verify your API key is correct
   - Check your internet connection
   - Ensure you have sufficient quota

3. **Document Processing Failures**
   - Check file permissions
   - Ensure documents are not password protected
   - Verify document format is supported

## 📚 Documentation

For more detailed information, see the [documentation](./docs/).

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 2. Deploy using Cloud Build

1. Set your project ID:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Submit the build to Cloud Build:
   ```bash
   gcloud builds submit --config=cloudbuild.yaml --substitutions=_GEMINI_API_KEY=your_gemini_api_key
   ```

### 3. Alternative: Manual Deployment

1. Build and push the Docker image:
   ```bash
   gcloud auth configure-docker
   docker build -t gcr.io/YOUR_PROJECT_ID/finance-rag-app .
   docker push gcr.io/YOUR_PROJECT_ID/finance-rag-app
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy finance-rag-app \
     --image gcr.io/YOUR_PROJECT_ID/finance-rag-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GEMINI_API_KEY=your_gemini_api_key \
     --port 8080
   ```

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `PORT`: Port to run the application on (default: 8050 for local, 8080 for production)
- `CHROMA_DB_PATH`: Path to store ChromaDB data (default: "./chroma_db")

## Architecture

The application consists of:

- **Frontend**: Dash web interface
- **Backend**: Python/FastAPI
- **Vector Store**: ChromaDB with Google Gemini embeddings
- **LLM**: Google's Gemini Pro

## License

MIT
