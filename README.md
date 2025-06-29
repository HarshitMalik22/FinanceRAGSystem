# Finance RAG System

A modern, containerized Finance-focused Retrieval-Augmented Generation (RAG) system with a React TypeScript frontend and Python Flask backend, powered by LangChain, ChromaDB, and Google's Generative AI.

## 🚀 Quick Start (Development)

### Prerequisites
- Python 3.10+
- Node.js 16+
- Docker & Docker Compose (recommended)
- API keys for Groq and Google Generative AI

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FinanceRAGSystem.git
   cd FinanceRAGSystem
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start with Docker Compose (recommended)**
   ```bash
   docker-compose up --build
   ```

4. **Or run manually**
   ```bash
   # Backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m nltk.downloader popular
   python -m spacy download en_core_web_sm
   python finance_rag_app.py
   
   # Frontend (in a new terminal)
   cd frontend
   npm install
   npm start
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:5000/api

## 🚀 Production Deployment

### Option 1: Deploy to Render (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yourusername/FinanceRAGSystem)

1. Click the "Deploy to Render" button above
2. Set the required environment variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `GOOGLE_API_KEY`: Your Google API key
   - `FLASK_ENV`: `production`
3. Click "Create Web Service"

### Option 2: Docker

```bash
# Build the image
docker build -t finance-rag-app .

# Run the container
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_groq_key \
  -e GOOGLE_API_KEY=your_google_key \
  finance-rag-app
```

## 🔧 Configuration

### Environment Variables

| Variable           | Description                      | Default        |
|--------------------|----------------------------------|----------------|
| `GROQ_API_KEY`    | Groq API key for LLM             | -              |
| `GOOGLE_API_KEY`  | Google API key for embeddings    | -              |
| `VECTOR_STORE_DIR`| Directory for ChromaDB storage   | `./chroma_db`  |
| `FLASK_ENV`       | Flask environment               | `development`  |
| `PORT`            | Port to run the application     | `5000`         |

## 🎯 Features

- **Modern Web Interface**: Responsive React TypeScript frontend with Material-UI
- **Document Processing**: Upload and process PDF, DOCX, TXT, CSV, XLSX files
- **Natural Language QA**: Ask questions about your financial documents
- **Multi-Provider Support**: Google Gemini, OpenAI, and more
- **Containerized**: Easy deployment with Docker and Docker Compose
- **Production-Ready**: Multi-stage builds, health checks, and monitoring
- **Developer Friendly**: Complete development environment with hot-reloading

## 🏗️ Architecture

```
├── frontend/             # React TypeScript frontend
│   ├── public/           # Static files
│   └── src/              # React application source
├── backend/              # Python Flask backend
│   ├── app/              # Application code
│   ├── tests/            # Backend tests
│   └── requirements.txt  # Python dependencies
├── docker/               # Docker configuration
├── docker-compose.yml    # Local development
└── render.yaml           # Render.com deployment config
```

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Google Generative AI](https://ai.google/)
- [React](https://reactjs.org/)
- [Material-UI](https://mui.com/)
