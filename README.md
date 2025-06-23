# Finance RAG System

A modern, containerized Finance-focused Retrieval-Augmented Generation (RAG) system with a React TypeScript frontend and Python FastAPI backend, powered by LangChain and Google's Gemini.

## 🚀 Features

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
│       ├── components/   # Reusable UI components
│       ├── hooks/        # Custom React hooks
│       ├── pages/        # Page components
│       └── services/     # API services
│
├── backend/             # Python FastAPI backend
│   ├── app/              # Application code
│   │   ├── api/          # API routes
│   │   ├── core/         # Core functionality
│   │   └── models/       # Data models
│   └── tests/            # Test suite
│
├── docker/              # Docker configuration
├── .github/              # GitHub workflows
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Node.js 18+ (for frontend development)
- Python 3.10+ (for backend development)

### Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FinanceRAGSystem.git
   cd FinanceRAGSystem
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Update `.env` with your API keys and configuration

4. Start the application:
   ```bash
   docker-compose up -d
   ```

5. Access the application at [http://localhost:3000](http://localhost:3000)

### Development Setup

1. Install dependencies:
   ```bash
   # Install backend dependencies
   cd backend
   poetry install
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

2. Start the development servers:
   ```bash
   # In one terminal (backend)
   cd backend
   poetry run uvicorn app.main:app --reload
   
   # In another terminal (frontend)
   cd frontend
   npm start
   ```

## 🐳 Deployment

### Production Deployment

1. Build and start the production stack:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   ```

2. The application will be available on port 3000

### Cloud Deployment

#### Google Cloud Run

```bash
# Build and push the container
gcloud builds submit --tag gcr.io/your-project-id/finance-rag

# Deploy to Cloud Run
gcloud run deploy finance-rag \
  --image gcr.io/your-project-id/finance-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_API_KEY=your-api-key"
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example` with the following variables:

```env
# Backend
PORT=8080
ENVIRONMENT=development
DEBUG=True

# Frontend
REACT_APP_API_URL=http://localhost:8080

# LLM Configuration
LLM_PROVIDER=google  # google or openai
GOOGLE_API_KEY=your-google-api-key
# OPENAI_API_KEY=your-openai-api-key

# Database
DATABASE_URL=postgresql://user:password@db:5432/financerag

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
cd backend
poetry run pytest

# Frontend tests
cd frontend
npm test

# End-to-end tests
npm run test:e2e
```

## 📚 Documentation

- [API Documentation](http://localhost:8080/docs) (available when running locally)
- [Frontend Documentation](./frontend/README.md)
- [Backend Documentation](./backend/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Google Gemini](https://ai.google.dev/)
- [React](https://reactjs.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Material-UI](https://mui.com/)
