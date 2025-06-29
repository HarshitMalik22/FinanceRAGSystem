// Configuration for the Finance RAG System frontend
const isDevelopment = process.env.NODE_ENV === 'development';
const isProduction = process.env.NODE_ENV === 'production';

// Determine the API base URL based on environment
let API_BASE_URL: string;

if (process.env.REACT_APP_API_BASE_URL) {
  // Use explicitly set environment variable
  API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
} else if (isProduction) {
  // Production URL
  API_BASE_URL = 'https://financeragsystem.onrender.com';
} else {
  // Development - check if we're running in Docker or locally
  const hostname = window.location.hostname;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    API_BASE_URL = 'http://localhost:5000';
  } else {
    // Running in Docker or other containerized environment
    API_BASE_URL = `http://${hostname}:5000`;
  }
}

console.log(`[Config] Environment: ${process.env.NODE_ENV}`);
console.log(`[Config] API Base URL: ${API_BASE_URL}`);

export const config = {
  api: {
    baseUrl: API_BASE_URL,
    endpoints: {
      upload: `${API_BASE_URL}/api/upload`,
      query: `${API_BASE_URL}/api/query`,
      documents: `${API_BASE_URL}/api/documents`,
    },
    timeout: 30000, // 30 seconds
    retries: 3,
  },
  app: {
    name: 'Finance RAG System',
    description: 'A sophisticated financial document analysis system',
    maxFileSize: 50 * 1024 * 1024, // 50MB
    supportedFileTypes: [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ],
    supportedExtensions: ['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
  },
};

export default config;