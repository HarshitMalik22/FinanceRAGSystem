// Use environment variable if set, otherwise use production URL in production or localhost in development
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://financeragsystem.onrender.com' 
    : 'http://localhost:5000');

export const config = {
  api: {
    baseUrl: API_BASE_URL,
    endpoints: {
      upload: `${API_BASE_URL}/api/upload`,
      query: `${API_BASE_URL}/api/query`,
      documents: `${API_BASE_URL}/api/documents`,
    },
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
  },
};

export default config;