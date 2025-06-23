import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import config from '../config';
import { Document, QueryResponse, DocumentListResponse } from '../types/api';

// Create an Axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: config.api.baseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor for API calls
apiClient.interceptors.request.use(
  (config) => {
    // You can add auth tokens or other headers here if needed
    // const token = localStorage.getItem('authToken');
    // if (token) {
    //   config.headers['Authorization'] = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response.data;
  },
  (error: AxiosError) => {
    // Handle errors globally
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('API Error Response:', error.response.data);
      console.error('Status:', error.response.status);
      console.error('Headers:', error.response.headers);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('API Error Request:', error.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('API Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Document endpoints
  uploadDocuments: (formData: FormData): Promise<Document> => {
    return apiClient.post(config.api.endpoints.upload, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Query endpoint
  query: (question: string): Promise<QueryResponse> => {
    return apiClient.post(config.api.endpoints.query, { question });
  },

  // Get all documents
  getDocuments: (): Promise<Document[]> => {
    return apiClient.get(config.api.endpoints.documents);
  },

  // Delete a document
  deleteDocument: (documentId: string): Promise<{ success: boolean }> => {
    return apiClient.delete(`${config.api.endpoints.documents}/${documentId}`);
  },
};

export default api;
