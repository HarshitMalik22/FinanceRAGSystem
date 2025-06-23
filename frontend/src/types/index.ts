export interface Document {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: string;
  status: 'uploaded' | 'processing' | 'error';
}

export interface QueryResponse {
  answer: string;
  sources: Array<{ source: string; content: string }>;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Array<{ source: string; content: string }>;
}

export interface AppState {
  documents: Document[];
  selectedFiles: File[];
  messages: ChatMessage[];
  currentQuery: string;
  isLoading: boolean;
  error: string | null;
}
