export interface Document {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadDate: string;
}

export interface Source {
  source: string;
  content: string;
  page?: number;
}

export interface QueryResponse {
  answer: string;
  result: string;
  sources: Source[];
  source_documents: Array<{
    page_content: string;
    metadata: {
      source: string;
      [key: string]: any;
    };
  }>;
}

export interface DocumentListResponse {
  documents: Document[];
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}
