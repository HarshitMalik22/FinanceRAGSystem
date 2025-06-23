import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Document as DocumentType } from '../types';
import api from '../services/api';

export const useDocuments = () => {
  const queryClient = useQueryClient();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  // Fetch all documents
  const { data: documents = [], isLoading, error } = useQuery<DocumentType[]>({
    queryKey: ['documents'],
    queryFn: async () => {
      const apiDocuments = await api.getDocuments();
      // Transform API documents to match our DocumentType
      return apiDocuments.map(doc => ({
        id: doc.id,
        name: doc.name,
        size: doc.size,
        type: doc.type,
        uploadedAt: doc.uploadDate || new Date().toISOString(),
        status: 'uploaded' as const
      }));
    },
  });

  // Upload documents mutation
  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await api.uploadDocuments(formData);
      return response;
    },
    onSuccess: () => {
      // Invalidate and refetch documents list
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setSelectedFiles([]);
      setUploadProgress(0);
    },
  });

  // Delete document mutation
  const deleteMutation = useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId),
    onSuccess: () => {
      // Invalidate and refetch documents list
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  // Handle file selection
  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;
    
    const newFiles = Array.from(files);
    setSelectedFiles(prev => [...prev, ...newFiles]);
  };

  // Handle file removal
  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Handle document upload
  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    
    try {
      await uploadMutation.mutateAsync(selectedFiles);
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  };

  // Handle document deletion
  const handleDeleteDocument = async (documentId: string) => {
    try {
      await deleteMutation.mutateAsync(documentId);
    } catch (error) {
      console.error('Delete failed:', error);
      throw error;
    }
  };

  return {
    documents,
    selectedFiles,
    uploadProgress,
    isLoading,
    error,
    isUploading: uploadMutation.isPending,
    isDeleting: deleteMutation.isPending,
    handleFileSelect,
    handleRemoveFile,
    handleUpload,
    handleDeleteDocument,
  };
};

export default useDocuments;
