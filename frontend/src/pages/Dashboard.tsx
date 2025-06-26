import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Paper, 
  TextField, 
  Typography, 
  useTheme,
  IconButton,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  Snackbar,
  Alert
} from '@mui/material';
import { 
  CloudUpload as CloudUploadIcon, 
  Description as DescriptionIcon, 
  Search as SearchIcon,
  Delete as DeleteIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { Document, QueryResponse } from '../types/api';
import api from '../services/api';

// Extend the Document interface if needed
interface ExtendedDocument extends Document {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadDate: string;
  status: 'uploaded' | 'processing' | 'error';
  uploadedAt?: string; 
}

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const [query, setQuery] = useState('');
  const [documents, setDocuments] = useState<ExtendedDocument[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [snackbar, setSnackbar] = useState<{ 
    open: boolean; 
    message: string; 
    severity: 'success' | 'error' | 'info' 
  }>({ 
    open: false, 
    message: '', 
    severity: 'info' 
  });

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleQuery = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) {
      showSnackbar('Please enter a question', 'info');
      return;
    }
    try {
      await executeQuery();
    } catch (error) {
      console.error('Query error:', error);
      showSnackbar('Error executing query', 'error');
    }
  };

  const { data: queryResult, refetch: executeQuery, isLoading: isQueryLoading, error: queryError } = useQuery<QueryResponse>({
    queryKey: ['query', query],
    queryFn: () => api.query(query),
    enabled: false,
    retry: false,
  });

  React.useEffect(() => {
    if (queryError) {
      showSnackbar('Error executing query', 'error');
      console.error('Query error:', queryError);
    }
  }, [queryError]);
  
  const isQuerying = isQueryLoading;

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const files = Array.from(event.target.files);
      setSelectedFiles(prev => [...prev, ...files]);
      
      setDocuments(prev => [
        ...prev,
        ...files.map(file => ({
          id: Math.random().toString(36).substring(2, 9),
          name: file.name,
          size: file.size,
          type: file.type,
          uploadDate: new Date().toISOString(),
          uploadedAt: new Date().toISOString(),
          status: 'uploaded' as const,
        }))
      ]);
      
      // Reset the input value to allow selecting the same file again if needed
      event.target.value = '';
    }
  };

  const handleRemoveDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
    showSnackbar('Document removed', 'info');
  };

  // Define the server document type
  interface ServerDocument {
    document_id: string;
    filename: string;
    file_size: number;
    chunks_processed: number;
    upload_time: string;
    processing_time_seconds: number;
  }

  // Define the upload response type
  interface UploadResponse {
    message: string;
    document_id: string;
    filename: string;
    file_size: number;
    chunks_processed: number;
    upload_time: string;
    processing_time_seconds: number;
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setSnackbar({
        open: true,
        message: 'Please select at least one file to upload',
        severity: 'error'
      });
      return;
    }

    // Create a new array of documents with processing status
    const newDocuments = selectedFiles.map(file => ({
      id: Math.random().toString(36).substring(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      uploadDate: new Date().toISOString(),
      status: 'processing' as const,
      file
    }));

    // Add new documents to the state
    setDocuments(prev => [...prev, ...newDocuments]);
    
    // Process each file one by one
    for (const doc of newDocuments) {
      const formData = new FormData();
      formData.append('file', doc.file);
      
      try {
        const response = await api.uploadDocuments(formData);
        
        if (response && response.message === 'File processed successfully') {
          // Update the document with server response
          setDocuments(prev => 
            prev.map(d => 
              d.id === doc.id
                ? { 
                    ...d, 
                    id: response.document_id,
                    size: response.file_size,
                    uploadDate: response.upload_time || new Date().toISOString(),
                    type: response.filename.split('.').pop()?.toUpperCase() || 'UNKNOWN',
                    status: 'uploaded' as const,
                    file: undefined // Remove the file reference
                  } 
                : d
            )
          );
          
          setSnackbar({
            open: true,
            message: `${doc.name} uploaded successfully`,
            severity: 'success'
          });
        } else {
          throw new Error(response?.message || 'Unknown error occurred');
        }
      } catch (error) {
        console.error('Upload error:', error);
        
        // Update status to error for the failed document
        setDocuments(prev => 
          prev.map(d => 
            d.id === doc.id 
              ? { ...d, status: 'error' as const } 
              : d
          )
        );
        
        setSnackbar({
          open: true,
          message: `Error uploading ${doc.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
          severity: 'error'
        });
      }
    }
    
    // Clear selected files after processing
    setSelectedFiles([]);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 4, width: '100%' }}>
        <Box sx={{ width: { xs: '100%', md: 'calc(33.333% - 32px)' }, minWidth: { md: '300px' } }}>
          <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                <DescriptionIcon sx={{ mr: 1 }} /> Document Upload
              </Typography>
              
              <Box 
                sx={{
                  border: '2px dashed',
                  borderColor: 'divider',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  mb: 3,
                  flexGrow: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'action.hover'
                  }
                }}
                component="label"
              >
                <input
                  type="file"
                  multiple
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                  aria-label="Upload documents"
                  title="Upload documents"
                />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" color="text.secondary" gutterBottom>
                  Drag & drop files here or click to browse
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Supported formats: PDF, DOCX, TXT, CSV, XLSX (Max 50MB)
                </Typography>
              </Box>
              
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={handleUpload}
                disabled={documents.length === 0}
                sx={{ mb: 2, py: 1.5 }}
              >
                Process Documents
              </Button>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Uploaded Documents ({documents.length})
              </Typography>
              
              <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', maxHeight: 300 }}>
                {documents.length === 0 ? (
                  <Box sx={{ p: 2, textAlign: 'center', color: 'text.secondary' }}>
                    No documents uploaded yet
                  </Box>
                ) : (
                  <List dense>
                    {documents.map((doc) => (
                      <motion.div
                        key={doc.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -100 }}
                        layout
                      >
                        <ListItem
                          secondaryAction={
                            <Tooltip title="Remove document">
                              <IconButton 
                                edge="end" 
                                aria-label="delete"
                                onClick={() => handleRemoveDocument(doc.id)}
                                size="small"
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          }
                          sx={{
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            '&:last-child': { borderBottom: 'none' },
                            pr: 8
                          }}
                        >
                          <ListItemIcon>
                            {doc.status === 'processing' ? (
                              <CircularProgress size={20} />
                            ) : doc.status === 'error' ? (
                              <ErrorIcon color="error" />
                            ) : (
                              <CheckCircleIcon color="success" />
                            )}
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Typography variant="body2" noWrap>
                                {doc.name}
                              </Typography>
                            }
                            secondary={
                              <>
                                <Typography variant="caption" display="block">
                                  {formatFileSize(doc.size)}
                                </Typography>
                                {doc.status === 'error' && (
                                  <Typography variant="caption" color="error">
                                    Error processing
                                  </Typography>
                                )}
                              </>
                            }
                          />
                        </ListItem>
                      </motion.div>
                    ))}
                  </List>
                )}
              </Paper>
            </CardContent>
          </Card>
        </Box>
        
        {/* Main Content - Chat Interface */}
        <Box sx={{ flex: 1, minWidth: { xs: '100%', md: 'calc(66.666% - 32px)' } }}>
          <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', p: 0 }}>
              {/* Chat Header */}
              <Box sx={{ p: 3, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Financial Document Assistant
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Ask questions about your financial documents
                </Typography>
              </Box>
              
              {/* Chat Messages */}
              <Box 
                sx={{ 
                  flexGrow: 1, 
                  overflowY: 'auto',
                  p: 3,
                  minHeight: 400,
                  maxHeight: 'calc(100vh - 400px)',
                  background: theme.palette.background.default
                }}
              >
                <AnimatePresence>
                  {queryResult ? (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Box sx={{ mb: 4 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Box sx={{ 
                            bgcolor: 'primary.main', 
                            color: 'primary.contrastText',
                            borderRadius: '50%',
                            width: 32,
                            height: 32,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 2,
                            flexShrink: 0
                          }}>
                            <InfoIcon fontSize="small" />
                          </Box>
                          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            Assistant
                          </Typography>
                        </Box>
                        <Box sx={{ pl: 6, pr: 2 }}>
                          <Typography variant="body1" sx={{ whiteSpace: 'pre-line', mb: 2 }}>
                            {queryResult?.result || 'No results found.'}
                          </Typography>
                          
                          {queryResult?.source_documents && queryResult.source_documents.length > 0 && (
                            <Box sx={{ mt: 2, borderTop: `1px solid ${theme.palette.divider}`, pt: 2 }}>
                              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                Sources:
                              </Typography>
                              <List dense>
                                {queryResult.source_documents.map((doc: any, index: number) => (
                                  <ListItem key={index} sx={{ pl: 0 }}>
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                      <DescriptionIcon fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                      primary={doc.metadata?.source || 'Document'}
                                      primaryTypographyProps={{ variant: 'body2' }}
                                      secondary={
                                        doc.page_content ? (
                                          <Typography variant="caption" component="div" sx={{ 
                                            display: '-webkit-box',
                                            WebkitLineClamp: 2,
                                            WebkitBoxOrient: 'vertical',
                                            overflow: 'hidden'
                                          }}>
                                            {doc.page_content}
                                          </Typography>
                                        ) : null
                                      }
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          )}
                        </Box>
                      </Box>
                    </motion.div>
                  ) : isQuerying ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                      <CircularProgress />
                    </Box>
                  ) : (
                    <Box sx={{ 
                      height: '100%', 
                      display: 'flex', 
                      flexDirection: 'column', 
                      justifyContent: 'center', 
                      alignItems: 'center',
                      textAlign: 'center',
                      p: 3,
                      color: 'text.secondary'
                    }}>
                      <SearchIcon sx={{ fontSize: 64, opacity: 0.5, mb: 2 }} />
                      <Typography variant="h6" gutterBottom>
                        Ask a question about your financial documents
                      </Typography>
                      <Typography variant="body2">
                        Upload documents and ask questions to get insights from your financial data.
                      </Typography>
                    </Box>
                  )}
                </AnimatePresence>
              </Box>
              
              {/* Input Area */}
              <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
                <form onSubmit={handleQuery}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                      fullWidth
                      variant="outlined"
                      placeholder="Ask a question about your documents..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      disabled={isQueryLoading}
                      label="Ask a question"
                      aria-label="Ask a question about your documents"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          '&:hover fieldset': {
                            borderColor: 'primary.main',
                          },
                          '&.Mui-focused fieldset': {
                            borderColor: 'primary.main',
                          },
                        },
                      }}
                    />
                    <Button 
                      type="submit"
                      variant="contained"
                      color="primary"
                      disabled={!query.trim() || isQueryLoading}
                      sx={{ minWidth: 'auto' }}
                    >
                      {isQueryLoading ? <CircularProgress size={24} color="inherit" /> : <SearchIcon />}
                    </Button>
                  </Box>
                </form>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Dashboard;
