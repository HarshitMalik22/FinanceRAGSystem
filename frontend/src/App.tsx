import React, { useState, useCallback } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  Chip,
  Grid,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SendIcon from '@mui/icons-material/Send';
import { useDropzone } from 'react-dropzone';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

interface QueryResponse {
  result: string;
  source_documents?: Array<{
    content: string;
    metadata: any;
  }>;
  processing_time_seconds?: number;
  error?: string;
}

interface UploadResponse {
  message: string;
  document_id: string;
  filename: string;
  chunks_processed: number;
  processing_time_seconds: number;
  error?: string;
}

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? window.location.origin 
    : 'http://localhost:5000';

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    setError(null);

    for (const file of acceptedFiles) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });

        const result: UploadResponse = await response.json();

        if (response.ok) {
          setUploadedFiles(prev => [...prev, result.filename]);
        } else {
          setError(result.error || 'Upload failed');
        }
      } catch (err) {
        setError(`Error uploading ${file.name}: ${err}`);
      }
    }

    setUploading(false);
  }, [API_BASE_URL]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: true,
  });

  const handleQuery = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      const result: QueryResponse = await response.json();

      if (response.ok) {
        setAnswer(result);
      } else {
        setError(result.error || 'Query failed');
      }
    } catch (err) {
      setError(`Error: ${err}`);
    }

    setLoading(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Finance RAG System
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" paragraph>
          Upload financial documents and ask questions about them using AI
        </Typography>

        <Grid container spacing={4}>
          {/* Upload Section */}
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Upload Documents
              </Typography>
              
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to select files
                </Typography>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Supported: PDF, DOCX, TXT, CSV, XLSX
                </Typography>
              </Box>

              {uploading && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  <Typography variant="body2">Uploading...</Typography>
                </Box>
              )}

              {uploadedFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Uploaded Files:
                  </Typography>
                  {uploadedFiles.map((filename, index) => (
                    <Chip
                      key={index}
                      label={filename}
                      size="small"
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Query Section */}
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Ask Questions
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={4}
                variant="outlined"
                label="Enter your question about the uploaded documents"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <Button
                variant="contained"
                onClick={handleQuery}
                disabled={loading || !question.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                fullWidth
              >
                {loading ? 'Processing...' : 'Ask Question'}
              </Button>
            </Paper>
          </Grid>
        </Grid>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mt: 3 }}>
            {error}
          </Alert>
        )}

        {/* Answer Display */}
        {answer && (
          <Card sx={{ mt: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Answer
              </Typography>
              <Typography variant="body1" paragraph>
                {answer.result}
              </Typography>
              
              {answer.processing_time_seconds && (
                <Typography variant="caption" color="text.secondary">
                  Processing time: {answer.processing_time_seconds}s
                </Typography>
              )}

              {answer.source_documents && answer.source_documents.length > 0 && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Sources
                  </Typography>
                  {answer.source_documents.map((doc, index) => (
                    <Box key={index} sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Source {index + 1}:
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {doc.content.substring(0, 200)}...
                      </Typography>
                    </Box>
                  ))}
                </>
              )}
            </CardContent>
          </Card>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;