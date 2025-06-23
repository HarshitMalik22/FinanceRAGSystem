/**
 * Formats a file size in bytes to a human-readable string
 * @param bytes - File size in bytes
 * @returns Formatted file size string (e.g., "2.5 MB")
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

/**
 * Gets the appropriate icon for a file based on its type
 * @param fileType - MIME type of the file
 * @returns Icon name from Material-UI icons
 */
export const getFileIcon = (fileType: string): string => {
  if (!fileType) return 'insert_drive_file';
  
  const typeMap: Record<string, string> = {
    'application/pdf': 'picture_as_pdf',
    'application/msword': 'description',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'description',
    'text/plain': 'text_snippet',
    'text/csv': 'table_chart',
    'application/vnd.ms-excel': 'table_chart',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'table_chart',
    'application/zip': 'folder_zip',
    'application/x-rar-compressed': 'folder_zip',
    'application/x-7z-compressed': 'folder_zip',
    'image/': 'image',
    'audio/': 'audiotrack',
    'video/': 'videocam',
  };
  
  // Check for exact matches first
  if (typeMap[fileType]) {
    return typeMap[fileType];
  }
  
  // Check for partial matches (e.g., image/*)
  for (const [prefix, icon] of Object.entries(typeMap)) {
    if (prefix.endsWith('/') && fileType.startsWith(prefix)) {
      return icon;
    }
  }
  
  return 'insert_drive_file';
};

/**
 * Truncates a string to a specified length and adds an ellipsis if needed
 * @param str - Input string
 * @param maxLength - Maximum length before truncation
 * @returns Truncated string with ellipsis if needed
 */
export const truncateString = (str: string, maxLength: number): string => {
  if (!str) return '';
  if (str.length <= maxLength) return str;
  return `${str.substring(0, maxLength)}...`;
};

/**
 * Formats a date to a readable string
 * @param date - Date object or ISO string
 * @returns Formatted date string (e.g., "Jan 1, 2023, 2:30 PM")
 */
export const formatDate = (date: Date | string): string => {
  if (!date) return '';
  
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(dateObj);
};

/**
 * Debounce a function call
 * @param func - Function to debounce
 * @param wait - Wait time in milliseconds
 * @returns Debounced function
 */
export const debounce = <F extends (...args: any[]) => any>(
  func: F,
  wait: number
): ((...args: Parameters<F>) => void) => {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  return function(this: ThisParameterType<F>, ...args: Parameters<F>) {
    const context = this;
    
    const later = () => {
      timeout = null;
      func.apply(context, args);
    };
    
    if (timeout !== null) {
      clearTimeout(timeout);
    }
    
    timeout = setTimeout(later, wait);
  };
};

/**
 * Generates a unique ID
 * @returns A unique string ID
 */
export const generateId = (): string => {
  return Math.random().toString(36).substr(2, 9);
};

/**
 * Validates if a file type is supported
 * @param file - File object
 * @param supportedTypes - Array of supported MIME types
 * @returns Boolean indicating if the file type is supported
 */
export const isValidFileType = (
  file: File,
  supportedTypes: string[]
): boolean => {
  if (!file || !supportedTypes || supportedTypes.length === 0) return false;
  
  // If any type is allowed
  if (supportedTypes.includes('*/*')) return true;
  
  // Check exact match or type/* pattern
  return supportedTypes.some(type => {
    // Exact match
    if (type === file.type) return true;
    
    // Wildcard match (e.g., image/*)
    if (type.endsWith('/*')) {
      const [typePrefix] = type.split('/');
      const [fileTypePrefix] = file.type.split('/');
      return typePrefix === fileTypePrefix;
    }
    
    return false;
  });
};

/**
 * Converts a file to a base64 string
 * @param file - File to convert
 * @returns Promise that resolves to the base64 string
 */
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = error => reject(error);
  });
};

/**
 * Creates a download link for a file
 * @param data - File data as string or Blob
 * @param filename - Name of the file to download
 * @param type - MIME type of the file
 */
export const downloadFile = (
  data: string | Blob,
  filename: string,
  type: string = 'application/octet-stream'
): void => {
  const blob = typeof data === 'string' ? new Blob([data], { type }) : data;
  const url = window.URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  
  // Cleanup
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
};
