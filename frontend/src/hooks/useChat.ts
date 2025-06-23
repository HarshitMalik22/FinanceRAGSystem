import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import api from '../services/api';

export interface Source {
  source: string;
  content: string;
  page?: number;
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Source[];
}

export const useChat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);


  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: uuidv4(),
      content,
      role: 'user',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.query(content);
      
      const assistantMessage: Message = {
        id: uuidv4(),
        content: response.answer || 'I apologize, but I encountered an error processing your request.',
        role: 'assistant',
        timestamp: new Date(),
        sources: response.sources,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
      
      const errorMessage: Message = {
        id: uuidv4(),
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear the chat history
  const clearChat = () => {
    setMessages([]);
  };

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
  };
};

export default useChat;
