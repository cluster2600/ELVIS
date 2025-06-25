import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(null);

  useEffect(() => {
    // Check for existing token on mount
    const savedToken = localStorage.getItem('elvis_token');
    if (savedToken) {
      try {
        // Verify token is valid
        const payload = JSON.parse(atob(savedToken.split('.')[1]));
        const currentTime = Date.now() / 1000;
        
        if (payload.exp > currentTime) {
          setToken(savedToken);
          setUser({ username: payload.user });
          setIsAuthenticated(true);
          apiService.setAuthToken(savedToken);
        } else {
          // Token expired
          localStorage.removeItem('elvis_token');
        }
      } catch (error) {
        console.error('Invalid token:', error);
        localStorage.removeItem('elvis_token');
      }
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    try {
      const response = await apiService.login(username, password);
      const { token } = response.data;
      
      localStorage.setItem('elvis_token', token);
      apiService.setAuthToken(token);
      
      // Decode token to get user info
      const payload = JSON.parse(atob(token.split('.')[1]));
      
      setToken(token);
      setUser({ username: payload.user });
      setIsAuthenticated(true);
      
      return { success: true };
    } catch (error) {
      console.error('Login failed:', error);
      return { 
        success: false, 
        error: error.response?.data?.error || 'Login failed' 
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('elvis_token');
    apiService.clearAuthToken();
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  const value = {
    isAuthenticated,
    user,
    token,
    loading,
    login,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};