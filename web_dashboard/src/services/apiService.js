import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('elvis_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor to handle errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          localStorage.removeItem('elvis_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  setAuthToken(token) {
    if (token) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete this.client.defaults.headers.common['Authorization'];
    }
  }

  clearAuthToken() {
    delete this.client.defaults.headers.common['Authorization'];
  }

  // Authentication
  async login(username, password) {
    return this.client.post('/api/auth/login', { username, password });
  }

  // Bot control
  async getBotStatus() {
    return this.client.get('/api/bot/status');
  }

  async startBot(mode = 'paper', strategy = 'ensemble') {
    return this.client.post('/api/bot/start', { mode, strategy });
  }

  async stopBot() {
    return this.client.post('/api/bot/stop');
  }

  // Account data
  async getBalance() {
    return this.client.get('/api/account/balance');
  }

  async getPositions() {
    return this.client.get('/api/positions');
  }

  // Trading data
  async getTradeHistory(params = {}) {
    return this.client.get('/api/trades/history', { params });
  }

  async getPerformanceStats() {
    return this.client.get('/api/performance/stats');
  }

  // Market data
  async getMarketPrice(symbol) {
    return this.client.get(`/api/market/price/${symbol}`);
  }

  async getMarketIndicators(symbol) {
    return this.client.get(`/api/market/indicators/${symbol}`);
  }

  // Configuration
  async getConfig() {
    return this.client.get('/api/config');
  }

  async updateConfig(config) {
    return this.client.put('/api/config', config);
  }

  // Dashboard-specific endpoints
  async getDashboardOverview() {
    return this.client.get('/api/dashboard/overview');
  }

  async getPerformanceChartData(days = 7) {
    return this.client.get('/api/dashboard/charts/performance', { params: { days } });
  }

  async getDashboardAlerts() {
    return this.client.get('/api/dashboard/alerts');
  }

  async markAlertRead(alertId) {
    return this.client.post(`/api/dashboard/alerts/${alertId}/read`);
  }

  async getWebSocketClients() {
    return this.client.get('/api/dashboard/websocket/clients');
  }

  async broadcastAlert(type, message, severity = 'info') {
    return this.client.post('/api/dashboard/broadcast/alert', {
      type,
      message,
      severity,
    });
  }

  // Health check
  async healthCheck() {
    return this.client.get('/health');
  }
}

export const apiService = new ApiService();
export default apiService;