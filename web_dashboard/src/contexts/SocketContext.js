import React, { createContext, useContext, useEffect, useState } from 'react';
import io from 'socket.io-client';
import { useAuth } from './AuthContext';

const SocketContext = createContext();

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState({
    botStatus: null,
    marketData: {},
    trades: [],
    positions: [],
    performance: null,
    alerts: [],
  });
  const { isAuthenticated, token } = useAuth();

  useEffect(() => {
    if (isAuthenticated && token) {
      // Create socket connection
      const newSocket = io(process.env.REACT_APP_API_URL || 'http://localhost:5000', {
        autoConnect: true,
        transports: ['websocket', 'polling'],
      });

      // Connection events
      newSocket.on('connect', () => {
        console.log('Connected to WebSocket server');
        setConnected(true);
        
        // Authenticate with the server
        newSocket.emit('authenticate', { token });
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from WebSocket server');
        setConnected(false);
      });

      newSocket.on('authenticated', (data) => {
        console.log('WebSocket authenticated:', data);
        
        // Subscribe to all data channels
        newSocket.emit('subscribe', {
          channels: ['bot_status', 'market_data', 'trades', 'positions', 'performance', 'alerts']
        });
      });

      newSocket.on('subscribed', (data) => {
        console.log('Subscribed to channels:', data.channels);
      });

      // Real-time data events
      newSocket.on('bot_status_update', (data) => {
        setRealTimeData(prev => ({ ...prev, botStatus: data }));
      });

      newSocket.on('market_data_update', (data) => {
        setRealTimeData(prev => ({
          ...prev,
          marketData: { ...prev.marketData, [data.symbol]: data }
        }));
      });

      newSocket.on('trades_update', (data) => {
        setRealTimeData(prev => ({ ...prev, trades: data.trades || [] }));
      });

      newSocket.on('positions_update', (data) => {
        setRealTimeData(prev => ({ ...prev, positions: data.positions || [] }));
      });

      newSocket.on('performance_update', (data) => {
        setRealTimeData(prev => ({ ...prev, performance: data }));
      });

      newSocket.on('alert', (alert) => {
        setRealTimeData(prev => ({
          ...prev,
          alerts: [alert, ...prev.alerts.slice(0, 49)] // Keep last 50 alerts
        }));
      });

      newSocket.on('trade_executed', (trade) => {
        console.log('Trade executed:', trade);
        // Add new trade to the beginning of the trades array
        setRealTimeData(prev => ({
          ...prev,
          trades: [trade, ...prev.trades.slice(0, 99)] // Keep last 100 trades
        }));
      });

      newSocket.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      setSocket(newSocket);

      return () => {
        newSocket.close();
      };
    } else {
      // Disconnect socket if not authenticated
      if (socket) {
        socket.close();
        setSocket(null);
        setConnected(false);
      }
    }
  }, [isAuthenticated, token]);

  const subscribeToChannel = (channel) => {
    if (socket && connected) {
      socket.emit('subscribe', { channels: [channel] });
    }
  };

  const unsubscribeFromChannel = (channel) => {
    if (socket && connected) {
      socket.emit('unsubscribe', { channels: [channel] });
    }
  };

  const getStatus = () => {
    if (socket && connected) {
      socket.emit('get_status');
    }
  };

  const getMarketData = (symbol) => {
    if (socket && connected) {
      socket.emit('get_market_data', { symbol });
    }
  };

  const value = {
    socket,
    connected,
    realTimeData,
    subscribeToChannel,
    unsubscribeFromChannel,
    getStatus,
    getMarketData,
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};