"""
Price fetcher utilities for the ELVIS project.
This module provides functionality for fetching real-time cryptocurrency candlestick data.
"""

import websocket
import json
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, Optional, Callable

class PriceFetcher:
    """
    Real-time candlestick data fetcher for cryptocurrencies using Binance WebSocket.
    """
    
    def __init__(self, logger: logging.Logger, symbol: str = "BTCUSDT", timeframe: str = "1m"):
        """
        Initialize the price fetcher.
        
        Args:
            logger (logging.Logger): The logger to use.
            symbol (str): The symbol to fetch prices for (e.g., "BTCUSDT").
            timeframe (str): The candlestick timeframe (e.g., "1m", "5m").
        """
        self.logger = logger
        self.symbol = symbol.lower()  # Binance WebSocket requires lowercase
        self.timeframe = timeframe
        self.running = False
        self.current_candle = {
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'volume': 0.0,
            'closed': False
        }
        self.candle_history = deque(maxlen=50)  # Store last 50 candles
        self.callbacks = []
        self.lock = threading.Lock()
        self.ws = None
        self.thread = None

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        data = json.loads(message)
        if 'k' not in data:
            return
        
        kline = data['k']
        with self.lock:
            self.current_candle = {
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closed': kline['x']
            }
            if kline['x']:  # Candle is closed
                self.candle_history.append(self.current_candle.copy())
                self.logger.info(f"New {self.timeframe} candle: O:{self.current_candle['open']:.2f} "
                               f"H:{self.current_candle['high']:.2f} "
                               f"L:{self.current_candle['low']:.2f} "
                               f"C:{self.current_candle['close']:.2f}")
            # Call callbacks with current candle
            for callback in self.callbacks:
                try:
                    callback(self.current_candle)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info("WebSocket connection closed")
        self.running = False

    def on_open(self, ws):
        self.logger.info("Connected to Binance WebSocket")

    def start(self):
        """Start the price fetcher."""
        if self.running:
            return
        self.running = True
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.timeframe}"
        self.ws = websocket.WebSocketApp(ws_url,
                                       on_open=self.on_open,
                                       on_message=self.on_message,
                                       on_error=self.on_error,
                                       on_close=self.on_close)
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"Price fetcher started for {self.symbol} ({self.timeframe})")

    def stop(self):
        """Stop the price fetcher."""
        if not self.running:
            return
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=2)
        self.logger.info(f"Price fetcher stopped for {self.symbol}")

    def get_current_price(self) -> float:
        """Get the current close price."""
        with self.lock:
            return self.current_candle['close']

    def get_current_candle(self) -> Dict[str, float]:
        """Get the current candlestick data."""
        with self.lock:
            return self.current_candle.copy()

    def get_candle_history(self) -> list:
        """Get the candlestick history."""
        with self.lock:
            return list(self.candle_history)

    def register_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Register a callback for candle updates."""
        with self.lock:
            self.callbacks.append(callback)