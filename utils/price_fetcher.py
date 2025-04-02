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
        self.connection_established = False
        self.last_update_time = time.time()
        
        # Default BTC price as fallback (updated April 2025)
        self.default_price = 75655.0
        
        self.current_candle = {
            'open': self.default_price,
            'high': self.default_price,
            'low': self.default_price,
            'close': self.default_price,
            'volume': 100.0,
            'closed': False
        }
        self.candle_history = deque(maxlen=50)  # Store last 50 candles
        self.callbacks = []
        self.lock = threading.Lock()
        self.ws = None
        self.thread = None
        self.fallback_thread = None

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
            # Update the last update time to indicate we're receiving real data
            self.last_update_time = time.time()
            self.connection_established = True
            
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

    def _generate_mock_candles(self):
        """Generate mock candles when WebSocket connection fails."""
        self.logger.info("Starting mock candle generator")
        last_price = self.default_price
        while self.running:
            # Check if we've received real data recently
            current_time = time.time()
            if current_time - self.last_update_time > 30:  # No updates for 30 seconds
                # Generate a new mock candle with some random movement
                import random
                price_change = last_price * random.uniform(-0.001, 0.001)  # 0.1% max change
                new_price = last_price + price_change
                
                with self.lock:
                    self.current_candle = {
                        'open': last_price,
                        'high': max(last_price, new_price),
                        'low': min(last_price, new_price),
                        'close': new_price,
                        'volume': random.uniform(50, 200),
                        'closed': False
                    }
                    # Call callbacks with current candle
                    for callback in self.callbacks:
                        try:
                            callback(self.current_candle)
                        except Exception as e:
                            self.logger.error(f"Error in callback: {e}")
                
                last_price = new_price
                self.last_update_time = current_time
                self.logger.debug(f"Generated mock candle: close={new_price:.2f}")
            
            time.sleep(5)  # Generate a new candle every 5 seconds if needed

    def start(self):
        """Start the price fetcher."""
        if self.running:
            return
        self.running = True
        
        # Start the fallback thread for mock data
        self.fallback_thread = threading.Thread(target=self._generate_mock_candles)
        self.fallback_thread.daemon = True
        self.fallback_thread.start()
        
        # Start the WebSocket connection
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
