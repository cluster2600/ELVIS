"""
PriceFetcher - Real-time Binance price streaming and indicator calculation for ELVIS Trading Bot

Handles:
- Fetching historical candles from Binance REST API with Redis caching.
- Streaming real-time kline updates via WebSocket.
- Calculating technical indicators (RSI, MACD, SMA, EMA) with caching.
- Updating Prometheus metrics for monitoring.

Dependencies: binance, websocket-client, prometheus_client, threading, logging, redis
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta

from binance.client import Client
import websocket

import pandas as pd
from prometheus_client import Gauge

from utils.redis_cache import get_cache, make_price_key, make_indicator_key

# Prometheus metrics
CURRENT_PRICE = Gauge('elvis_current_price', 'Current BTC price', ['symbol'])
RSI_GAUGE = Gauge('elvis_rsi', 'Relative Strength Index', ['symbol'])
MACD_GAUGE = Gauge('elvis_macd', 'MACD line', ['symbol'])
MACD_SIGNAL_GAUGE = Gauge('elvis_macd_signal', 'MACD signal line', ['symbol'])
SMA_GAUGE = Gauge('elvis_sma', 'Simple Moving Average', ['symbol'])
EMA_SHORT_GAUGE = Gauge('elvis_ema_short', 'Short-term Exponential Moving Average', ['symbol'])
EMA_LONG_GAUGE = Gauge('elvis_ema_long', 'Long-term Exponential Moving Average', ['symbol'])

class PriceFetcher:
    """
    Class to fetch and stream BTC price data and calculate technical indicators.
    """

    def __init__(self, logger, client=None, symbols=['BTCUSDT'], timeframe='5m', history_limit=200):
        self.logger = logger or logging.getLogger(__name__)
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.history_limit = history_limit
        self.ws = None
        self.candles = {symbol: [] for symbol in symbols}
        self.running = False
        self.lock = threading.Lock()
        self.cache = get_cache()
        self.cache_ttl = 60  # 1 minute cache for prices
        self.indicator_cache_ttl = 30  # 30 seconds for indicators

    def get_historical_data(self):
        """
        Fetch historical kline/candle data from Binance REST API for all symbols.
        """
        if self.client:
            for symbol in self.symbols:
                try:
                    klines = self.client.get_historical_klines(symbol=symbol, interval=self.timeframe, limit=self.history_limit)
                    with self.lock:
                        self.candles[symbol] = klines
                    self.logger.info(f"Fetched {len(klines)} historical klines for {symbol}.")
                except Exception as e:
                    self.logger.error(f"Error fetching historical klines for {symbol}: {e}")
        else:
            self.logger.error("Client not initialized.")

    def calculate_indicators(self, symbol):
        """
        Calculate and publish technical indicators for a specific symbol from the latest candle data with caching.
        """
        try:
            if len(self.candles[symbol]) < 26:
                return  # Not enough data for EMA, MACD, etc.

            df = pd.DataFrame(self.candles[symbol], columns=['open_time', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(7)])
            df['close'] = df['close'].astype(float)

            close_prices = df['close']
            current_price = close_prices.iloc[-1]

            # Check cache for indicators
            indicators_cache_key = f"indicators:{symbol}:all"
            cached_indicators = self.cache.get(indicators_cache_key)
            
            if cached_indicators:
                # Use cached values
                rsi = cached_indicators.get('rsi')
                macd = cached_indicators.get('macd')
                signal = cached_indicators.get('signal')
                sma = cached_indicators.get('sma')
                ema_short = cached_indicators.get('ema_short')
                ema_long = cached_indicators.get('ema_long')
            else:
                # Calculate fresh
                rsi = self.calculate_rsi(close_prices, window=14)
                macd, signal = self.calculate_macd(close_prices)
                sma = self.calculate_sma(close_prices, window=20)
                ema_short = self.calculate_ema(close_prices, window=9)
                ema_long = self.calculate_ema(close_prices, window=21)
                
                # Cache the indicators
                indicators_data = {
                    'rsi': rsi,
                    'macd': macd,
                    'signal': signal,
                    'sma': sma,
                    'ema_short': ema_short,
                    'ema_long': ema_long,
                    'timestamp': time.time()
                }
                self.cache.set(indicators_cache_key, indicators_data, ttl=self.indicator_cache_ttl)
                
                # Also cache individual indicators for specific queries
                self.cache.set(make_indicator_key(symbol, 'rsi', 14), rsi, ttl=self.indicator_cache_ttl)
                self.cache.set(make_indicator_key(symbol, 'sma', 20), sma, ttl=self.indicator_cache_ttl)

            # Update Prometheus Gauges
            CURRENT_PRICE.labels(symbol).set(current_price)
            RSI_GAUGE.labels(symbol).set(rsi)
            MACD_GAUGE.labels(symbol).set(macd)
            MACD_SIGNAL_GAUGE.labels(symbol).set(signal)
            SMA_GAUGE.labels(symbol).set(sma)
            EMA_SHORT_GAUGE.labels(symbol).set(ema_short)
            EMA_LONG_GAUGE.labels(symbol).set(ema_long)
            
            # Cache current price
            self.cache.set(make_price_key(symbol), current_price, ttl=self.cache_ttl)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")

    @staticmethod
    def calculate_rsi(close, window=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    @staticmethod
    def calculate_macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]

    @staticmethod
    def calculate_sma(close, window=20):
        return close.rolling(window=window).mean().iloc[-1]

    @staticmethod
    def calculate_ema(close, window=20):
        return close.ewm(span=window, adjust=False).mean().iloc[-1]

    def on_message(self, ws, message):
        """
        WebSocket message handler.
        Updates the latest candle and recalculates indicators.
        """
        try:
            data = json.loads(message)['data']
            if data.get('e') == 'kline':
                kline = data['k']
                symbol = kline['s']
                with self.lock:
                    self.candles[symbol].append([
                        kline['t'], kline['o'], kline['h'],
                        kline['l'], kline['c'], kline['v']
                    ])
                    if len(self.candles[symbol]) > self.history_limit:
                        self.candles[symbol].pop(0)
                self.calculate_indicators(symbol)
                self.logger.info(f"Received kline for {symbol}: {kline}")
            else:
                self.logger.debug(f"Ignoring message without 'e': {data}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info("WebSocket closed")
        self.running = False

    def on_open(self, ws):
        self.logger.info("WebSocket opened")
        params = [f"{symbol.lower()}@kline_{self.timeframe}" for symbol in self.symbols]
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))

    def start(self):
        """
        Start the price fetcher: historical fetch + WebSocket stream.
        """
        self.get_historical_data()
        
        streams = '/'.join([f"{symbol.lower()}@kline_{self.timeframe}" for symbol in self.symbols])
        self.ws = websocket.WebSocketApp(
            f"wss://stream.binance.com:9443/stream?streams={streams}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open
        self.running = True
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.logger.info("Price fetcher started.")

    def get_current_price(self, symbol):
        """Get current price with Redis caching"""
        # Try to get from cache first
        cache_key = make_price_key(symbol)
        cached_price = self.cache.get(cache_key)
        
        if cached_price is not None:
            return cached_price
        
        # If not in cache, get from candles
        with self.lock:
            if self.candles[symbol]:
                price = float(self.candles[symbol][-1][4])  # Close price
                # Cache the price
                self.cache.set(cache_key, price, ttl=self.cache_ttl)
                return price
            return None

    def get_current_candle(self, symbol):
        with self.lock:
            if self.candles[symbol]:
                return self.candles[symbol][-1]
            return None

    def get_candle_history(self, symbol):
        with self.lock:
            return list(self.candles[symbol])

    def get_order_book(self, symbol: str, limit: int = 100):
        """
        Fetch order book data for a specific symbol with caching.
        """
        cache_key = f"order_book:{symbol}:{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.debug(f"Returning cached order book for {symbol}")
            return cached_data

        if self.client:
            try:
                order_book = self.client.get_order_book(symbol=symbol, limit=limit)
                self.cache.set(cache_key, order_book, ttl=self.indicator_cache_ttl) # Use indicator TTL
                self.logger.info(f"Fetched and cached order book for {symbol}.")
                return order_book
            except Exception as e:
                self.logger.error(f"Error fetching order book for {symbol}: {e}")
                return None
        else:
            self.logger.error("Client not initialized.")
            return None

    def get_historical_klines(self, symbol: str, interval: str, limit: int = 200):
        """
        Fetch historical kline data for a specific symbol and interval with caching.
        """
        cache_key = f"historical_klines:{symbol}:{interval}:{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.debug(f"Returning cached klines for {symbol} {interval}")
            return pd.DataFrame(cached_data)

        if self.client:
            try:
                klines = self.client.get_historical_klines(symbol=symbol, interval=interval, limit=limit)
                self.cache.set(cache_key, klines, ttl=self.cache_ttl)
                self.logger.info(f"Fetched and cached {len(klines)} klines for {symbol} {interval}.")
                return pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(6)])
            except Exception as e:
                self.logger.error(f"Error fetching historical klines for {symbol} {interval}: {e}")
                return pd.DataFrame()
        else:
            self.logger.error("Client not initialized.")
            return pd.DataFrame()
