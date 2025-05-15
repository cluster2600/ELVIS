"""
PriceFetcher - Real-time Binance price streaming and indicator calculation for ELVIS Trading Bot

Handles:
- Fetching historical candles from Binance REST API.
- Streaming real-time kline updates via WebSocket.
- Calculating technical indicators (RSI, MACD, SMA, EMA).
- Updating Prometheus metrics for monitoring.

Dependencies: binance, websocket-client, prometheus_client, threading, logging
"""

import logging
import threading
import time
import json

from binance.client import Client
import websocket

import pandas as pd
from prometheus_client import Gauge

# Prometheus metrics
CURRENT_PRICE = Gauge('elvis_current_price', 'Current BTC price')
RSI_GAUGE = Gauge('elvis_rsi', 'Relative Strength Index')
MACD_GAUGE = Gauge('elvis_macd', 'MACD line')
MACD_SIGNAL_GAUGE = Gauge('elvis_macd_signal', 'MACD signal line')
SMA_GAUGE = Gauge('elvis_sma', 'Simple Moving Average')
EMA_SHORT_GAUGE = Gauge('elvis_ema_short', 'Short-term Exponential Moving Average')
EMA_LONG_GAUGE = Gauge('elvis_ema_long', 'Long-term Exponential Moving Average')

class PriceFetcher:
    """
    Class to fetch and stream BTC price data and calculate technical indicators.
    """

    def __init__(self, logger, client=None, symbol='BTCUSDT', timeframe='5m', history_limit=200):
        self.logger = logger or logging.getLogger(__name__)
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.history_limit = history_limit
        self.ws = None
        self.candles = []
        self.running = False
        self.lock = threading.Lock()

    def get_historical_data(self):
        """
        Fetch historical kline/candle data from Binance REST API.
        """
        if self.client:
            try:
                klines = self.client.get_historical_klines(symbol=self.symbol, interval=self.timeframe, limit=self.history_limit)
                with self.lock:
                    self.candles = klines
                self.logger.info(f"Fetched {len(klines)} historical klines for {self.symbol}.")
            except Exception as e:
                self.logger.error(f"Error fetching historical klines: {e}")
        else:
            self.logger.error("Client not initialized.")

    def calculate_indicators(self):
        """
        Calculate and publish technical indicators from the latest candle data.
        """
        try:
            if len(self.candles) < 26:
                return  # Not enough data for EMA, MACD, etc.

            df = pd.DataFrame(self.candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(7)])
            df['close'] = df['close'].astype(float)

            close_prices = df['close']

            rsi = self.calculate_rsi(close_prices, window=14)
            macd, signal = self.calculate_macd(close_prices)
            sma = self.calculate_sma(close_prices, window=20)
            ema_short = self.calculate_ema(close_prices, window=9)
            ema_long = self.calculate_ema(close_prices, window=21)

            # Update Prometheus Gauges
            CURRENT_PRICE.set(close_prices.iloc[-1])
            RSI_GAUGE.set(rsi)
            MACD_GAUGE.set(macd)
            MACD_SIGNAL_GAUGE.set(signal)
            SMA_GAUGE.set(sma)
            EMA_SHORT_GAUGE.set(ema_short)
            EMA_LONG_GAUGE.set(ema_long)
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")

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
            data = json.loads(message)
            if data.get('e') == 'kline':
                kline = data['k']
                with self.lock:
                    self.candles.append([
                        kline['t'], kline['o'], kline['h'],
                        kline['l'], kline['c'], kline['v']
                    ])
                    if len(self.candles) > self.history_limit:
                        self.candles.pop(0)
                self.calculate_indicators()
                self.logger.info(f"Received kline: {kline}")
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
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{self.symbol.lower()}@kline_{self.timeframe}"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))

    def start(self):
        """
        Start the price fetcher: historical fetch + WebSocket stream.
        """
        self.get_historical_data()
        self.ws = websocket.WebSocketApp(
            f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.timeframe}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open
        self.running = True
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.logger.info("Price fetcher started.")

    def get_current_price(self):
        with self.lock:
            if self.candles:
                return float(self.candles[-1][4])  # Close price
            return None

    def get_current_candle(self):
        with self.lock:
            if self.candles:
                return self.candles[-1]
            return None

    def get_candle_history(self):
        with self.lock:
            return list(self.candles)
