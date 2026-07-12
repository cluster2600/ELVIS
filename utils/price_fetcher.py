"""
PriceFetcher - Real-time Binance price streaming and indicator calculation for ELVIS Trading Bot

Handles:
- Fetching historical candles from Binance REST API with Redis caching.
- Streaming real-time kline updates via WebSocket.
- Calculating technical indicators (RSI, MACD, SMA, EMA) with caching.
- Updating Prometheus metrics for monitoring.

Dependencies: binance, websocket-client, prometheus_client, threading, logging, redis
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta

try:
    from binance.error import ClientError
    from binance.um_futures import UMFutures

    FUTURES_AVAILABLE = True
except ImportError:
    FUTURES_AVAILABLE = False

import pandas as pd
import websocket
from binance.client import Client
from prometheus_client import Gauge

from utils.redis_cache import get_cache, make_indicator_key, make_price_key

# Prometheus metrics
CURRENT_PRICE = Gauge("elvis_current_price", "Current BTC price", ["symbol"])
RSI_GAUGE = Gauge("elvis_rsi", "Relative Strength Index", ["symbol"])
MACD_GAUGE = Gauge("elvis_macd", "MACD line", ["symbol"])
MACD_SIGNAL_GAUGE = Gauge("elvis_macd_signal", "MACD signal line", ["symbol"])
SMA_GAUGE = Gauge("elvis_sma", "Simple Moving Average", ["symbol"])
EMA_SHORT_GAUGE = Gauge(
    "elvis_ema_short", "Short-term Exponential Moving Average", ["symbol"]
)
EMA_LONG_GAUGE = Gauge(
    "elvis_ema_long", "Long-term Exponential Moving Average", ["symbol"]
)


class PriceFetcher:
    """
    Class to fetch and stream BTC price data and calculate technical indicators.
    """

    def __init__(
        self,
        logger,
        client=None,
        symbols=["BTCUSDT"],
        timeframe="5m",
        history_limit=200,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # Initialize client if not provided
        if client is None:
            try:
                from config.config import API_CONFIG

                # Use futures testnet keys first for better compatibility
                api_key = (
                    API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY
                    or API_CONFIG.BINANCE_API_KEY
                )
                api_secret = (
                    API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET
                    or API_CONFIG.BINANCE_API_SECRET
                )

                # Check if we have valid API keys (not placeholder values)
                if (
                    api_key
                    and api_secret
                    and api_key != "your_binance_api_key_here"
                    and api_secret != "your_binance_api_secret_here"
                ):

                    # Try to use futures connector first, fallback to spot
                    if FUTURES_AVAILABLE:
                        try:
                            self.client = UMFutures(
                                key=api_key, secret=api_secret, timeout=10
                            )
                            self.logger.info(
                                "Using authenticated Binance Futures client"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Futures client failed, using spot client: {e}"
                            )
                            self.client = Client(
                                api_key, api_secret, requests_params={"timeout": 10}
                            )
                            self.logger.info("Using authenticated Binance spot client")
                    else:
                        self.client = Client(
                            api_key, api_secret, requests_params={"timeout": 10}
                        )
                        self.logger.info("Using authenticated Binance spot client")
                else:
                    # Use public client for price data (no trading)
                    self.client = Client(requests_params={"timeout": 10})
                    self.logger.info(
                        "Using public Binance client for price data (no API keys)"
                    )
            except Exception as e:
                self.logger.warning(f"Could not initialize Binance client: {e}")
                self.logger.info(
                    "Continuing without Binance client - will use mock data"
                )
                self.client = None
        else:
            self.client = client

        self.symbols = symbols
        self.timeframe = timeframe
        self.history_limit = history_limit
        self.ws = None
        self.candles = {symbol: [] for symbol in symbols}
        self.running = False
        self.lock = threading.Lock()
        try:
            self.cache = get_cache()
        except Exception as e:
            self.logger.warning(f"Redis cache not available: {e}, using memory cache")
            self.cache = {}  # Simple dict as fallback
        self.cache_ttl = 10  # 10 seconds cache for prices (more live updates)
        self.indicator_cache_ttl = 5  # 5 seconds for indicators (very fresh)

        # Symbols that are only available on spot market
        self.spot_only_symbols = {"BNBBTC", "ETHBTC", "LTCBTC", "XRPBTC", "ADABTC"}

    def _is_spot_only_symbol(self, symbol):
        """Check if a symbol is only available on spot market"""
        return symbol in self.spot_only_symbols

    def get_historical_data(self):
        """
        Fetch historical kline/candle data from Binance REST API for all symbols.
        """
        if self.client:
            for symbol in self.symbols:
                try:
                    # Use spot client for symbols that are spot-only
                    if self._is_spot_only_symbol(symbol):
                        # Force use of spot client for spot-only symbols
                        if hasattr(self, "_spot_client"):
                            spot_client = self._spot_client
                        else:
                            # Create a temporary spot client
                            spot_client = Client(requests_params={"timeout": 10})
                            self._spot_client = spot_client
                        klines = spot_client.get_historical_klines(
                            symbol=symbol,
                            interval=self.timeframe,
                            limit=self.history_limit,
                        )
                    elif FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                        klines = self.client.klines(
                            symbol=symbol,
                            interval=self.timeframe,
                            limit=self.history_limit,
                        )
                    else:
                        klines = self.client.get_historical_klines(
                            symbol=symbol,
                            interval=self.timeframe,
                            limit=self.history_limit,
                        )
                    with self.lock:
                        self.candles[symbol] = klines
                    self.logger.info(
                        f"Fetched {len(klines)} historical klines for {symbol}."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error fetching historical klines for {symbol}: {e}"
                    )
                    self.logger.warning(f"⚠️ No data for {symbol}")
        else:
            self.logger.error("Client not initialized.")

    def calculate_indicators(self, symbol):
        """
        Calculate and publish technical indicators for a specific symbol from the latest candle data with caching.
        """
        try:
            if len(self.candles[symbol]) < 26:
                return  # Not enough data for EMA, MACD, etc.

            # Dynamically handle columns based on received data length
            candle_data = self.candles[symbol]
            num_columns = len(candle_data[0])

            base_columns = ["open_time", "open", "high", "low", "close", "volume"]
            extra_columns = [
                f"extra_{i}" for i in range(num_columns - len(base_columns))
            ]

            df = pd.DataFrame(candle_data, columns=base_columns + extra_columns)
            df["close"] = df["close"].astype(float)

            close_prices = df["close"]
            current_price = close_prices.iloc[-1]

            # Check cache for indicators
            indicators_cache_key = f"indicators:{symbol}:all"
            cached_indicators = self._cache_get(indicators_cache_key)

            if cached_indicators:
                # Use cached values
                rsi = cached_indicators.get("rsi")
                macd = cached_indicators.get("macd")
                signal = cached_indicators.get("signal")
                sma = cached_indicators.get("sma")
                ema_short = cached_indicators.get("ema_short")
                ema_long = cached_indicators.get("ema_long")
            else:
                # Calculate fresh
                rsi = self.calculate_rsi(close_prices, window=14)
                macd, signal = self.calculate_macd(close_prices)
                sma = self.calculate_sma(close_prices, window=20)
                ema_short = self.calculate_ema(close_prices, window=9)
                ema_long = self.calculate_ema(close_prices, window=21)

                # Cache the indicators
                indicators_data = {
                    "rsi": rsi,
                    "macd": macd,
                    "signal": signal,
                    "sma": sma,
                    "ema_short": ema_short,
                    "ema_long": ema_long,
                    "timestamp": time.time(),
                }
                self._cache_set(
                    indicators_cache_key, indicators_data, ttl=self.indicator_cache_ttl
                )

                # Also cache individual indicators for specific queries
                self._cache_set(
                    make_indicator_key(symbol, "rsi", 14),
                    rsi,
                    ttl=self.indicator_cache_ttl,
                )
                self._cache_set(
                    make_indicator_key(symbol, "sma", 20),
                    sma,
                    ttl=self.indicator_cache_ttl,
                )

            # Update Prometheus Gauges
            CURRENT_PRICE.labels(symbol).set(current_price)
            RSI_GAUGE.labels(symbol).set(rsi)
            MACD_GAUGE.labels(symbol).set(macd)
            MACD_SIGNAL_GAUGE.labels(symbol).set(signal)
            SMA_GAUGE.labels(symbol).set(sma)
            EMA_SHORT_GAUGE.labels(symbol).set(ema_short)
            EMA_LONG_GAUGE.labels(symbol).set(ema_long)

            # Cache current price
            self._cache_set(make_price_key(symbol), current_price, ttl=self.cache_ttl)

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
            message_data = json.loads(message)
            if "data" not in message_data:
                self.logger.debug(f"Ignoring message without 'data' key: {message}")
                return

            data = message_data["data"]
            if data.get("e") == "kline":
                kline = data["k"]
                symbol = kline["s"]
                with self.lock:
                    self.candles[symbol].append(
                        [
                            kline["t"],
                            kline["o"],
                            kline["h"],
                            kline["l"],
                            kline["c"],
                            kline["v"],
                        ]
                    )
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
        subscribe_message = {"method": "SUBSCRIBE", "params": params, "id": 1}
        ws.send(json.dumps(subscribe_message))

    def start(self):
        """
        Start the price fetcher: historical fetch + WebSocket stream.
        """
        self.get_historical_data()

        streams = "/".join(
            [f"{symbol.lower()}@kline_{self.timeframe}" for symbol in self.symbols]
        )
        self.ws = websocket.WebSocketApp(
            f"wss://stream.binance.com:9443/stream?streams={streams}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.on_open = self.on_open
        self.running = True
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.logger.info("Price fetcher started.")

    def get_current_price(self, symbol):
        """Get current price with Redis caching"""
        # Try to get from cache first
        cache_key = make_price_key(symbol)
        cached_price = self._cache_get(cache_key)

        if cached_price is not None:
            return cached_price

        # If not in cache, get from API
        if self.client:
            try:
                if self._is_spot_only_symbol(symbol):
                    # Force use of spot client for spot-only symbols
                    if hasattr(self, "_spot_client"):
                        spot_client = self._spot_client
                    else:
                        spot_client = Client(requests_params={"timeout": 10})
                        self._spot_client = spot_client
                    ticker = spot_client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker["price"])
                elif FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                    ticker = self.client.ticker_price(symbol)
                    price = float(ticker["price"])
                else:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker["price"])

                self._cache_set(cache_key, price, ttl=self.cache_ttl)
                return price
            except Exception as e:
                self.logger.error(f"Error fetching ticker for {symbol}: {e}")

        # Fallback to last candle close price
        with self.lock:
            if self.candles.get(symbol):
                price = float(self.candles[symbol][-1][4])  # Close price
                # Cache the price
                self._cache_set(cache_key, price, ttl=self.cache_ttl)
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
        cached_data = self._cache_get(cache_key)
        if cached_data:
            self.logger.debug(f"Returning cached order book for {symbol}")
            return cached_data

        if self.client:
            try:
                if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                    # Use futures depth endpoint
                    order_book = self.client.depth(symbol=symbol, limit=limit)
                else:
                    # Use spot order book endpoint
                    order_book = self.client.depth(symbol=symbol, limit=limit)
                self._cache_set(
                    cache_key, order_book, ttl=self.indicator_cache_ttl
                )  # Use indicator TTL
                self.logger.debug(f"Fetched and cached order book for {symbol}.")
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
        cached_data = self._cache_get(cache_key)
        if cached_data:
            self.logger.debug(f"Returning cached klines for {symbol} {interval}")
            # Ensure cached data has proper column structure
            df = pd.DataFrame(cached_data)
            if not df.empty and "close" not in df.columns:
                # Reconstruct with proper column names if needed
                if len(df.columns) >= 6:
                    df.columns = [
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ][: len(df.columns)]
            return df

        if self.client:
            try:
                if self._is_spot_only_symbol(symbol):
                    # Force use of spot client for spot-only symbols
                    if hasattr(self, "_spot_client"):
                        spot_client = self._spot_client
                    else:
                        spot_client = Client(requests_params={"timeout": 10})
                        self._spot_client = spot_client
                    klines = spot_client.get_historical_klines(
                        symbol=symbol, interval=interval, limit=limit
                    )
                elif FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                    # Use futures klines endpoint
                    klines = self.client.klines(
                        symbol=symbol, interval=interval, limit=limit
                    )
                else:
                    # Use spot klines endpoint
                    klines = self.client.get_historical_klines(
                        symbol=symbol, interval=interval, limit=limit
                    )
                self._cache_set(cache_key, klines, ttl=self.cache_ttl)
                self.logger.debug(
                    f"Fetched and cached {len(klines)} klines for {symbol} {interval}."
                )

                # Create DataFrame with proper column names
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ],
                )

                # Convert to proper data types
                numeric_columns = ["open", "high", "low", "close", "volume"]
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Convert timestamps
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

                return df
            except Exception as e:
                self.logger.error(
                    f"Error fetching historical klines for {symbol} {interval}: {e}"
                )
                # Return empty DataFrame with proper columns for consistency
                return pd.DataFrame(
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ]
                )
        else:
            self.logger.error("Client not initialized.")
            # Return empty DataFrame with proper columns for consistency
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ]
            )

    def _cache_get(self, key):
        """Get from cache with fallback for dict cache"""
        if hasattr(self.cache, "get"):
            return self.cache.get(key)
        else:
            # Simple dict fallback with TTL
            if key in self.cache:
                value, expiry = self.cache[key]
                if expiry is None or time.time() < expiry:
                    return value
                else:
                    del self.cache[key]
            return None

    def _cache_set(self, key, value, ttl=None):
        """Set in cache with fallback for dict cache"""
        if hasattr(self.cache, "set"):
            return self.cache.set(key, value, ttl=ttl)
        else:
            # Simple dict fallback with TTL
            self.cache[key] = (value, time.time() + ttl if ttl else None)
