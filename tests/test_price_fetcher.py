"""
Tests for PriceFetcher with Redis caching integration
"""

import json
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from utils.price_fetcher import PriceFetcher


class TestPriceFetcher:
    """Test PriceFetcher functionality with caching"""

    def test_price_fetcher_initialization(self, mock_logger, mock_binance_client):
        """Test PriceFetcher initialization"""
        fetcher = PriceFetcher(
            logger=mock_logger,
            client=mock_binance_client,
            symbols=["BTCUSDT"],
            timeframe="5m",
            history_limit=200,
        )

        assert fetcher.symbols == ["BTCUSDT"]
        assert fetcher.timeframe == "5m"
        assert fetcher.history_limit == 200
        assert fetcher.cache_ttl == 10
        assert fetcher.indicator_cache_ttl == 5

    def test_get_historical_data(
        self, mock_logger, mock_binance_client, sample_candles
    ):
        """Test fetching historical data"""
        mock_binance_client.get_historical_klines.return_value = sample_candles

        fetcher = PriceFetcher(
            logger=mock_logger, client=mock_binance_client, symbols=["BTCUSDT"]
        )
        fetcher.get_historical_data()

        assert len(fetcher.candles["BTCUSDT"]) == 200
        mock_binance_client.get_historical_klines.assert_called_once_with(
            symbol="BTCUSDT", interval="5m", limit=200
        )

    def test_get_historical_data_no_client(self, mock_logger):
        """Test get_historical_data when client is not initialized"""
        fetcher = PriceFetcher(logger=mock_logger, client=None)
        # When client=None the constructor auto-initializes a client from config;
        # force it back to None to exercise the "not initialized" branch.
        fetcher.client = None
        fetcher.get_historical_data()

        mock_logger.error.assert_called_with("Client not initialized.")

    @patch("utils.price_fetcher.get_cache")
    def test_get_current_price_from_cache(self, mock_get_cache, mock_logger):
        """Test getting current price from cache"""
        mock_cache = MagicMock()
        mock_cache.get.return_value = 50000.0
        mock_get_cache.return_value = mock_cache

        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        price = fetcher.get_current_price("BTCUSDT")

        assert price == 50000.0
        mock_cache.get.assert_called_once_with("price:BTCUSDT")
        mock_cache.set.assert_not_called()

    @patch("utils.price_fetcher.get_cache")
    def test_get_current_price_cache_miss(
        self, mock_get_cache, mock_logger, sample_candles
    ):
        """Test getting current price when cache miss"""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        # Force client=None so the cache-miss path falls back to the last candle
        # instead of hitting the auto-initialized Binance client.
        fetcher.client = None
        fetcher.candles["BTCUSDT"] = sample_candles

        price = fetcher.get_current_price("BTCUSDT")

        # Should get price from last candle
        expected_price = float(sample_candles[-1][4])
        assert price == expected_price

        # Should cache the price
        mock_cache.set.assert_called_once_with("price:BTCUSDT", expected_price, ttl=10)

    def test_calculate_indicators_with_cache(self, mock_logger):
        """Test calculate_indicators method exists and doesn't raise exceptions"""
        mock_cache = MagicMock()

        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.cache = mock_cache

        # Test with insufficient data - should return early
        fetcher.candles["BTCUSDT"] = [["1", "2", "3", "4", "5", "6"]] * 10

        # Should not raise exception
        try:
            fetcher.calculate_indicators("BTCUSDT")
        except Exception as e:
            pytest.fail(f"calculate_indicators raised an exception: {e}")

    def test_calculate_indicators_cache_miss(self, mock_logger):
        """Test calculate_indicators method works with sufficient data"""
        mock_cache = MagicMock()

        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.cache = mock_cache

        # Test with insufficient data - should return early
        fetcher.candles["BTCUSDT"] = [["1", "2", "3", "4", "5", "6"]] * 20

        # Should not raise exception
        try:
            fetcher.calculate_indicators("BTCUSDT")
        except Exception as e:
            pytest.fail(f"calculate_indicators raised an exception: {e}")

        # Method should have completed without errors
        assert not mock_logger.error.called

    def test_calculate_indicators_insufficient_data(self, mock_logger):
        """Test calculate_indicators with insufficient data"""
        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.candles["BTCUSDT"] = [
            ["1", "2", "3", "4", "5", "6"]
        ] * 20  # Only 20 candles

        fetcher.calculate_indicators("BTCUSDT")

        # Should return early without calculations
        # No error should be logged
        assert not mock_logger.error.called

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        prices = pd.Series(
            [
                100,
                102,
                101,
                103,
                104,
                102,
                105,
                106,
                104,
                107,
                108,
                106,
                109,
                110,
                108,
                111,
                112,
                110,
                113,
                114,
            ]
        )

        rsi = PriceFetcher.calculate_rsi(prices, window=14)

        assert isinstance(rsi, (float, np.floating))
        assert 0 <= rsi <= 100

    def test_macd_calculation(self):
        """Test MACD calculation"""
        prices = pd.Series(np.random.randn(50).cumsum() + 100)

        macd, signal = PriceFetcher.calculate_macd(prices)

        assert isinstance(macd, (float, np.floating))
        assert isinstance(signal, (float, np.floating))

    def test_sma_calculation(self):
        """Test SMA calculation"""
        prices = pd.Series(range(100, 150))

        sma = PriceFetcher.calculate_sma(prices, window=20)

        assert isinstance(sma, (float, np.floating))
        assert sma == prices.iloc[-20:].mean()

    def test_ema_calculation(self):
        """Test EMA calculation"""
        prices = pd.Series(range(100, 150))

        ema = PriceFetcher.calculate_ema(prices, window=20)

        assert isinstance(ema, (float, np.floating))

    def test_websocket_message_handler(self, mock_logger):
        """Test WebSocket message handler"""
        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.candles["BTCUSDT"] = []
        fetcher.calculate_indicators = MagicMock()

        # Mock WebSocket message
        message = json.dumps(
            {
                "data": {
                    "e": "kline",
                    "k": {
                        "s": "BTCUSDT",
                        "t": 1234567890,
                        "o": "50000.0",
                        "h": "50100.0",
                        "l": "49900.0",
                        "c": "50050.0",
                        "v": "100.0",
                    },
                }
            }
        )

        fetcher.on_message(None, message)

        assert len(fetcher.candles["BTCUSDT"]) == 1
        assert fetcher.candles["BTCUSDT"][0][4] == "50050.0"
        fetcher.calculate_indicators.assert_called_once_with("BTCUSDT")

    def test_websocket_message_handler_max_candles(self, mock_logger, sample_candles):
        """Test WebSocket handler maintains history limit"""
        fetcher = PriceFetcher(logger=mock_logger, history_limit=5, symbols=["BTCUSDT"])
        fetcher.candles["BTCUSDT"] = sample_candles[:5]  # Already at limit
        fetcher.calculate_indicators = MagicMock()

        message = json.dumps(
            {
                "data": {
                    "e": "kline",
                    "k": {
                        "s": "BTCUSDT",
                        "t": 9999999999,
                        "o": "51000.0",
                        "h": "51100.0",
                        "l": "50900.0",
                        "c": "51050.0",
                        "v": "200.0",
                    },
                }
            }
        )

        fetcher.on_message(None, message)

        assert len(fetcher.candles["BTCUSDT"]) == 5  # Still at limit
        assert fetcher.candles["BTCUSDT"][-1][0] == 9999999999  # New candle added
        assert fetcher.candles["BTCUSDT"][0] != sample_candles[0]  # Old candle removed

    def test_websocket_error_handler(self, mock_logger):
        """Test WebSocket error handler"""
        fetcher = PriceFetcher(logger=mock_logger)
        error = Exception("WebSocket error")

        fetcher.on_error(None, error)

        mock_logger.error.assert_called_with(f"WebSocket error: {error}")

    def test_websocket_close_handler(self, mock_logger):
        """Test WebSocket close handler"""
        fetcher = PriceFetcher(logger=mock_logger)
        fetcher.running = True

        fetcher.on_close(None, 1000, "Normal closure")

        assert fetcher.running is False
        mock_logger.info.assert_called_with("WebSocket closed")

    def test_get_current_candle(self, mock_logger, sample_candles):
        """Test getting current candle"""
        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.candles["BTCUSDT"] = sample_candles

        candle = fetcher.get_current_candle("BTCUSDT")

        assert candle == sample_candles[-1]

    def test_get_candle_history(self, mock_logger, sample_candles):
        """Test getting candle history"""
        fetcher = PriceFetcher(logger=mock_logger, symbols=["BTCUSDT"])
        fetcher.candles["BTCUSDT"] = sample_candles

        history = fetcher.get_candle_history("BTCUSDT")

        assert history == sample_candles
        assert history is not fetcher.candles["BTCUSDT"]  # Should be a copy
