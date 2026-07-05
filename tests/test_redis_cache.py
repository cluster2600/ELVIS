"""
Tests for Redis cache utility
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import redis

from utils.redis_cache import (
    RedisCache,
    get_cache,
    make_indicator_key,
    make_model_prediction_key,
    make_orderbook_key,
    make_price_key,
)


class TestRedisCache:
    """Test RedisCache class functionality"""

    @patch("redis.Redis")
    def test_redis_cache_initialization_success(self, mock_redis_class):
        """Test successful Redis cache initialization"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(host="localhost", port=6379)

        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.redis_client is not None
        mock_redis.ping.assert_called_once()

    @patch("redis.Redis")
    def test_redis_cache_initialization_failure(self, mock_redis_class):
        """Test Redis cache initialization when connection fails"""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis.ConnectionError()
        mock_redis_class.return_value = mock_redis

        cache = RedisCache(host="localhost", port=6379)

        assert cache.redis_client is None

    def test_get_when_disconnected(self):
        """Test get operation when Redis is disconnected"""
        cache = RedisCache()
        cache.redis_client = None

        result = cache.get("test_key")
        assert result is None

    def test_set_when_disconnected(self):
        """Test set operation when Redis is disconnected"""
        cache = RedisCache()
        cache.redis_client = None

        result = cache.set("test_key", "test_value")
        assert result is False

    @patch("redis.Redis")
    def test_get_with_json_value(self, mock_redis_class):
        """Test getting JSON serialized value from cache"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        test_data = {"price": 50000, "volume": 100}
        mock_redis.get.return_value = json.dumps(test_data)
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.get("test_key")

        assert result == test_data
        mock_redis.get.assert_called_with("test_key")

    @patch("redis.Redis")
    def test_get_with_string_value(self, mock_redis_class):
        """Test getting string value from cache"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = "simple_string"
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.get("test_key")

        assert result == "simple_string"

    @patch("redis.Redis")
    def test_set_with_dict_value(self, mock_redis_class):
        """Test setting dictionary value in cache"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        test_data = {"price": 50000, "volume": 100}
        result = cache.set("test_key", test_data, ttl=60)

        assert result is True
        expected_value = json.dumps(test_data)
        mock_redis.setex.assert_called_with("test_key", 60, expected_value)

    @patch("redis.Redis")
    def test_delete_existing_key(self, mock_redis_class):
        """Test deleting existing key from cache"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.delete("test_key")

        assert result is True
        mock_redis.delete.assert_called_with("test_key")

    @patch("redis.Redis")
    def test_exists_key(self, mock_redis_class):
        """Test checking if key exists in cache"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.exists.return_value = 1
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.exists("test_key")

        assert result is True
        mock_redis.exists.assert_called_with("test_key")

    @patch("redis.Redis")
    def test_clear_pattern(self, mock_redis_class):
        """Test clearing keys by pattern"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = ["price:BTC:1m", "price:BTC:5m"]
        mock_redis.delete.return_value = 2
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.clear_pattern("price:*")

        assert result == 2
        mock_redis.keys.assert_called_with("price:*")
        mock_redis.delete.assert_called_with("price:BTC:1m", "price:BTC:5m")

    @patch("redis.Redis")
    def test_get_ttl(self, mock_redis_class):
        """Test getting TTL for a key"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.ttl.return_value = 120
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.get_ttl("test_key")

        assert result == 120
        mock_redis.ttl.assert_called_with("test_key")

    @patch("redis.Redis")
    def test_error_handling_in_get(self, mock_redis_class):
        """Test error handling in get operation"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.side_effect = Exception("Redis error")
        mock_redis_class.return_value = mock_redis

        cache = RedisCache()
        result = cache.get("test_key")

        assert result is None


class TestCacheKeyGenerators:
    """Test cache key generation functions"""

    def test_make_price_key(self):
        """Test price cache key generation"""
        assert make_price_key("BTCUSDT") == "price:BTCUSDT"
        assert make_price_key("BTCUSDT", "5m") == "price:BTCUSDT:5m"

    def test_make_orderbook_key(self):
        """Test order book cache key generation"""
        assert make_orderbook_key("BTCUSDT") == "orderbook:BTCUSDT:20"
        assert make_orderbook_key("BTCUSDT", 50) == "orderbook:BTCUSDT:50"

    def test_make_indicator_key(self):
        """Test indicator cache key generation"""
        key = make_indicator_key("BTCUSDT", "RSI", 14)
        assert key == "indicator:BTCUSDT:RSI:14"

    def test_make_model_prediction_key(self):
        """Test model prediction cache key generation"""
        key = make_model_prediction_key("random_forest", "BTCUSDT")
        assert key == "prediction:random_forest:BTCUSDT"


class TestGlobalCacheInstance:
    """Test global cache instance management"""

    @patch("utils.redis_cache.RedisCache")
    def test_get_cache_singleton(self, mock_redis_cache_class):
        """Test that get_cache returns singleton instance"""
        # Reset global instance
        import utils.redis_cache

        utils.redis_cache._cache_instance = None

        mock_instance = MagicMock()
        mock_redis_cache_class.return_value = mock_instance

        # First call should create instance
        cache1 = get_cache()
        assert cache1 == mock_instance
        mock_redis_cache_class.assert_called_once()

        # Second call should return same instance
        cache2 = get_cache()
        assert cache2 == cache1
        assert mock_redis_cache_class.call_count == 1
