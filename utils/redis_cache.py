"""
Redis cache utility for ELVIS Trading Bot
Provides caching functionality for market data and other frequently accessed data
"""

import json
import logging
import os
from typing import Any, Optional, Union
import redis
from redis.exceptions import ConnectionError, TimeoutError
from .secrets_manager import get_enhanced_secrets_manager

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache wrapper with automatic serialization/deserialization"""
    
    def __init__(self, host: str = None, port: int = None, db: int = None, 
                 password: str = None, decode_responses: bool = True):
        """
        Initialize Redis cache connection
        
        Args:
            host: Redis host (defaults to env var or localhost)
            port: Redis port (defaults to env var or 6379)
            db: Redis database number (defaults to env var or 0)
            password: Redis password (optional)
            decode_responses: Whether to decode responses to strings
        """
        # Use enhanced secrets manager for Redis credentials
        secrets_manager = get_enhanced_secrets_manager()
        redis_creds = secrets_manager.get_redis_credentials()
        
        self.host = host or redis_creds.get('host', 'localhost')
        self.port = int(port or redis_creds.get('port', 6379))
        self.db = int(db or redis_creds.get('db', 0))
        self.password = password or redis_creds.get('password')
        
        try:
            # Only include password if it's actually provided
            redis_params = {
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'decode_responses': decode_responses,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True
            }
            
            if self.password:  # Only add password if it's not None/empty
                redis_params['password'] = self.password
                logger.debug(f"Connecting to Redis with authentication at {self.host}:{self.port}")
            else:
                logger.debug(f"Connecting to Redis without authentication at {self.host}:{self.port}")
            
            self.redis_client = redis.Redis(**redis_params)
            
            # Test connection
            self.redis_client.ping()
            auth_info = "with authentication" if self.password else "without authentication"
            logger.info(f"Connected to Redis {auth_info} at {self.host}:{self.port}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache will be disabled.")
            self.redis_client = None
    
    def _is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self._is_connected():
            return None
            
        try:
            value = self.redis_client.get(key)
            if value:
                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 5 minutes)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected():
            return False
            
        try:
            # Serialize to JSON if not string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            return self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._is_connected():
            return False
            
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if not self._is_connected():
            return False
            
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Pattern to match (e.g., "price:*")
            
        Returns:
            Number of keys deleted
        """
        if not self._is_connected():
            return 0
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self._is_connected():
            return -2
            
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -2


# Global cache instance
_cache_instance = None


def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


# Cache key generators
def make_price_key(symbol: str, interval: str = None) -> str:
    """Generate cache key for price data"""
    if interval:
        return f"price:{symbol}:{interval}"
    return f"price:{symbol}"


def make_orderbook_key(symbol: str, depth: int = 20) -> str:
    """Generate cache key for order book data"""
    return f"orderbook:{symbol}:{depth}"


def make_indicator_key(symbol: str, indicator: str, period: int) -> str:
    """Generate cache key for technical indicators"""
    return f"indicator:{symbol}:{indicator}:{period}"


def make_model_prediction_key(model_name: str, symbol: str) -> str:
    """Generate cache key for model predictions"""
    return f"prediction:{model_name}:{symbol}"
