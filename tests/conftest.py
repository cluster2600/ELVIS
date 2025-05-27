"""
Pytest configuration and fixtures for ELVIS Trading Bot tests
"""

import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Disable logging during tests unless explicitly needed
logging.disable(logging.CRITICAL)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing"""
    return MagicMock()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    prices = 50000 + np.random.randn(100).cumsum() * 100
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 10,
        'high': prices + abs(np.random.randn(100)) * 20,
        'low': prices - abs(np.random.randn(100)) * 20,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    })


@pytest.fixture
def sample_candles():
    """Generate sample candle data in Binance format"""
    base_time = int(datetime.now().timestamp() * 1000)
    candles = []
    
    for i in range(200):
        open_price = 50000 + np.random.randn() * 100
        close_price = open_price + np.random.randn() * 50
        high_price = max(open_price, close_price) + abs(np.random.randn()) * 20
        low_price = min(open_price, close_price) - abs(np.random.randn()) * 20
        
        candle = [
            base_time + i * 300000,  # 5 min intervals
            str(open_price),
            str(high_price),
            str(low_price),
            str(close_price),
            str(np.random.randint(100, 1000)),  # volume
            base_time + (i + 1) * 300000 - 1,  # close time
            str(np.random.uniform(1000000, 5000000)),  # quote asset volume
            np.random.randint(100, 500),  # number of trades
            str(np.random.uniform(100, 500)),  # taker buy base asset volume
            str(np.random.uniform(50000, 250000)),  # taker buy quote asset volume
            "0"  # ignore
        ]
        candles.append(candle)
    
    return candles


@pytest.fixture
def mock_binance_client():
    """Mock Binance client for testing"""
    client = MagicMock()
    
    # Mock account data
    client.get_account.return_value = {
        'balances': [
            {'asset': 'USDT', 'free': '10000.0', 'locked': '0.0'},
            {'asset': 'BTC', 'free': '0.5', 'locked': '0.0'}
        ]
    }
    
    # Mock ticker price
    client.get_symbol_ticker.return_value = {'symbol': 'BTCUSDT', 'price': '50000.0'}
    
    # Mock order book
    client.get_order_book.return_value = {
        'bids': [['49900.0', '0.5'], ['49800.0', '1.0']],
        'asks': [['50100.0', '0.5'], ['50200.0', '1.0']]
    }
    
    return client


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing"""
    cache = MagicMock()
    cache_data = {}
    
    def get_side_effect(key):
        return cache_data.get(key)
    
    def set_side_effect(key, value, ttl=None):
        cache_data[key] = value
        return True
    
    def delete_side_effect(key):
        if key in cache_data:
            del cache_data[key]
            return True
        return False
    
    cache.get.side_effect = get_side_effect
    cache.set.side_effect = set_side_effect
    cache.delete.side_effect = delete_side_effect
    cache.exists.side_effect = lambda key: key in cache_data
    
    return cache


@pytest.fixture
def trading_config():
    """Sample trading configuration"""
    return {
        'SYMBOL': 'BTCUSDT',
        'MAX_POSITION_SIZE': 0.1,
        'MAX_DAILY_TRADES': 5,
        'RISK_PER_TRADE': 0.02,
        'STOP_LOSS_PCT': 0.02,
        'TAKE_PROFIT_PCT': 0.05,
        'STARTING_BALANCE': 10000.0
    }


@pytest.fixture
def mock_model():
    """Mock machine learning model"""
    model = MagicMock()
    model.predict.return_value = np.array([0.7])  # Bullish prediction
    model.get_params.return_value = {'n_estimators': 100, 'max_depth': 10}
    return model


# Re-enable logging after all tests
def pytest_sessionfinish(session, exitstatus):
    """Re-enable logging after test session"""
    logging.disable(logging.NOTSET)
