# Configuration file for ELVIS trading bot

import os
from utils.secrets_manager import get_enhanced_secrets_manager

class APIConfig:
    def __init__(self):
        self._secrets = get_enhanced_secrets_manager()
    
    @property
    def BINANCE_API_KEY(self):
        return self._secrets.get_secret('BINANCE_API_KEY', 'api_keys') or os.getenv('BINANCE_API_KEY', 'your_binance_api_key_here')

    @property
    def BINANCE_API_SECRET(self):
        return self._secrets.get_secret('BINANCE_API_SECRET', 'api_keys') or os.getenv('BINANCE_API_SECRET', 'your_binance_api_secret_here')
    
    @property
    def BINANCE_FUTURES_TESTNET_API_KEY(self):
        return self._secrets.get_secret('BINANCE_FUTURES_TESTNET_API_KEY', 'api_keys') or os.getenv('BINANCE_FUTURES_TESTNET_API_KEY', 'your_futures_testnet_api_key_here')

    @property
    def BINANCE_FUTURES_TESTNET_API_SECRET(self):
        return self._secrets.get_secret('BINANCE_FUTURES_TESTNET_API_SECRET', 'api_keys') or os.getenv('BINANCE_FUTURES_TESTNET_API_SECRET', 'your_futures_testnet_api_secret_here')

API_CONFIG = APIConfig()

TRADING_CONFIG = {
    'DEFAULT_MODE': 'futures_testnet',  # Changed to use futures testnet by default
    'SYMBOL': 'BTCUSDT',
    'DATA_LIMIT': 200,
    'MAX_POSITION_SIZE': 0.1,  # Example value
    'MAX_DAILY_TRADES': 5,
    'MAX_DAILY_LOSS': 0.05,
    'MAX_DRAWDOWN': 0.1,
    'RISK_PER_TRADE': 0.02,
    'STOP_LOSS_PCT': 0.02,  # Added to fix the error; adjust as needed
    'TAKE_PROFIT_PCT': 0.02,  # Added to fix the current error; adjust as needed
    'LEVERAGE_MAX': 125,      # Maximum leverage for futures
    'LEVERAGE_MIN': 1,        # Minimum leverage for futures
    'DEFAULT_LEVERAGE': 100,  # Default leverage for maximum trading power
    'MAX_TRADES_PER_DAY': 10,  # Added to fix MAX_TRADES_PER_DAY error
    'DAILY_PROFIT_TARGET_USD': 100,  # Added to fix DAILY_PROFIT_TARGET_USD error
    'DAILY_LOSS_LIMIT_USD': 100,     # Added to fix DAILY_LOSS_LIMIT_USD error
    'MIN_CAPITAL_USD': 1000,         # Added to fix MIN_CAPITAL_USD error
    'COOLDOWN': 0                    # No cooldown - maximum trading speed
}

LOGGING_CONFIG = {
    'LOG_LEVEL': 'DEBUG',
    'LOG_TO_FILE': True,
}

POSTGRES_CONFIG = {
    'HOST': 'localhost',
    'PORT': 5432,
    'USER': 'elvis_user',
    'PASSWORD': 'elvis_password',
    'DBNAME': 'elvis_trading'
}
