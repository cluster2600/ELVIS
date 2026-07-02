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
    
    # FEWER, BIGGER TRADES STRATEGY
    'MAX_LOSS_PER_TRADE_USD': 50.0,     # Larger stop: $50.00 loss per trade  
    'PROFIT_TARGET_USD': 25.0,          # Bigger target: $25.00 profit per trade
    'MAX_DAILY_LOSS_USD': 150.0,        # Higher daily risk for bigger trades
    'MAX_RISK_PER_TRADE_USD': 75.0,     # Much larger positions: $75.00 risk per trade
    
    # SELECTIVE TRADING SETTINGS (FEWER TRADES)
    'MIN_SIGNAL_CONFIDENCE': 0.85,      # Higher threshold: 85% confidence for trades
    'HIGH_CONFIDENCE_THRESHOLD': 0.92,   # 92%+ for maximum sizing
    'CONFLUENCE_REQUIRED': 4,           # Stricter: 4/5 indicators must agree
    'TRADE_COOLDOWN_MINUTES': 15,       # 15 min cooldown between trades
    'MAX_DAILY_TRADES': 8,              # Fewer total trades per day
    
    # HIGH LEVERAGE for maximum trading power
    'LEVERAGE_MAX': 125,      # Maximum leverage for futures
    'LEVERAGE_MIN': 1,        # Minimum leverage for futures
    'DEFAULT_LEVERAGE': 100,  # Default leverage at 100x for maximum power
    
    'MAX_TRADES_PER_DAY': 10,           # Added to fix MAX_TRADES_PER_DAY error
    'DAILY_PROFIT_TARGET_USD': 30,     # Realistic $30 daily profit target
    'MIN_CAPITAL_USD': 1000,           # Added to fix MIN_CAPITAL_USD error
    'COOLDOWN': 5                      # 5 second cooldown between trades
}

# Paper Trading Configuration
PAPER_TRADING_CONFIG = {
    'INITIAL_USDT_BALANCE': 1000.0,  # Starting USDT balance
    'INITIAL_BNB_BALANCE': 1000.0,   # Starting BNB balance
    'ENABLE_MULTI_ASSET': True,      # Enable trading multiple assets
    'TRACK_PERFORMANCE': True,       # Track performance metrics
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




# BNB Trading and Fee Optimization Configuration
BNB_CONFIG = {
    'ENABLE_BNB_FEES': True,           # Use BNB to pay trading fees (10% discount on futures, 25% on spot)
    'BNB_TRADING_ENABLED': True,       # Allow trading BNB pairs
    'MIN_BNB_BALANCE': 0.1,           # Minimum BNB balance to maintain for fees
    'AUTO_BUY_BNB': True,             # Automatically buy BNB when balance is low
    'MAX_BNB_BUY_PERCENT': 5.0,       # Max % of portfolio to spend on BNB auto-buy
    'BNB_SYMBOLS': ['BNBUSDT', 'BNBBTC'],  # Available BNB trading pairs
    'BNB_REBALANCE_THRESHOLD': 0.05,  # Rebalance when BNB balance drops below this
}



# Multi-Asset Trading Configuration  
SYMBOLS_CONFIG = {
    'PRIMARY_SYMBOLS': ['BTCUSDT', 'BNBUSDT'],           # Primary trading pairs - REMOVED BNBBTC due to pricing issues
    'SECONDARY_SYMBOLS': ['ETHUSDT', 'ADAUSDT'],          # Secondary pairs (optional)
    'STABLE_PAIRS': ['BTCUSDT', 'ETHUSDT'],              # Stable, high-liquidity pairs
    'FEE_OPTIMIZATION_PAIRS': ['BNBUSDT'],               # Pairs for fee optimization - REMOVED BNBBTC
    'MAX_CONCURRENT_PAIRS': 2,                           # Maximum pairs to trade simultaneously - REDUCED
    'CROSS_ASSET_PAIRS': [],                             # REMOVED crypto-to-crypto pairs due to pricing issues
}

