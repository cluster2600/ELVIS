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
    # Issue #14: Default leverage reduced from 100x to 3x to prevent catastrophic
    # losses on startup.  Override via DEFAULT_LEVERAGE env var (integer).
    'DEFAULT_LEVERAGE': int(os.getenv('DEFAULT_LEVERAGE', '3')),
    'MAX_TRADES_PER_DAY': 10,  # Added to fix MAX_TRADES_PER_DAY error
    'DAILY_PROFIT_TARGET_USD': 100,  # Added to fix DAILY_PROFIT_TARGET_USD error
    'DAILY_LOSS_LIMIT_USD': 100,     # Added to fix DAILY_LOSS_LIMIT_USD error
    'MIN_CAPITAL_USD': 1000,         # Added to fix MIN_CAPITAL_USD error
    'COOLDOWN': 0                    # No cooldown - maximum trading speed
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
    'HOST': os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'localhost')),
    'PORT': int(os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))),
    'USER': os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'elvis_user')),
    'PASSWORD': os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASSWORD')),
    'DBNAME': os.getenv('POSTGRES_DBNAME', os.getenv('DB_NAME', 'elvis_trading')),
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
    'PRIMARY_SYMBOLS': ['BTCUSDT', 'BNBUSDT'],     # Primary trading pairs
    'SECONDARY_SYMBOLS': ['ETHUSDT', 'ADAUSDT'],    # Secondary pairs (optional)
    'STABLE_PAIRS': ['BTCUSDT', 'ETHUSDT'],        # Stable, high-liquidity pairs
    'FEE_OPTIMIZATION_PAIRS': ['BNBUSDT'],         # Pairs for fee optimization
    'MAX_CONCURRENT_PAIRS': 3,                     # Maximum pairs to trade simultaneously
}



# ---------------------------------------------------------------------------
# Issue #14: Leverage safety validation
# Call validate_leverage_config() before starting the trading engine.
# ---------------------------------------------------------------------------
import logging as _logging

_leverage_logger = _logging.getLogger(__name__)

def validate_leverage_config(leverage: int = None) -> int:
    """
    Validate the configured leverage and refuse to start if it is dangerously
    high without an explicit operator override.

    Rules:
      - leverage > 10x requires OVERRIDE_HIGH_LEVERAGE=true in the environment.
      - leverage > 5x emits a WARNING log on every startup.
      - leverage <= 0 raises ValueError immediately.

    Args:
        leverage: The leverage value to validate.  Defaults to
                  TRADING_CONFIG['DEFAULT_LEVERAGE'].

    Returns:
        The validated leverage value (int).

    Raises:
        ValueError:       If leverage <= 0 or an invalid value is supplied.
        EnvironmentError: If leverage > 10x and OVERRIDE_HIGH_LEVERAGE != 'true'.
    """
    if leverage is None:
        leverage = TRADING_CONFIG['DEFAULT_LEVERAGE']

    leverage = int(leverage)

    if leverage <= 0:
        raise ValueError(f"Leverage must be a positive integer, got {leverage}.")

    if leverage > 5:
        _leverage_logger.warning(
            "⚠️  Leverage is set to %dx which is above the recommended maximum of 5x. "
            "High leverage significantly increases liquidation risk.",
            leverage,
        )

    if leverage > 10:
        override = os.getenv("OVERRIDE_HIGH_LEVERAGE", "false").strip().lower()
        if override != "true":
            raise EnvironmentError(
                f"Leverage {leverage}x exceeds the 10x safety limit. "
                "Set OVERRIDE_HIGH_LEVERAGE=true in your environment to acknowledge "
                "the risk and allow startup."
            )
        _leverage_logger.warning(
            "🚨 OVERRIDE_HIGH_LEVERAGE=true detected — starting with %dx leverage. "
            "Ensure you understand the liquidation risks.",
            leverage,
        )

    return leverage
