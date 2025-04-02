"""
Configuration module for the ELVIS project.
This module centralizes all configuration parameters for the project.

ELVIS: Enhanced Leveraged Virtual Investment System
"""

# ASCII Art for ELVIS
ELVIS_ASCII = r"""
 _______ __     ____  __ ____ _____
|  ____| \ \   / /  \/  |___ \_   _|
| |__     \ \_/ /| \  / | __) || |
|  __|     \   / | |\/| ||__ < | |
| |____     | |  | |  | |___) || |_
|______|    |_|  |_|  |_|____/_____|

Enhanced Leveraged Virtual Investment System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_CONFIG = {
    'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
    'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET'),
    'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
}

# Trading Configuration
TRADING_CONFIG = {
    'SYMBOL': 'BTCUSDT',
    'TIMEFRAME': '1h',
    'LEVERAGE_MIN': 75,
    'LEVERAGE_MAX': 125,
    'STOP_LOSS_PCT': 0.01,
    'TAKE_PROFIT_PCT': 0.03,
    'DAILY_PROFIT_TARGET_USD': 1000.0,
    'DAILY_LOSS_LIMIT_USD': -500.0,
    'COOLDOWN': 3600.0,  # 1 hour
    'SLEEP_INTERVAL': 300.0,  # 5 minutes
    'MAX_TRADES_PER_DAY': 2,
    'MIN_CAPITAL_USD': 1000.0,
    'DATA_LIMIT': 200,  # Number of recent candles to fetch
    'PRODUCTION_MODE': False,  # Set to False for non-production mode
    'DEFAULT_MODE': 'paper'  # 'live', 'paper', or 'backtest'
}

# --- Configuration Validation ---
def validate_config():
    """Basic validation for critical config values."""
    # Leverage
    assert 1 <= TRADING_CONFIG['LEVERAGE_MIN'] <= 125, "LEVERAGE_MIN must be between 1 and 125"
    assert 1 <= TRADING_CONFIG['LEVERAGE_MAX'] <= 125, "LEVERAGE_MAX must be between 1 and 125"
    assert TRADING_CONFIG['LEVERAGE_MIN'] <= TRADING_CONFIG['LEVERAGE_MAX'], "LEVERAGE_MIN cannot be greater than LEVERAGE_MAX"

    # Percentages (Stop Loss / Take Profit)
    assert 0 < TRADING_CONFIG['STOP_LOSS_PCT'] < 1, "STOP_LOSS_PCT must be between 0 and 1 (exclusive)"
    assert 0 < TRADING_CONFIG['TAKE_PROFIT_PCT'] < 1, "TAKE_PROFIT_PCT must be between 0 and 1 (exclusive)"

    # Limits
    assert TRADING_CONFIG['DAILY_LOSS_LIMIT_USD'] <= 0, "DAILY_LOSS_LIMIT_USD should be zero or negative"
    assert TRADING_CONFIG['DAILY_PROFIT_TARGET_USD'] >= 0, "DAILY_PROFIT_TARGET_USD should be zero or positive"
    assert TRADING_CONFIG['MAX_TRADES_PER_DAY'] >= 0, "MAX_TRADES_PER_DAY cannot be negative"
    assert TRADING_CONFIG['MIN_CAPITAL_USD'] > 0, "MIN_CAPITAL_USD must be positive"

    # Intervals
    assert TRADING_CONFIG['COOLDOWN'] >= 0, "COOLDOWN cannot be negative"
    assert TRADING_CONFIG['SLEEP_INTERVAL'] > 0, "SLEEP_INTERVAL must be positive"
    assert TRADING_CONFIG['DATA_LIMIT'] > 10, "DATA_LIMIT should be reasonably large (e.g., > 10)"

    # Mode
    assert TRADING_CONFIG['DEFAULT_MODE'] in ['live', 'paper', 'backtest'], "Invalid DEFAULT_MODE"

# Run validation on import
validate_config()
# --- End Validation ---


# Technical Indicators
TECH_INDICATORS = ['rsi', 'macd', 'dx', 'obv']

# Backtesting Configuration
BACKTEST_CONFIG = {
    'TRADE_START_DATE': "2023-01-01",
    'TRADE_END_DATE': "2023-01-31",
    'PICKLE_RESULTS': [
        "res_2023-01-23__16_32_55_model_WF_ppo_5m_3H_20k",
        "res_2023-01-23__17_07_49_model_KCV_ppo_5m_3H_20005k",
        "res_2023-01-23__16_44_30_model_CPCV_ppo_5m_3H_20k"
    ]
}

# File Paths
FILE_PATHS = {
    'TRAIN_RESULTS_DIR': './train_results/',
    'DATA_DIR': './data/trade_data/',
    'SPY_INDEX_FILE': 'data/SPY_Crypto_Broad_Digital_Market_Index - Sheet1.csv',
    'METRICS_FILE': 'plots_and_metrics/test_metrics.txt',
    'PLOT_DIR': './plots_and_metrics/',
    'METRICS_DIR': './plots_and_metrics/',
    'DASHBOARD_DIR': './dashboard/'
}

# Logging Configuration
LOGGING_CONFIG = {
    'LOG_LEVEL': 'INFO',
    'LOG_TO_FILE': True
}
