"""
Configuration for BTC_BOT
"""

# Trading Configuration
TRADING_CONFIG = {
    'STOP_LOSS_PCT': 0.02,        # 2% stop loss
    'TAKE_PROFIT_PCT': 0.03,      # 3% take profit
    'LEVERAGE_MAX': 125,
    'LEVERAGE_MIN': 1,
    'MAX_TRADES_PER_DAY': 20,
    'DAILY_PROFIT_TARGET_USD': 100.0,
    'DAILY_LOSS_LIMIT_USD': -50.0,
    'MIN_CAPITAL_USD': 50.0,
    'COOLDOWN': 10,               # 10 seconds cooldown between trades
}
# API Configuration
API_CONFIG = {
    'NEWS_API_KEY': 'YOUR_NEWS_API_KEY',
    'TWITTER_API_KEY': 'YOUR_TWITTER_API_KEY',
    'API_KEY': 'demo_api_key',
    'API_SECRET': 'demo_api_secret',
    'BINANCE_API_KEY': 'demo_api_key',
    'BINANCE_API_SECRET': 'demo_api_secret',
}
# Telegram Configuration
TELEGRAM_CONFIG = {
    'ENABLED': True,                    # True to send Telegram messages
    'BOT_TOKEN': 'YOUR_TELEGRAM_BOT_TOKEN',
    'CHAT_ID': 'YOUR_TELEGRAM_CHAT_ID'
}

# File paths configuration
FILE_PATHS = {
    'MODEL_DIR': 'models/',
    'DATA_DIR': 'data/',
    'LOGS_DIR': 'logs/',
    'CHECKPOINTS_DIR': 'models/checkpoints/',
    'TENSORBOARD_LOGS': 'models/logs/tensorboard/',
    'TRAINING_DATA': 'data/processed/training_data.csv',
    'TRANSFORMER_MODEL': 'models/transformer_model.pt',
    'NN_MODEL': 'models/nn_model.h5',
    'RF_MODEL': 'models/model_rf.ydf/',
    'COREML_MODEL': 'models/NNModel.mlpackage/'
}