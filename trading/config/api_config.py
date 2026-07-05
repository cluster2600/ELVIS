# config/api_config.py

import os

# Detect mode: paper = testnet, otherwise live
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()

if TRADING_MODE == "paper":
    API_CONFIG = {
        "api_key": os.getenv("TESTNET_API_SPOT_KEY", ""),
        "api_secret": os.getenv("TESTNET_API_SPOT_SECRET", ""),
        "base_url": "https://testnet.binance.vision",
    }
else:
    API_CONFIG = {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "base_url": "https://api.binance.com",
    }
