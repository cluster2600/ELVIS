# config/api_config.py

import os

# Retained paper-compatibility configuration. Live mode is not executable.
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()

if TRADING_MODE != "paper":
    raise RuntimeError("TRADING_MODE must be 'paper'; live mode is disabled")

API_CONFIG = {
    "api_key": os.getenv("TESTNET_API_SPOT_KEY", ""),
    "api_secret": os.getenv("TESTNET_API_SPOT_SECRET", ""),
    "base_url": "https://testnet.binance.vision",
}
