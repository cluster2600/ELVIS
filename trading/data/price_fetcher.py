"""
Price fetcher module for the ELVIS trading system.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

import ccxt


@dataclass
class PriceFetcher:
    """Fetches price data from exchanges."""

    def __init__(self, config: Dict):
        """Initialize the price fetcher."""
        self.config = config
        self.exchange = ccxt.binance(
            {
                "apiKey": config.get("api_key", ""),
                "secret": config.get("api_secret", ""),
                "enableRateLimit": True,
            }
        )
        self.api_calls = 0  # Track number of API calls
        self.last_call_time = 0
        self.rate_limit = 1.0  # Minimum seconds between calls

    def get_price(self, symbol: str = "BTC/USDT") -> Optional[float]:
        """Get current price for a symbol."""
        try:
            current_time = time.time()
            if current_time - self.last_call_time < self.rate_limit:
                time.sleep(self.rate_limit - (current_time - self.last_call_time))

            ticker = self.exchange.fetch_ticker(symbol)
            self.api_calls += 1
            self.last_call_time = time.time()
            return ticker["last"]

        except Exception as e:
            print(f"Error fetching price: {str(e)}")
            return None

    def get_order_book(
        self, symbol: str = "BTC/USDT", limit: int = 5
    ) -> Optional[Dict]:
        """Get order book for a symbol."""
        try:
            current_time = time.time()
            if current_time - self.last_call_time < self.rate_limit:
                time.sleep(self.rate_limit - (current_time - self.last_call_time))

            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            self.api_calls += 1
            self.last_call_time = time.time()
            return orderbook

        except Exception as e:
            print(f"Error fetching order book: {str(e)}")
            return None
