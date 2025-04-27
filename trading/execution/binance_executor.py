import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import API_CONFIG  # Import to access API keys

class BinanceExecutor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.client = None  # Initialize client

    def initialize(self):
        try:
            self.client = Client(api_key=API_CONFIG['API_KEY'], api_secret=API_CONFIG['API_SECRET'])  # Use API keys from config
            self.logger.info("BinanceExecutor initialized successfully with API keys.")
        except BinanceAPIException as e:
            self.logger.error(f"Error initializing Binance client: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"API configuration error: {e}. Please check API_CONFIG.")
            raise

    def get_balance(self):
        if self.client:
            try:
                return self.client.get_asset_balance(asset='USDT')  # Example for USDT balance
            except BinanceAPIException as e:
                self.logger.error(f"Error getting balance: {e}")
                return None
        else:
            self.logger.error("Client not initialized.")
            return None

    def get_funding_rate(self, symbol):
        if self.client:
            try:
                # Fetch funding rate for the symbol
                funding_rate = self.client.get_funding_rate(symbol=symbol)
                return funding_rate  # Return the funding rate data
            except BinanceAPIException as e:
                self.logger.error(f"Error getting funding rate: {e}")
                return None
        else:
            self.logger.error("Client not initialized.")
            return None

    def get_order_book(self, symbol, limit=10):
        if self.client:
            try:
                return self.client.get_order_book(symbol=symbol, limit=limit)
            except BinanceAPIException as e:
                self.logger.error(f"Error getting order book: {e}")
                return None
        else:
            self.logger.error("Client not initialized.")
            return None

    # Add other methods as needed
