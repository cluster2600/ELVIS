import logging
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import API_CONFIG
from trading.execution.base_executor import BaseExecutor
from typing import Dict, Any

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger = None, api_key: str = None, api_secret: str = None, is_testnet: bool = False, **kwargs):
        super().__init__(logger, **kwargs)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet

    def initialize(self) -> None:
        try:
            # Use passed parameters or fallback to config
            api_key = self.api_key or API_CONFIG.get('API_KEY')
            api_secret = self.api_secret or API_CONFIG.get('API_SECRET')
            
            if not api_key or not api_secret:
                if self.is_testnet:
                    # For paper trading, use dummy keys if none provided
                    self.logger.warning("Using dummy API keys for paper trading mode")
                    self.client = None  # Will use mock trading
                    return
                else:
                    raise KeyError("API_KEY or API_SECRET missing in API_CONFIG")
            
            # Initialize client with testnet configuration
            self.client = Client(api_key, api_secret, testnet=self.is_testnet)
            self.logger.info(f"BinanceExecutor initialized successfully ({'testnet' if self.is_testnet else 'live'} mode).")
        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            raise
        except BinanceAPIException as e:
            self.logger.error(f"Error initializing Binance client: {e}")
            raise

    def get_balance(self) -> Dict[str, float]:
        if self.client is None:  # Paper trading mode
            return {'USDT': 10000.0, 'BTC': 0.0}  # Mock balance
        try:
            account = self.client.get_account()
            balances = {item['asset']: float(item['free']) for item in account['balances']}
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"Error getting balance: {e}")
            return {'USDT': 10000.0, 'BTC': 0.0}  # Fallback mock balance

    def get_position(self, symbol: str) -> Dict[str, Any]:
        try:
            positions = self.client.get_account()['positions']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            return position
        except BinanceAPIException as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.logger.info(f"Leverage for {symbol} set to {leverage}x.")
        except BinanceAPIException as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")

    def execute_buy(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        if self.client is None:  # Paper trading mode
            mock_order = {
                'symbol': symbol,
                'orderId': f"MOCK_{symbol}_{int(time.time())}",
                'side': 'BUY',
                'quantity': str(quantity),
                'price': str(price) if price else "MARKET",
                'status': 'FILLED',
                'type': 'LIMIT' if price else 'MARKET'
            }
            self.logger.info(f"[PAPER TRADE] Executed BUY order: {mock_order}")
            return mock_order
            
        order_type = Client.ORDER_TYPE_MARKET if price is None else Client.ORDER_TYPE_LIMIT
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=order_type,
                quantity=quantity,
                price=price,
                timeInForce=Client.TIME_IN_FORCE_GTC if order_type == Client.ORDER_TYPE_LIMIT else None
            )
            self.logger.info(f"Executed BUY order: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error executing BUY order for {symbol}: {e}")
            return {}

    def execute_sell(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        if self.client is None:  # Paper trading mode
            mock_order = {
                'symbol': symbol,
                'orderId': f"MOCK_{symbol}_{int(time.time())}",
                'side': 'SELL',
                'quantity': str(quantity),
                'price': str(price) if price else "MARKET",
                'status': 'FILLED',
                'type': 'LIMIT' if price else 'MARKET'
            }
            self.logger.info(f"[PAPER TRADE] Executed SELL order: {mock_order}")
            return mock_order
            
        order_type = Client.ORDER_TYPE_MARKET if price is None else Client.ORDER_TYPE_LIMIT
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=order_type,
                quantity=quantity,
                price=price,
                timeInForce=Client.TIME_IN_FORCE_GTC if order_type == Client.ORDER_TYPE_LIMIT else None
            )
            self.logger.info(f"Executed SELL order: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error executing SELL order for {symbol}: {e}")
            return {}

    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
                quantity=quantity,
                stopPrice=stop_price,
                price=stop_price, # Required for STOP_LOSS_LIMIT
                timeInForce=Client.TIME_IN_FORCE_GTC
            )
            self.logger.info(f"Executed STOP_LOSS order: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error executing STOP_LOSS order for {symbol}: {e}")
            return {}

    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_TAKE_PROFIT_LIMIT,
                quantity=quantity,
                stopPrice=take_profit_price,
                price=take_profit_price, # Required for TAKE_PROFIT_LIMIT
                timeInForce=Client.TIME_IN_FORCE_GTC
            )
            self.logger.info(f"Executed TAKE_PROFIT order: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error executing TAKE_PROFIT order for {symbol}: {e}")
            return {}

    def execute_trailing_stop_loss(self, symbol: str, quantity: float, activation_price: float, callback_rate: float) -> Dict[str, Any]:
        """ Executes a trailing stop loss order. """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type='TRAILING_STOP_MARKET',
                quantity=quantity,
                activationPrice=activation_price,
                callbackRate=callback_rate
            )
            self.logger.info(f"Executed TRAILING_STOP_LOSS order: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error executing TRAILING_STOP_LOSS for {symbol}: {e}")
            return {}

    def execute_partial_take_profit(self, symbol: str, position_size: float, partial_exit_percentage: float, price: float) -> Dict[str, Any]:
        """ Sells a percentage of the current position. """
        quantity_to_sell = position_size * (partial_exit_percentage / 100.0)
        return self.execute_sell(symbol, quantity_to_sell, price)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Cancelled order {order_id} for {symbol}.")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            return False

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error(f"Error getting status for order {order_id} on {symbol}: {e}")
            return {}
    
    def get_account_balance(self) -> float:
        """Get USDT balance for trading calculations."""
        try:
            balance = self.get_balance()
            return balance.get('USDT', 10000.0)  # Default paper trading balance
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 10000.0  # Default paper trading balance
    
    def place_order(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Unified method to place orders."""
        try:
            if side.lower() == 'buy':
                return self.execute_buy(symbol, quantity, price)
            elif side.lower() == 'sell':
                return self.execute_sell(symbol, quantity, price)
            else:
                raise ValueError(f"Invalid order side: {side}")
        except Exception as e:
            self.logger.error(f"Error placing {side} order for {symbol}: {e}")
            return {}
