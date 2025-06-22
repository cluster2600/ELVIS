import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import API_CONFIG
from trading.execution.base_executor import BaseExecutor
from typing import Dict, Any

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger = None, **kwargs):
        super().__init__(logger, **kwargs)
        self.client = None

    def initialize(self) -> None:
        try:
            api_key = API_CONFIG.get('API_KEY')
            api_secret = API_CONFIG.get('API_SECRET')
            if not api_key or not api_secret:
                raise KeyError("API_KEY or API_SECRET missing in API_CONFIG")
            self.client = Client(api_key, api_secret)
            self.logger.info("BinanceExecutor initialized successfully.")
        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            raise
        except BinanceAPIException as e:
            self.logger.error(f"Error initializing Binance client: {e}")
            raise

    def get_balance(self) -> Dict[str, float]:
        try:
            account = self.client.get_account()
            balances = {item['asset']: float(item['free']) for item in account['balances']}
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"Error getting balance: {e}")
            return {}

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
