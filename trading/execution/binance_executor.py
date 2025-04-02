"""
Binance executor for the ELVIS project.
This module provides a concrete implementation of the BaseExecutor for Binance.
"""

import logging
import time
import math
import ccxt
from typing import Dict, Any, List, Optional, Tuple

from trading.execution.base_executor import BaseExecutor
from config import API_CONFIG, TRADING_CONFIG
from utils import print_info, print_error

class BinanceExecutor(BaseExecutor):
    """
    Executor for Binance exchange.
    Handles order execution on Binance Futures.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the Binance executor.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Initialize Binance client
        self.api_key = API_CONFIG['BINANCE_API_KEY']
        self.api_secret = API_CONFIG['BINANCE_API_SECRET']
        
        # Initialize exchange
        self.exchange = None
        
        # Track orders
        self.orders = {}
        self.positions = {}
        
        # Telegram notification
        self.telegram_token = API_CONFIG['TELEGRAM_TOKEN']
        self.telegram_chat_id = API_CONFIG['TELEGRAM_CHAT_ID']
        
        # Initialize with default leverage
        self.current_leverage = TRADING_CONFIG['LEVERAGE_MIN']
    
    def initialize(self) -> None:
        """
        Initialize the executor.
        """
        try:
            self.logger.info("Initializing Binance executor")
            
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'urls': {
                    'api': {
                        'public': 'https://fapi.binance.com/fapi/v1',
                        'private': 'https://fapi.binance.com/fapi/v1'
                    }
                },
            })
            
            self.logger.info("Binance executor initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Binance executor: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get the account balance.
        
        Returns:
            Dict[str, float]: The account balance.
        """
        try:
            self.logger.info("Fetching account balance")
            
            # Fetch balance
            balance = self.exchange.fetch_balance({'type': 'future'})
            
            # Extract relevant information
            result = {
                'USDT': float(balance.get('USDT', {}).get('total', 0)),
                'BTC': float(balance.get('BTC', {}).get('total', 0)),
            }
            
            self.logger.info(f"Account balance: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return {'USDT': 0.0, 'BTC': 0.0}
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current position for the specified symbol.
        
        Args:
            symbol (str): The symbol to get the position for.
            
        Returns:
            Dict[str, Any]: The position.
        """
        try:
            self.logger.info(f"Fetching position for {symbol}")
            
            # Fetch positions
            positions = self.exchange.fetch_positions([symbol])
            
            if positions and len(positions) > 0:
                position = positions[0]
                
                # Extract relevant information
                result = {
                    'symbol': position['symbol'],
                    'contracts': float(position['contracts']),
                    'notional': float(position['notional']),
                    'leverage': float(position['leverage']),
                    'entryPrice': float(position['entryPrice']) if position['entryPrice'] else None,
                    'unrealizedPnl': float(position['unrealizedPnl']),
                    'side': position['side'],
                }
                
                self.logger.info(f"Position for {symbol}: {result}")
                
                return result
            else:
                self.logger.info(f"No position found for {symbol}")
                return {
                    'symbol': symbol,
                    'contracts': 0.0,
                    'notional': 0.0,
                    'leverage': self.current_leverage,
                    'entryPrice': None,
                    'unrealizedPnl': 0.0,
                    'side': 'flat',
                }
            
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return {
                'symbol': symbol,
                'contracts': 0.0,
                'notional': 0.0,
                'leverage': self.current_leverage,
                'entryPrice': None,
                'unrealizedPnl': 0.0,
                'side': 'flat',
            }
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for the specified symbol.
        
        Args:
            symbol (str): The symbol to get the price for.
            
        Returns:
            float: The current price.
        """
        try:
            self.logger.info(f"Fetching current price for {symbol}")
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Extract last price
            price = float(ticker['last'])
            
            self.logger.info(f"Current price for {symbol}: {price}")
            
            return price
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Set the leverage for the specified symbol.
        
        Args:
            symbol (str): The symbol to set the leverage for.
            leverage (int): The leverage to set.
        """
        try:
            self.logger.info(f"Setting leverage for {symbol} to {leverage}x")
            
            # Set leverage
            self.exchange.set_leverage(leverage, symbol)
            
            # Update current leverage
            self.current_leverage = leverage
            
            self.logger.info(f"Leverage for {symbol} set to {leverage}x")
            
        except Exception as e:
            self.logger.error(f"Error setting leverage for {symbol} to {leverage}x: {e}")
            self.logger.info(f"Falling back to default leverage: {TRADING_CONFIG['LEVERAGE_MIN']}x")
            
            try:
                # Try to set default leverage
                self.exchange.set_leverage(TRADING_CONFIG['LEVERAGE_MIN'], symbol)
                self.current_leverage = TRADING_CONFIG['LEVERAGE_MIN']
            except Exception as e2:
                self.logger.error(f"Error setting default leverage: {e2}")
    
    def adjust_quantity(self, symbol: str, quantity: float, current_price: float) -> float:
        """
        Adjust the quantity based on the symbol's lot size.
        
        Args:
            symbol (str): The symbol to adjust the quantity for.
            quantity (float): The quantity to adjust.
            current_price (float): The current price.
            
        Returns:
            float: The adjusted quantity.
        """
        try:
            self.logger.info(f"Adjusting quantity for {symbol}: {quantity}")
            
            # Get market info
            market = self.exchange.market(symbol)
            
            # Get lot size filter
            lot_size_filter = next((f for f in market['info']['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                # Get step size
                step_size = float(lot_size_filter['stepSize'])
                
                # Adjust quantity
                adjusted_quantity = math.floor(quantity / step_size) * step_size
                
                # Check minimum quantity
                min_qty = float(lot_size_filter['minQty'])
                if adjusted_quantity < min_qty:
                    self.logger.warning(f"Adjusted quantity {adjusted_quantity} is below minimum {min_qty}. Using minimum.")
                    adjusted_quantity = min_qty
                
                # Check minimum notional
                min_notional_filter = next((f for f in market['info']['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                if min_notional_filter:
                    min_notional = float(min_notional_filter['notional'])
                    if adjusted_quantity * current_price < min_notional:
                        self.logger.warning(f"Notional value {adjusted_quantity * current_price} is below minimum {min_notional}. Adjusting quantity.")
                        adjusted_quantity = math.ceil(min_notional / current_price / step_size) * step_size
                
                self.logger.info(f"Adjusted quantity for {symbol}: {adjusted_quantity}")
                
                return adjusted_quantity
            else:
                self.logger.warning(f"No LOT_SIZE filter found for {symbol}. Using original quantity.")
                return quantity
            
        except Exception as e:
            self.logger.error(f"Error adjusting quantity for {symbol}: {e}")
            return quantity
    
    def execute_buy(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a buy order.
        
        Args:
            symbol (str): The symbol to buy.
            quantity (float): The quantity to buy.
            price (float): The price to buy at.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        try:
            self.logger.info(f"Executing buy order for {symbol}: {quantity} @ {price}")
            
            # Adjust quantity
            adjusted_quantity = self.adjust_quantity(symbol, quantity, price)
            
            # Create order
            order_type = kwargs.get('order_type', 'limit')
            
            if order_type == 'market':
                order = self.exchange.create_market_buy_order(
                    symbol,
                    adjusted_quantity,
                    params={"reduceOnly": False}
                )
            else:
                order = self.exchange.create_limit_buy_order(
                    symbol,
                    adjusted_quantity,
                    price,
                    params={"reduceOnly": False}
                )
            
            # Store order
            self.orders[order['id']] = order
            
            # Log order
            self.logger.info(f"Buy order executed: {order}")
            
            # Send notification
            self._send_notification(f"🟢 BUY {adjusted_quantity} {symbol} @ {price} (Leverage: {self.current_leverage}x)")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
            self._send_notification(f"❌ Error executing buy order for {symbol}: {e}")
            return {}
    
    def execute_sell(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a sell order.
        
        Args:
            symbol (str): The symbol to sell.
            quantity (float): The quantity to sell.
            price (float): The price to sell at.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        try:
            self.logger.info(f"Executing sell order for {symbol}: {quantity} @ {price}")
            
            # Adjust quantity
            adjusted_quantity = self.adjust_quantity(symbol, quantity, price)
            
            # Create order
            order_type = kwargs.get('order_type', 'limit')
            reduce_only = kwargs.get('reduce_only', True)
            
            if order_type == 'market':
                order = self.exchange.create_market_sell_order(
                    symbol,
                    adjusted_quantity,
                    params={"reduceOnly": reduce_only}
                )
            else:
                order = self.exchange.create_limit_sell_order(
                    symbol,
                    adjusted_quantity,
                    price,
                    params={"reduceOnly": reduce_only}
                )
            
            # Store order
            self.orders[order['id']] = order
            
            # Log order
            self.logger.info(f"Sell order executed: {order}")
            
            # Send notification
            self._send_notification(f"🔴 SELL {adjusted_quantity} {symbol} @ {price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing sell order for {symbol}: {e}")
            self._send_notification(f"❌ Error executing sell order for {symbol}: {e}")
            return {}
    
    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a stop loss order.
        
        Args:
            symbol (str): The symbol to set the stop loss for.
            quantity (float): The quantity to sell.
            stop_price (float): The stop price.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        try:
            self.logger.info(f"Setting stop loss for {symbol}: {quantity} @ {stop_price}")
            
            # Adjust quantity
            adjusted_quantity = self.adjust_quantity(symbol, quantity, stop_price)
            
            # Create order
            order = self.exchange.create_order(
                symbol,
                'stop',
                'sell',
                adjusted_quantity,
                None,
                params={
                    'stopPrice': stop_price,
                    'reduceOnly': True
                }
            )
            
            # Store order
            self.orders[order['id']] = order
            
            # Log order
            self.logger.info(f"Stop loss set: {order}")
            
            # Send notification
            self._send_notification(f"🛑 Stop Loss set for {adjusted_quantity} {symbol} @ {stop_price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error setting stop loss for {symbol}: {e}")
            self._send_notification(f"❌ Error setting stop loss for {symbol}: {e}")
            return {}
    
    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a take profit order.
        
        Args:
            symbol (str): The symbol to set the take profit for.
            quantity (float): The quantity to sell.
            take_profit_price (float): The take profit price.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        try:
            self.logger.info(f"Setting take profit for {symbol}: {quantity} @ {take_profit_price}")
            
            # Adjust quantity
            adjusted_quantity = self.adjust_quantity(symbol, quantity, take_profit_price)
            
            # Create order
            order = self.exchange.create_order(
                symbol,
                'limit',
                'sell',
                adjusted_quantity,
                take_profit_price,
                params={
                    'reduceOnly': True
                }
            )
            
            # Store order
            self.orders[order['id']] = order
            
            # Log order
            self.logger.info(f"Take profit set: {order}")
            
            # Send notification
            self._send_notification(f"💰 Take Profit set for {adjusted_quantity} {symbol} @ {take_profit_price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error setting take profit for {symbol}: {e}")
            self._send_notification(f"❌ Error setting take profit for {symbol}: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id (str): The order ID to cancel.
            
        Returns:
            bool: True if the order was cancelled successfully, False otherwise.
        """
        try:
            self.logger.info(f"Cancelling order {order_id}")
            
            # Get order
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found in local orders")
                return False
            
            order = self.orders[order_id]
            
            # Cancel order
            self.exchange.cancel_order(order_id, order['symbol'])
            
            # Remove from local orders
            del self.orders[order_id]
            
            self.logger.info(f"Order {order_id} cancelled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id (str): The order ID to get the status for.
            
        Returns:
            Dict[str, Any]: The order status.
        """
        try:
            self.logger.info(f"Getting status for order {order_id}")
            
            # Get order
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found in local orders")
                return {}
            
            order = self.orders[order_id]
            
            # Fetch order status
            updated_order = self.exchange.fetch_order(order_id, order['symbol'])
            
            # Update local order
            self.orders[order_id] = updated_order
            
            self.logger.info(f"Order {order_id} status: {updated_order['status']}")
            
            return updated_order
            
        except Exception as e:
            self.logger.error(f"Error getting status for order {order_id}: {e}")
            return {}
    
    def _send_notification(self, message: str) -> None:
        """
        Send a notification via Telegram.
        
        Args:
            message (str): The message to send.
        """
        try:
            if self.telegram_token and self.telegram_chat_id:
                import requests
                import urllib.parse
                
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage?chat_id={self.telegram_chat_id}&parse_mode=Markdown&text={urllib.parse.quote(message)}"
                requests.get(url, timeout=10)
                self.logger.info(f"Telegram notification sent: {message}")
            else:
                self.logger.debug("Telegram notification not sent: missing token or chat ID")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {e}")
