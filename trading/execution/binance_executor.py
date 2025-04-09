import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple

from binance.um_futures import UMFutures
from trading.execution.base_executor import BaseExecutor
from config import API_CONFIG, TRADING_CONFIG

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.is_testnet = not TRADING_CONFIG.get('PRODUCTION_MODE', False)
        if self.is_testnet:
            self.api_key = API_CONFIG['TESTNET_FUTURES_API']
            self.api_secret = API_CONFIG['TESTNET_FUTURES_SECRET']
            self.logger.info(f"Using Futures Testnet API Key: {self.api_key[:5]}...")
            self.logger.info(f"Using Futures Testnet API Secret: {self.api_secret[:5]}...")
        else:
            self.api_key = API_CONFIG['BINANCE_API_KEY']
            self.api_secret = API_CONFIG['BINANCE_API_SECRET']
            self.logger.info(f"Using Production API Key: {self.api_key[:5]}...")
            self.logger.info(f"Using Production API Secret: {self.api_secret[:5]}...")
        self.client = None
        self.orders = {}
        self.positions = {}  # Local position tracking
        self.telegram_token = API_CONFIG['TELEGRAM_TOKEN']
        self.telegram_chat_id = API_CONFIG['TELEGRAM_CHAT_ID']
        self.current_leverage = TRADING_CONFIG['LEVERAGE_MIN']
    
    def initialize(self) -> None:
        try:
            self.logger.info("Initializing Binance Futures executor")
            base_url = 'https://testnet.binancefuture.com' if self.is_testnet else 'https://fapi.binance.com'
            self.client = UMFutures(
                key=self.api_key,
                secret=self.api_secret,
                base_url=base_url,
                timeout=5
            )
            server_time = self.client.time()
            self.logger.info(f"Server time: {server_time}")
            self._test_public_data()
            account = self.client.account()
            self.logger.info(f"Account initialized: {account['totalWalletBalance']}")
            self.set_leverage(TRADING_CONFIG['SYMBOL'], self.current_leverage)
            self.logger.info("Binance Futures executor initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Binance Futures executor: {e}")
            raise
    
    def _test_public_data(self) -> None:
        try:
            self.logger.debug("Testing public klines data fetch")
            klines = self.client.klines(symbol='BTCUSDT', interval='1m', limit=5)
            self.logger.info(f"Sample klines data: {klines[:1]}")
        except Exception as e:
            self.logger.error(f"Error fetching public klines: {e}")
    
    def get_balance(self) -> Dict[str, float]:
        try:
            self.logger.info("Fetching account balance")
            account = self.client.account()
            result = {
                'USDT': float(account['totalWalletBalance']),
                'BTC': 0.0
            }
            self.logger.info(f"Account balance: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return {'USDT': 0.0, 'BTC': 0.0}
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Fetching position for {symbol}")
            account = self.client.account()
            position = next((p for p in account['positions'] if p['symbol'] == symbol and float(p['positionAmt']) != 0), None)
            if position:
                result = {
                    'symbol': position['symbol'],
                    'contracts': float(position['positionAmt']),
                    'notional': float(position['notional']),
                    'leverage': float(position['leverage']),
                    'entryPrice': float(position['entryPrice']),
                    'unrealizedPnl': float(position['unrealizedProfit']),
                    'side': 'long' if float(position['positionAmt']) > 0 else 'short'
                }
                self.positions[symbol] = result
                self.logger.info(f"Position for {symbol}: {result}")
                return result
            else:
                self.logger.info(f"No position found for {symbol} in API")
                return self.positions.get(symbol, {
                    'symbol': symbol,
                    'contracts': 0.0,
                    'notional': 0.0,
                    'leverage': self.current_leverage,
                    'entryPrice': 0.0,
                    'unrealizedPnl': 0.0,
                    'side': 'flat'
                })
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return self.positions.get(symbol, {
                'symbol': symbol,
                'contracts': 0.0,
                'notional': 0.0,
                'leverage': self.current_leverage,
                'entryPrice': 0.0,
                'unrealizedPnl': 0.0,
                'side': 'flat'
            })
    
    def get_current_price(self, symbol: str) -> float:
        try:
            self.logger.info(f"Fetching current price for {symbol}")
            ticker = self.client.mark_price(symbol)
            price = float(ticker['markPrice'])
            self.logger.info(f"Current price for {symbol}: {price}")
            return price
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.logger.info(f"Setting leverage for {symbol} to {leverage}x")
            self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.current_leverage = leverage
            self.logger.info(f"Leverage for {symbol} set to {leverage}x")
        except Exception as e:
            self.logger.error(f"Error setting leverage for {symbol} to {leverage}x: {e}")
            self.logger.info(f"Falling back to default leverage: {TRADING_CONFIG['LEVERAGE_MIN']}x")
            try:
                self.client.change_leverage(symbol=symbol, leverage=TRADING_CONFIG['LEVERAGE_MIN'])
                self.current_leverage = TRADING_CONFIG['LEVERAGE_MIN']
            except Exception as e2:
                self.logger.error(f"Error setting default leverage: {e2}")
    
    def adjust_quantity(self, symbol: str, quantity: float, current_price: float) -> float:
        try:
            self.logger.info(f"Adjusting quantity for {symbol}: {quantity}")
            exchange_info = self.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found in exchange info")
            precision = symbol_info['quantityPrecision']
            step_size = float(next(f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'))
            adjusted_quantity = math.floor(quantity / step_size) * step_size
            self.logger.info(f"Adjusted quantity for {symbol}: {adjusted_quantity}")
            return adjusted_quantity
        except Exception as e:
            self.logger.error(f"Error adjusting quantity for {symbol}: {e}")
            return quantity
    
    def execute_buy(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Executing buy order for {symbol}: {quantity} @ {price}")
            adjusted_quantity = self.adjust_quantity(symbol, quantity, price)
            order_type = 'MARKET'  # Switch to market order for immediate execution
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': order_type,
                'quantity': adjusted_quantity
            }
            order = self.client.new_order(**params)
            self.orders[order['orderId']] = order
            self.logger.info(f"Buy order executed: {order}")
            # Update local position immediately
            self.positions[symbol] = {
                'symbol': symbol,
                'contracts': adjusted_quantity,
                'notional': adjusted_quantity * price,
                'leverage': self.current_leverage,
                'entryPrice': price,
                'unrealizedPnl': 0.0,
                'side': 'long'
            }
            self._send_notification(f"🟢 BUY {adjusted_quantity} {symbol} @ {price} (Leverage: {self.current_leverage}x)")
            # Verify order status
            time.sleep(1)  # Wait briefly for fill
            status = self.get_order_status(order['orderId'])
            self.logger.debug(f"Buy order status: {status}")
            return order
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
            self._send_notification(f"❌ Error executing buy order for {symbol}: {e}")
            return {}
    
    def execute_sell(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Executing sell order for {symbol}: {quantity} @ {price}")
            adjusted_quantity = self.adjust_quantity(symbol, quantity, price)
            order_type = 'MARKET'  # Switch to market order
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': order_type,
                'quantity': adjusted_quantity
            }
            order = self.client.new_order(**params)
            self.orders[order['orderId']] = order
            self.logger.info(f"Sell order executed: {order}")
            # Clear local position
            if symbol in self.positions:
                del self.positions[symbol]
            self._send_notification(f"🔴 SELL {adjusted_quantity} {symbol} @ {price}")
            # Verify order status
            time.sleep(1)
            status = self.get_order_status(order['orderId'])
            self.logger.debug(f"Sell order status: {status}")
            return order
        except Exception as e:
            self.logger.error(f"Error executing sell order for {symbol}: {e}")
            self._send_notification(f"❌ Error executing sell order for {symbol}: {e}")
            return {}
    
    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Setting stop loss for {symbol}: {quantity} @ {stop_price}")
            adjusted_quantity = self.adjust_quantity(symbol, quantity, stop_price)
            order = self.client.new_order(
                symbol=symbol,
                side='SELL',
                type='STOP_MARKET',
                quantity=adjusted_quantity,
                stopPrice=stop_price
            )
            self.orders[order['orderId']] = order
            self.logger.info(f"Stop loss set: {order}")
            self._send_notification(f"🛑 Stop Loss set for {adjusted_quantity} {symbol} @ {stop_price}")
            return order
        except Exception as e:
            self.logger.error(f"Error setting stop loss for {symbol}: {e}")
            self._send_notification(f"❌ Error setting stop loss for {symbol}: {e}")
            return {}
    
    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Setting take profit for {symbol}: {quantity} @ {take_profit_price}")
            adjusted_quantity = self.adjust_quantity(symbol, quantity, take_profit_price)
            order = self.client.new_order(
                symbol=symbol,
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                quantity=adjusted_quantity,
                stopPrice=take_profit_price
            )
            self.orders[order['orderId']] = order
            self.logger.info(f"Take profit set: {order}")
            self._send_notification(f"💰 Take Profit set for {adjusted_quantity} {symbol} @ {take_profit_price}")
            return order
        except Exception as e:
            self.logger.error(f"Error setting take profit for {symbol}: {e}")
            self._send_notification(f"❌ Error setting take profit for {symbol}: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.logger.info(f"Cancelling order {order_id}")
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found in local orders")
                return False
            order = self.orders[order_id]
            self.client.cancel_order(symbol=order['symbol'], orderId=order_id)
            del self.orders[order_id]
            self.logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Getting status for order {order_id}")
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found in local orders")
                return {}
            order = self.orders[order_id]
            updated_order = self.client.query_order(symbol=order['symbol'], orderId=order_id)
            self.orders[order_id] = updated_order
            self.logger.info(f"Order {order_id} status: {updated_order['status']}")
            return updated_order
        except Exception as e:
            self.logger.error(f"Error getting status for order {order_id}: {e}")
            return {}
    
    def _send_notification(self, message: str) -> None:
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