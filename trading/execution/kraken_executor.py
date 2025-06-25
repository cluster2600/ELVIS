"""
Kraken Exchange Executor
Provides trading execution capabilities for Kraken exchange
"""

import ccxt
import logging
import time
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime

from .base_executor import BaseExecutor
from utils.logger_config import get_logger


class KrakenExecutor(BaseExecutor):
    """
    Kraken exchange executor implementation
    Handles order execution, balance management, and market data for Kraken
    """
    
    def __init__(self, logger: logging.Logger = None, api_key: str = None, 
                 api_secret: str = None, is_testnet: bool = True, **kwargs):
        """
        Initialize Kraken executor
        
        Args:
            logger: Logger instance
            api_key: Kraken API key
            api_secret: Kraken API secret
            is_testnet: Whether to use sandbox/testnet (Kraken doesn't have testnet)
            **kwargs: Additional configuration
        """
        super().__init__(logger, **kwargs)
        self.logger = logger or get_logger(__name__)
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet  # Note: Kraken doesn't have testnet
        
        self.client = None
        self.exchange_info = {}
        self.symbol_mapping = {
            'BTCUSDT': 'BTC/USDT',
            'ETHUSDT': 'ETH/USDT',
            'ADAUSDT': 'ADA/USDT',
            'DOTUSDT': 'DOT/USDT',
            'LINKUSDT': 'LINK/USDT'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        self.logger.info("Kraken executor initialized")
    
    def initialize(self) -> bool:
        """Initialize connection to Kraken exchange"""
        try:
            # Initialize CCXT Kraken client
            self.client = ccxt.kraken({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': False,  # Kraken doesn't have sandbox
                'enableRateLimit': True,
                'rateLimit': 1000,  # 1 request per second
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            if self.api_key and self.api_secret:
                balance = self.client.fetch_balance()
                self.logger.info("✓ Kraken authenticated connection established")
            else:
                # Test public endpoints
                ticker = self.client.fetch_ticker('BTC/USDT')
                self.logger.info("✓ Kraken public connection established")
            
            # Load exchange info
            self.exchange_info = self.client.load_markets()
            self.logger.info(f"✓ Loaded {len(self.exchange_info)} Kraken markets")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kraken connection: {e}")
            return False
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol format for Kraken"""
        return self.symbol_mapping.get(symbol, symbol)
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            self._rate_limit()
            
            if not self.client:
                raise Exception("Client not initialized")
            
            balance = self.client.fetch_balance()
            
            # Convert to standard format
            result = {}
            for currency, amount in balance['free'].items():
                if amount > 0:
                    result[currency] = {
                        'free': float(amount),
                        'locked': float(balance['used'].get(currency, 0)),
                        'total': float(balance['total'].get(currency, 0))
                    }
            
            self.logger.debug(f"Kraken balance retrieved: {len(result)} currencies")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken balance: {e}")
            return {}
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a symbol (Kraken spot doesn't have positions)"""
        try:
            kraken_symbol = self._convert_symbol(symbol)
            
            # For spot trading, return balance-based position
            balance = self.get_balance()
            base_currency = kraken_symbol.split('/')[0]
            
            if base_currency in balance:
                amount = balance[base_currency]['free']
                return {
                    'symbol': symbol,
                    'size': amount,
                    'side': 'long' if amount > 0 else 'none',
                    'entry_price': 0,  # Not tracked in spot
                    'unrealized_pnl': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'symbol': symbol,
                'size': 0,
                'side': 'none',
                'entry_price': 0,
                'unrealized_pnl': 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken position for {symbol}: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            ticker = self.client.fetch_ticker(kraken_symbol)
            price = float(ticker['last'])
            
            self.logger.debug(f"Kraken price for {symbol}: {price}")
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken price for {symbol}: {e}")
            return 0.0
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            order_book = self.client.fetch_order_book(kraken_symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': order_book['bids'][:limit],
                'asks': order_book['asks'][:limit],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken order book for {symbol}: {e}")
            return {}
    
    def execute_buy(self, symbol: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Execute buy order"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            # Determine order type
            if price:
                order = self.client.create_limit_buy_order(kraken_symbol, quantity, price)
                order_type = 'limit'
            else:
                order = self.client.create_market_buy_order(kraken_symbol, quantity)
                order_type = 'market'
            
            result = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'buy',
                'type': order_type,
                'quantity': quantity,
                'price': price or order.get('price', 0),
                'status': order['status'],
                'timestamp': order['timestamp'],
                'exchange': 'kraken',
                'raw_response': order
            }
            
            self.logger.info(f"Kraken BUY executed: {symbol} {quantity} @ {price or 'market'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Kraken buy order: {e}")
            return {'error': str(e)}
    
    def execute_sell(self, symbol: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Execute sell order"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            # Determine order type
            if price:
                order = self.client.create_limit_sell_order(kraken_symbol, quantity, price)
                order_type = 'limit'
            else:
                order = self.client.create_market_sell_order(kraken_symbol, quantity)
                order_type = 'market'
            
            result = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'sell',
                'type': order_type,
                'quantity': quantity,
                'price': price or order.get('price', 0),
                'status': order['status'],
                'timestamp': order['timestamp'],
                'exchange': 'kraken',
                'raw_response': order
            }
            
            self.logger.info(f"Kraken SELL executed: {symbol} {quantity} @ {price or 'market'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Kraken sell order: {e}")
            return {'error': str(e)}
    
    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float) -> Dict[str, Any]:
        """Execute stop loss order"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            # Kraken stop-loss order
            order = self.client.create_order(
                kraken_symbol,
                'stop-loss',
                'sell',
                quantity,
                None,
                None,
                {
                    'stopPrice': stop_price,
                    'timeInForce': 'GTC'
                }
            )
            
            result = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'sell',
                'type': 'stop_loss',
                'quantity': quantity,
                'stop_price': stop_price,
                'status': order['status'],
                'timestamp': order['timestamp'],
                'exchange': 'kraken',
                'raw_response': order
            }
            
            self.logger.info(f"Kraken STOP LOSS set: {symbol} {quantity} @ {stop_price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Kraken stop loss: {e}")
            return {'error': str(e)}
    
    def execute_take_profit(self, symbol: str, quantity: float, tp_price: float) -> Dict[str, Any]:
        """Execute take profit order"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            # Kraken take-profit order
            order = self.client.create_order(
                kraken_symbol,
                'take-profit',
                'sell',
                quantity,
                None,
                None,
                {
                    'stopPrice': tp_price,
                    'timeInForce': 'GTC'
                }
            )
            
            result = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'sell',
                'type': 'take_profit',
                'quantity': quantity,
                'take_profit_price': tp_price,
                'status': order['status'],
                'timestamp': order['timestamp'],
                'exchange': 'kraken',
                'raw_response': order
            }
            
            self.logger.info(f"Kraken TAKE PROFIT set: {symbol} {quantity} @ {tp_price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Kraken take profit: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self._rate_limit()
            
            result = self.client.cancel_order(order_id)
            self.logger.info(f"Kraken order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling Kraken order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            self._rate_limit()
            
            order = self.client.fetch_order(order_id)
            
            return {
                'order_id': order_id,
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'average_price': order['average'],
                'timestamp': order['timestamp'],
                'exchange': 'kraken',
                'raw_response': order
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken order status {order_id}: {e}")
            return {}
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            self._rate_limit()
            
            if symbol:
                kraken_symbol = self._convert_symbol(symbol)
                orders = self.client.fetch_open_orders(kraken_symbol)
            else:
                orders = self.client.fetch_open_orders()
            
            result = []
            for order in orders:
                result.append({
                    'order_id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': order['amount'],
                    'price': order['price'],
                    'status': order['status'],
                    'timestamp': order['timestamp'],
                    'exchange': 'kraken'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken open orders: {e}")
            return []
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            self._rate_limit()
            
            if symbol:
                kraken_symbol = self._convert_symbol(symbol)
                trades = self.client.fetch_my_trades(kraken_symbol, limit=limit)
            else:
                trades = self.client.fetch_my_trades(limit=limit)
            
            result = []
            for trade in trades:
                result.append({
                    'trade_id': trade['id'],
                    'order_id': trade['order'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['amount'],
                    'price': trade['price'],
                    'fee': trade['fee']['cost'],
                    'fee_currency': trade['fee']['currency'],
                    'timestamp': trade['timestamp'],
                    'exchange': 'kraken'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken trade history: {e}")
            return []
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict[str, Any]:
        """Get historical market data"""
        try:
            self._rate_limit()
            kraken_symbol = self._convert_symbol(symbol)
            
            ohlcv = self.client.fetch_ohlcv(kraken_symbol, timeframe, limit=limit)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': [
                    {
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    }
                    for candle in ohlcv
                ],
                'exchange': 'kraken'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Kraken market data for {symbol}: {e}")
            return {}
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (not supported on Kraken spot)"""
        self.logger.warning("Leverage setting not supported on Kraken spot trading")
        return False
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        return {
            'name': 'Kraken',
            'has_testnet': False,
            'supported_symbols': list(self.symbol_mapping.keys()),
            'fees': {
                'trading': {
                    'maker': 0.0016,  # 0.16%
                    'taker': 0.0026   # 0.26%
                }
            },
            'limits': {
                'amount': {
                    'min': 0.0001,
                    'max': 1000000
                }
            },
            'api_rate_limit': '1 request/second',
            'exchange_info': self.exchange_info
        }