"""
Coinbase Exchange Executor
Provides trading execution capabilities for Coinbase Advanced Trade (formerly Coinbase Pro)
"""

import ccxt
import logging
import time
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime

from .base_executor import BaseExecutor
from utils.logger_config import get_logger


class CoinbaseExecutor(BaseExecutor):
    """
    Coinbase Advanced Trade executor implementation
    Handles order execution, balance management, and market data for Coinbase
    """
    
    def __init__(self, logger: logging.Logger = None, api_key: str = None, 
                 api_secret: str = None, passphrase: str = None, is_testnet: bool = True, **kwargs):
        """
        Initialize Coinbase executor
        
        Args:
            logger: Logger instance
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            passphrase: Coinbase API passphrase
            is_testnet: Whether to use sandbox environment
            **kwargs: Additional configuration
        """
        super().__init__(logger, **kwargs)
        self.logger = logger or get_logger(__name__)
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.is_testnet = is_testnet
        
        self.client = None
        self.exchange_info = {}
        self.symbol_mapping = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'ADAUSDT': 'ADA-USD',
            'DOTUSDT': 'DOT-USD',
            'LINKUSDT': 'LINK-USD',
            'LTCUSDT': 'LTC-USD',
            'BCHUSDT': 'BCH-USD',
            'XLMUSDT': 'XLM-USD'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        self.logger.info("Coinbase executor initialized")
    
    def initialize(self) -> bool:
        """Initialize connection to Coinbase exchange"""
        try:
            # Initialize CCXT Coinbase client
            self.client = ccxt.coinbase({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.passphrase,
                'sandbox': self.is_testnet,
                'enableRateLimit': True,
                'rateLimit': 1000,  # 1000ms between requests
                'options': {
                    'advanced': True,  # Use Advanced Trade API
                }
            })
            
            # Test connection
            if self.api_key and self.api_secret and self.passphrase:
                balance = self.client.fetch_balance()
                env_type = "sandbox" if self.is_testnet else "production"
                self.logger.info(f"✓ Coinbase authenticated connection established ({env_type})")
            else:
                # Test public endpoints
                ticker = self.client.fetch_ticker('BTC-USD')
                self.logger.info("✓ Coinbase public connection established")
            
            # Load exchange info
            self.exchange_info = self.client.load_markets()
            self.logger.info(f"✓ Loaded {len(self.exchange_info)} Coinbase markets")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Coinbase connection: {e}")
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
        """Convert symbol format for Coinbase"""
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
            
            self.logger.debug(f"Coinbase balance retrieved: {len(result)} currencies")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase balance: {e}")
            return {}
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for a symbol (Coinbase spot doesn't have positions)"""
        try:
            coinbase_symbol = self._convert_symbol(symbol)
            
            # For spot trading, return balance-based position
            balance = self.get_balance()
            base_currency = coinbase_symbol.split('-')[0]
            
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
            self.logger.error(f"Error getting Coinbase position for {symbol}: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            ticker = self.client.fetch_ticker(coinbase_symbol)
            price = float(ticker['last'])
            
            self.logger.debug(f"Coinbase price for {symbol}: {price}")
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase price for {symbol}: {e}")
            return 0.0
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            order_book = self.client.fetch_order_book(coinbase_symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': order_book['bids'][:limit],
                'asks': order_book['asks'][:limit],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase order book for {symbol}: {e}")
            return {}
    
    def execute_buy(self, symbol: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Execute buy order"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            # Determine order type
            if price:
                order = self.client.create_limit_buy_order(coinbase_symbol, quantity, price)
                order_type = 'limit'
            else:
                order = self.client.create_market_buy_order(coinbase_symbol, quantity)
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
                'exchange': 'coinbase',
                'raw_response': order
            }
            
            self.logger.info(f"Coinbase BUY executed: {symbol} {quantity} @ {price or 'market'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Coinbase buy order: {e}")
            return {'error': str(e)}
    
    def execute_sell(self, symbol: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Execute sell order"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            # Determine order type
            if price:
                order = self.client.create_limit_sell_order(coinbase_symbol, quantity, price)
                order_type = 'limit'
            else:
                order = self.client.create_market_sell_order(coinbase_symbol, quantity)
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
                'exchange': 'coinbase',
                'raw_response': order
            }
            
            self.logger.info(f"Coinbase SELL executed: {symbol} {quantity} @ {price or 'market'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Coinbase sell order: {e}")
            return {'error': str(e)}
    
    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float) -> Dict[str, Any]:
        """Execute stop loss order"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            # Coinbase stop order
            order = self.client.create_order(
                coinbase_symbol,
                'stop',
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
                'exchange': 'coinbase',
                'raw_response': order
            }
            
            self.logger.info(f"Coinbase STOP LOSS set: {symbol} {quantity} @ {stop_price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Coinbase stop loss: {e}")
            return {'error': str(e)}
    
    def execute_take_profit(self, symbol: str, quantity: float, tp_price: float) -> Dict[str, Any]:
        """Execute take profit order (using limit order)"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            # Coinbase doesn't have explicit take-profit, use limit order
            order = self.client.create_limit_sell_order(coinbase_symbol, quantity, tp_price)
            
            result = {
                'order_id': order['id'],
                'symbol': symbol,
                'side': 'sell',
                'type': 'take_profit',
                'quantity': quantity,
                'take_profit_price': tp_price,
                'status': order['status'],
                'timestamp': order['timestamp'],
                'exchange': 'coinbase',
                'raw_response': order
            }
            
            self.logger.info(f"Coinbase TAKE PROFIT set: {symbol} {quantity} @ {tp_price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing Coinbase take profit: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self._rate_limit()
            
            result = self.client.cancel_order(order_id)
            self.logger.info(f"Coinbase order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling Coinbase order {order_id}: {e}")
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
                'exchange': 'coinbase',
                'raw_response': order
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase order status {order_id}: {e}")
            return {}
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            self._rate_limit()
            
            if symbol:
                coinbase_symbol = self._convert_symbol(symbol)
                orders = self.client.fetch_open_orders(coinbase_symbol)
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
                    'exchange': 'coinbase'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase open orders: {e}")
            return []
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            self._rate_limit()
            
            if symbol:
                coinbase_symbol = self._convert_symbol(symbol)
                trades = self.client.fetch_my_trades(coinbase_symbol, limit=limit)
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
                    'exchange': 'coinbase'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase trade history: {e}")
            return []
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict[str, Any]:
        """Get historical market data"""
        try:
            self._rate_limit()
            coinbase_symbol = self._convert_symbol(symbol)
            
            ohlcv = self.client.fetch_ohlcv(coinbase_symbol, timeframe, limit=limit)
            
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
                'exchange': 'coinbase'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Coinbase market data for {symbol}: {e}")
            return {}
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol (not supported on Coinbase spot)"""
        self.logger.warning("Leverage setting not supported on Coinbase spot trading")
        return False
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        return {
            'name': 'Coinbase Advanced Trade',
            'has_testnet': True,
            'supported_symbols': list(self.symbol_mapping.keys()),
            'fees': {
                'trading': {
                    'maker': 0.005,   # 0.5%
                    'taker': 0.005    # 0.5%
                }
            },
            'limits': {
                'amount': {
                    'min': 0.001,
                    'max': 1000000
                }
            },
            'api_rate_limit': '10 requests/second',
            'exchange_info': self.exchange_info
        }