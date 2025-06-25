import logging
import time
try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
    FUTURES_AVAILABLE = True
except ImportError:
    # Fallback to spot client if futures connector not available
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    FUTURES_AVAILABLE = False
    
from config import API_CONFIG
from trading.execution.base_executor import BaseExecutor
from typing import Dict, Any
from utils.paper_trade_db import record_trade, add_open_position, close_open_position, get_open_positions

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger = None, api_key: str = None, api_secret: str = None, is_testnet: bool = False, **kwargs):
        super().__init__(logger, **kwargs)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        
        # Initialize database for paper trading
        self.db_available = False  # Default to false
        if is_testnet:
            self._init_paper_trading_db()

    def initialize(self) -> bool:
        try:
            # Use passed parameters or fallback to config based on testnet mode
            if self.is_testnet:
                # Use futures testnet keys for paper trading
                api_key = self.api_key or API_CONFIG.get('BINANCE_FUTURES_TESTNET_API_KEY')
                api_secret = self.api_secret or API_CONFIG.get('BINANCE_FUTURES_TESTNET_API_SECRET')
            else:
                # Use regular keys for live trading
                api_key = self.api_key or API_CONFIG.get('API_KEY')
                api_secret = self.api_secret or API_CONFIG.get('API_SECRET')
            
            # Check for placeholder values
            if (not api_key or not api_secret or 
                api_key == 'your_binance_api_key_here' or 
                api_secret == 'your_binance_api_secret_here'):
                
                if self.is_testnet:
                    # For paper trading, we don't need real API keys
                    self.logger.info("Paper trading mode - no API keys required")
                    self.client = None  # Will use mock trading
                    return True
                else:
                    raise KeyError("Valid API_KEY and API_SECRET required for live trading")
            
            # Initialize client with futures connector if available
            if FUTURES_AVAILABLE:
                base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
                self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
                self.logger.info(f"BinanceExecutor initialized with Futures connector ({'testnet' if self.is_testnet else 'live'} mode).")
            else:
                # Fallback to spot client
                self.client = Client(api_key, api_secret, testnet=self.is_testnet)
                self.logger.info(f"BinanceExecutor initialized with spot client ({'testnet' if self.is_testnet else 'live'} mode).")
            
            return True
                
        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            if not self.is_testnet:
                raise
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize BinanceExecutor: {e}")
            return False

    def get_balance(self) -> Dict[str, float]:
        if self.client is None or self.is_testnet:  # Paper trading mode
            # Calculate dynamic balance based on trades
            return self._calculate_paper_balance()
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
        if self.client is None or self.is_testnet:  # Paper trading mode (no client OR testnet mode)
            try:
                # Record trade in PostgreSQL
                current_price = price if price else self._get_mock_price(symbol)
                fee = quantity * current_price * 0.0004  # Binance taker fee
                
                self.logger.info(f"[PAPER TRADE] Executing BUY: {quantity:.6f} {symbol} at ${current_price:.2f}")
                
                # Try to record in database, but continue even if it fails
                try:
                    if self.db_available:
                        record_trade(symbol, 'BUY', current_price, quantity, 0.0, fee)
                        add_open_position(symbol, current_price, quantity, 1.0)  # Default leverage 1x
                        self.logger.info("Trade recorded in database")
                    else:
                        self.logger.info("Database not available, trade executed but not recorded")
                except Exception as e:
                    self.logger.warning(f"Failed to record trade in database: {e}")
                
                mock_order = {
                    'symbol': symbol,
                    'orderId': f"MOCK_{symbol}_{int(time.time())}",
                    'side': 'BUY',
                    'quantity': str(quantity),
                    'price': str(current_price),
                    'status': 'FILLED',
                    'type': 'LIMIT' if price else 'MARKET'
                }
                self.logger.info(f"[PAPER TRADE] BUY order completed successfully: {mock_order}")
                return mock_order
            except Exception as e:
                self.logger.error(f"[PAPER TRADE] Error executing BUY order: {e}")
                return {}
            
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                # Use futures order endpoint
                order_params = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'MARKET' if price is None else 'LIMIT',
                    'quantity': quantity,
                }
                if price is not None:
                    order_params['price'] = price
                    order_params['timeInForce'] = 'GTC'
                
                order = self.client.new_order(**order_params)
                self.logger.info(f"Executed BUY order: {order}")
                return order
            else:
                # Use spot order endpoint
                order_type = Client.ORDER_TYPE_MARKET if price is None else Client.ORDER_TYPE_LIMIT
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
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error executing BUY order for {symbol}: {e}")
            return {}

    def execute_sell(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        if self.client is None or self.is_testnet:  # Paper trading mode (no client OR testnet mode)
            try:
                # Record trade in PostgreSQL
                current_price = price if price else self._get_mock_price(symbol)
                fee = quantity * current_price * 0.0004  # Binance taker fee
                
                self.logger.info(f"[PAPER TRADE] Executing SELL: {quantity:.6f} {symbol} at ${current_price:.2f}")
                
                # Calculate PnL from open positions
                pnl = self._calculate_position_pnl(symbol, current_price, quantity)
                
                # Try to record in database, but continue even if it fails
                try:
                    if self.db_available:
                        record_trade(symbol, 'SELL', current_price, quantity, pnl, fee)
                        close_open_position(symbol)  # Close the position
                        self.logger.info("Trade recorded in database")
                    else:
                        self.logger.info("Database not available, trade executed but not recorded")
                        pnl = 10.0  # Mock PnL for testing
                except Exception as e:
                    self.logger.warning(f"Failed to record trade in database: {e}")
                    pnl = 10.0  # Mock PnL for testing
                
                mock_order = {
                    'symbol': symbol,
                    'orderId': f"MOCK_{symbol}_{int(time.time())}",
                    'side': 'SELL',
                    'quantity': str(quantity),
                    'price': str(current_price),
                    'status': 'FILLED',
                    'type': 'LIMIT' if price else 'MARKET'
                }
                self.logger.info(f"[PAPER TRADE] SELL order completed successfully: {mock_order} | PnL: ${pnl:.2f}")
                return mock_order
            except Exception as e:
                self.logger.error(f"[PAPER TRADE] Error executing SELL order: {e}")
                return {}
            
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                # Use futures order endpoint
                order_params = {
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'MARKET' if price is None else 'LIMIT',
                    'quantity': quantity,
                }
                if price is not None:
                    order_params['price'] = price
                    order_params['timeInForce'] = 'GTC'
                
                order = self.client.new_order(**order_params)
                self.logger.info(f"Executed SELL order: {order}")
                return order
            else:
                # Use spot order endpoint
                order_type = Client.ORDER_TYPE_MARKET if price is None else Client.ORDER_TYPE_LIMIT
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
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
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
        """Get total portfolio value in USDT for trading calculations."""
        try:
            balance = self.get_balance()
            usdt_balance = balance.get('USDT', 10000.0)
            btc_balance = balance.get('BTC', 0.0)
            
            # Calculate total value (USDT + BTC converted to USDT)
            # For paper trading, use a reasonable BTC price estimate
            btc_price = self._get_mock_price('BTCUSDT')
            total_value = usdt_balance + (btc_balance * btc_price)
            
            self.logger.debug(f"Account balance - USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.6f} "
                            f"(${btc_balance * btc_price:.2f}), Total: ${total_value:.2f}")
            
            return total_value
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

    def _get_mock_price(self, symbol: str) -> float:
        """Get a mock price for paper trading mode."""
        # For simplicity, use a fixed price. In production, this could fetch real market data
        if symbol == 'BTCUSDT':
            return 97000.0  # Mock BTC price
        elif symbol == 'ETHUSDT':
            return 3500.0   # Mock ETH price
        else:
            return 100.0    # Default mock price

    def _calculate_position_pnl(self, symbol: str, current_price: float, quantity: float) -> float:
        """Calculate PnL for a position being closed."""
        try:
            positions = get_open_positions()
            position = next((p for p in positions if p[1] == symbol), None)
            if position:
                # position structure: (id, symbol, entry_price, quantity, leverage, entry_time)
                entry_price = position[2]
                position_quantity = position[3]
                # Calculate PnL: (exit_price - entry_price) * quantity
                pnl = (current_price - entry_price) * min(quantity, position_quantity)
                return pnl
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
            return 0.0
    
    def _init_paper_trading_db(self):
        """Initialize the paper trading database."""
        try:
            from utils.paper_trade_db import init_db, record_trade
            init_db()
            self.logger.info("Paper trading database initialized successfully")
            
            # Test database connection
            record_trade("BTCUSDT", "TEST", 97000.0, 0.001, 0.0, 0.1)
            self.logger.info("Database test trade recorded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize paper trading database: {e}")
            self.logger.warning("Paper trading will continue without database persistence")
            # Set a flag to skip database operations
            self.db_available = False
            return
        
        self.db_available = True
    
    def _calculate_paper_balance(self) -> Dict[str, float]:
        """Calculate current balance based on trades and positions."""
        try:
            if not self.db_available:
                self.logger.debug("Database not available, using fallback balance")
                return {'USDT': 10000.0, 'BTC': 0.0}  # Fallback
            
            from utils.paper_trade_db import get_all_trades, get_open_positions
            
            # Start with initial balance
            usdt_balance = 10000.0
            btc_balance = 0.0
            
            # Get all trades and calculate balance changes
            trades = get_all_trades(limit=1000)  # Get all trades
            trade_count = 0
            
            for trade in trades:
                # trade = (id, timestamp, symbol, side, price, quantity, pnl, fee)
                if len(trade) >= 8:
                    symbol = trade[2]
                    side = trade[3]
                    price = float(trade[4])
                    quantity = float(trade[5])
                    fee = float(trade[7])
                    
                    if symbol == 'BTCUSDT':
                        if side == 'BUY':
                            # Spent USDT to buy BTC
                            cost = price * quantity + fee
                            usdt_balance -= cost
                            btc_balance += quantity
                            trade_count += 1
                            self.logger.debug(f"BUY trade: -{cost:.2f} USDT, +{quantity:.6f} BTC")
                        elif side == 'SELL':
                            # Sold BTC for USDT
                            proceeds = price * quantity - fee
                            usdt_balance += proceeds
                            btc_balance -= quantity
                            trade_count += 1
                            self.logger.debug(f"SELL trade: +{proceeds:.2f} USDT, -{quantity:.6f} BTC")
            
            # Ensure no negative balances (could happen due to fees or calculation errors)
            usdt_balance = max(0.0, usdt_balance)
            btc_balance = max(0.0, btc_balance)
            
            self.logger.info(f"Paper balance calculated from {trade_count} trades: "
                           f"USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.6f}")
            
            return {'USDT': usdt_balance, 'BTC': btc_balance}
            
        except Exception as e:
            self.logger.error(f"Error calculating paper balance: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'USDT': 10000.0, 'BTC': 0.0}  # Fallback
