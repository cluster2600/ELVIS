import logging
import time
from binance.exceptions import BinanceAPIException

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
from trading.fees.binance_fee_calculator import BinanceFeeCalculator

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger = None, api_key: str = None, api_secret: str = None, is_testnet: bool = False, use_futures: bool = False, default_leverage: int = 10, **kwargs):
        super().__init__(logger, **kwargs)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        self.use_futures = use_futures
        self.default_leverage = default_leverage
        
        # Initialize fee calculator for comprehensive cost tracking
        self.fee_calculator = BinanceFeeCalculator(logger)
        
        # Initialize database for paper trading
        self.db_available = False  # Default to false
        if is_testnet:
            self._init_paper_trading_db()

    def initialize(self) -> bool:
        try:
            # For pure paper trading (not futures testnet), no client is needed.
            if self.is_testnet and not self.use_futures:
                self.logger.info("Paper trading mode (spot) - no API keys required, using mock execution.")
                self.client = None
                return True

            # Determine API keys based on mode
            if self.use_futures:
                api_key = self.api_key or getattr(API_CONFIG, 'BINANCE_FUTURES_TESTNET_API_KEY' if self.is_testnet else 'BINANCE_API_KEY')
                api_secret = self.api_secret or getattr(API_CONFIG, 'BINANCE_FUTURES_TESTNET_API_SECRET' if self.is_testnet else 'BINANCE_API_SECRET')
            else: # Spot (live or testnet)
                api_key = self.api_key or getattr(API_CONFIG, 'BINANCE_API_KEY')
                api_secret = self.api_secret or getattr(API_CONFIG, 'BINANCE_API_SECRET')

            # Validate keys
            if not api_key or not api_secret or 'your_' in api_key or 'your_' in api_secret:
                raise KeyError("Valid API_KEY and API_SECRET are required for this mode.")

            # Initialize client based on use_futures flag
            if FUTURES_AVAILABLE and self.use_futures:
                base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
                self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
                self.logger.info(f"BinanceExecutor initialized with Futures connector ({'testnet' if self.is_testnet else 'live'} mode).")
                
                # Set default leverage
                try:
                    self.set_leverage('BTCUSDT', self.default_leverage)
                except Exception as e:
                    self.logger.warning(f"Could not set leverage for 'BTCUSDT': {e}")
            else:
                self.client = Client(api_key, api_secret, testnet=self.is_testnet)
                self.logger.info(f"BinanceExecutor initialized with Spot client ({'testnet' if self.is_testnet else 'live'} mode).")
            
            return True

        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize BinanceExecutor: {e}")
            return False

    def get_balance(self) -> Dict[str, float]:
        if self.client is None or (self.is_testnet and not self.use_futures):  # Paper trading mode
            # Calculate dynamic balance based on trades
            return self._calculate_paper_balance()
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                # For futures, get account balance
                account = self.client.balance()
                balances = {}
                for item in account:
                    if float(item['balance']) > 0:
                        balances[item['asset']] = float(item['balance'])
                
                # Get account info for wallet balance
                account_info = self.client.account()
                wallet_balance = float(account_info['totalWalletBalance'])
                
                self.logger.info(f"Futures account - Wallet Balance: ${wallet_balance:.2f}")
                return {'USDT': wallet_balance, 'BTC': 0.0}
            else:
                # Spot trading
                account = self.client.get_account()
                balances = {item['asset']: float(item['free']) for item in account['balances']}
                return balances
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting balance: {e}")
            return {'USDT': 10000.0, 'BTC': 0.0}  # Fallback mock balance

    def get_position(self, symbol: str) -> Dict[str, Any]:
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                # For futures, get positions
                positions = self.client.get_position_risk(symbol=symbol)
                if positions:
                    return positions[0]  # Return first position for the symbol
                return {}
            else:
                # Spot trading doesn't have positions
                return {}
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        try:
            if self.use_futures:
                return float(self.client.ticker_price(symbol=symbol)['price'])
            else:
                return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.logger.info(f"Leverage for {symbol} set to {leverage}x.")
        except BinanceAPIException as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")

    def execute_buy(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        if self.client is None or (self.is_testnet and not self.use_futures):  # Paper trading mode
            try:
                # Record trade in PostgreSQL
                current_price = price if price else self._get_mock_price(symbol)
                # Calculate comprehensive trading fee
                fee = self.fee_calculator.calculate_trading_fee(current_price, quantity, is_maker=False, is_futures=self.use_futures)
                
                self.logger.info(f"[PAPER TRADE] Executing BUY: {quantity:.6f} {symbol} at ${current_price:.2f}")
                
                # Try to record in database, but continue even if it fails
                try:
                    if self.db_available:
                        record_trade(symbol, 'BUY', current_price, quantity, 0.0, fee)
                        add_open_position(symbol, current_price, quantity, self.default_leverage)
                        self.logger.info(f"Trade recorded in database with {self.default_leverage}x leverage")
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
                    'type': 'LIMIT' if price else 'MARKET',
                    'leverage': self.default_leverage
                }
                self.logger.info(f"[PAPER TRADE] BUY order completed successfully with {self.default_leverage}x leverage: {mock_order}")
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
        if self.client is None or (self.is_testnet and not self.use_futures):  # Paper trading mode
            try:
                # Record trade in PostgreSQL
                current_price = price if price else self._get_mock_price(symbol)
                # Calculate comprehensive trading fee
                fee = self.fee_calculator.calculate_trading_fee(current_price, quantity, is_maker=False, is_futures=self.use_futures)
                
                self.logger.info(f"[PAPER TRADE] Executing SELL: {quantity:.6f} {symbol} at ${current_price:.2f}")
                
                # Calculate PnL from open positions with leverage
                pnl = self._calculate_position_pnl(symbol, current_price, quantity)
                
                # Try to record in database, but continue even if it fails
                try:
                    if self.db_available:
                        record_trade(symbol, 'SELL', current_price, quantity, pnl, fee)
                        close_open_position(symbol)  # Close the position
                        self.logger.info(f"Trade recorded in database with PnL: ${pnl:.2f}")
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
                    'type': 'LIMIT' if price else 'MARKET',
                    'leverage': self.default_leverage
                }
                self.logger.info(f"[PAPER TRADE] SELL order completed successfully with {self.default_leverage}x leverage: {mock_order} | PnL: ${pnl:.2f}")
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
        """Calculate comprehensive P&L for a position being closed, including all Binance fees."""
        try:
            positions = get_open_positions()
            position = next((p for p in positions if p[1] == symbol), None)
            if position:
                entry_price = float(position[2])
                position_quantity = float(position[3])
                leverage = int(position[4]) if len(position) > 4 else self.default_leverage
                entry_time = position[5] if len(position) > 5 else datetime.now()
                
                from datetime import datetime
                if isinstance(entry_time, str):
                    try:
                        entry_datetime = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    except:
                        entry_datetime = datetime.now()
                else:
                    entry_datetime = entry_time
                
                hours_held = (datetime.now() - entry_datetime).total_seconds() / 3600
                
                trade_quantity = min(quantity, position_quantity)
                fee_calculation = self.fee_calculator.calculate_total_position_cost(
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=trade_quantity,
                    leverage=leverage,
                    hours_held=hours_held
                )
                
                net_pnl = fee_calculation['net_pnl']
                
                self.logger.info(f"Comprehensive P&L calculation for {symbol}:")
                self.logger.info(f"  Gross P&L: ${fee_calculation['gross_pnl']:+.2f}")
                self.logger.info(f"  Total Fees: ${fee_calculation['total_fees']:.6f}")
                self.logger.info(f"  - Entry Fee: ${fee_calculation['entry_fee']:.6f}")
                self.logger.info(f"  - Exit Fee: ${fee_calculation['exit_fee']:.6f}")
                self.logger.info(f"  - Funding Fee: ${fee_calculation['funding_fee']:.6f}")
                self.logger.info(f"  - Borrowing Cost: ${fee_calculation['borrowing_cost']:.6f}")
                self.logger.info(f"  Net P&L: ${net_pnl:+.2f}")
                
                return net_pnl
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive PnL: {e}")
            try:
                positions = get_open_positions()
                position = next((p for p in positions if p[1] == symbol), None)
                if position:
                    entry_price = float(position[2])
                    position_quantity = float(position[3])
                    simple_pnl = (current_price - entry_price) * min(quantity, position_quantity)
                    return simple_pnl
            except:
                pass
            return 0.0
    
    def calculate_open_position_pnl(self, symbol: str, current_price: float, entry_price: float, quantity: float, leverage: int, entry_time) -> Dict[str, float]:
        """Calculate real-time P&L for open position including all fees."""
        try:
            from datetime import datetime
            
            # Calculate hours held
            if isinstance(entry_time, str):
                try:
                    entry_datetime = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                except:
                    entry_datetime = datetime.now()
            else:
                entry_datetime = entry_time if entry_time else datetime.now()
            
            hours_held = (datetime.now() - entry_datetime).total_seconds() / 3600
            
            # Calculate gross P&L
            gross_pnl = (current_price - entry_price) * quantity
            
            # Calculate ongoing costs (but not exit fee since position is still open)
            position_value = quantity * entry_price
            
            # Entry fee (already paid)
            entry_fee = self.fee_calculator.calculate_trading_fee(entry_price, quantity, is_maker=False, is_futures=self.use_futures)
            
            # Ongoing funding fees
            funding_fee = self.fee_calculator.calculate_funding_fee(position_value, hours_held)
            
            # Ongoing borrowing costs for leverage
            margin_used = position_value / leverage
            borrowed_amount = position_value - margin_used
            borrowing_cost = self.fee_calculator.calculate_borrowing_cost(borrowed_amount, hours_held)
            
            # Total ongoing costs (not including exit fee)
            ongoing_costs = entry_fee + funding_fee + borrowing_cost
            
            # Net P&L (what you'd get if you closed now, minus exit fee)
            estimated_exit_fee = self.fee_calculator.calculate_trading_fee(current_price, quantity, is_maker=False, is_futures=self.use_futures)
            net_pnl = gross_pnl - ongoing_costs - estimated_exit_fee
            
            return {
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'ongoing_costs': ongoing_costs,
                'entry_fee': entry_fee,
                'funding_fee': funding_fee,
                'borrowing_cost': borrowing_cost,
                'estimated_exit_fee': estimated_exit_fee,
                'hours_held': hours_held
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating open position P&L: {e}")
            # Fallback to simple calculation
            simple_pnl = (current_price - entry_price) * quantity
            return {
                'gross_pnl': simple_pnl,
                'net_pnl': simple_pnl * 0.999,  # Rough 0.1% fee estimate
                'ongoing_costs': simple_pnl * 0.001,
                'entry_fee': 0,
                'funding_fee': 0,
                'borrowing_cost': 0,
                'estimated_exit_fee': 0,
                'hours_held': 0
            }
    
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
            
            # Get all trades and calculate balance changes (exclude TEST trades)
            trades = get_all_trades(limit=1000, exclude_test=True)  # Get real trades only
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
