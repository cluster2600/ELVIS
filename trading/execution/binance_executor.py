import logging
import time
from binance.exceptions import BinanceAPIException

try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
    FUTURES_AVAILABLE = True
except ImportError:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    FUTURES_AVAILABLE = False
    
from config import API_CONFIG
from trading.execution.base_executor import BaseExecutor
from typing import Dict, Any
from utils.paper_trade_db import record_trade, add_open_position, close_open_position, get_open_positions
from trading.fees.binance_fee_calculator import BinanceFeeCalculator
from datetime import datetime

class BinanceExecutor(BaseExecutor):
    def __init__(self, logger: logging.Logger = None, api_key: str = None, api_secret: str = None, is_testnet: bool = False, use_futures: bool = False, default_leverage: int = 100, **kwargs):
        super().__init__(logger, **kwargs)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        self.use_futures = use_futures
        self.default_leverage = default_leverage
        self.fee_calculator = BinanceFeeCalculator(logger)
        self.db_available = False
        if is_testnet:
            self._init_paper_trading_db()

    def initialize(self) -> bool:
        try:
            if self.is_testnet and not self.use_futures:
                self.logger.info("Paper trading mode (spot) - no API keys required, using mock execution.")
                self.client = None
                return True  # Paper trading always succeeds

            if self.use_futures:
                api_key = self.api_key or (API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY if self.is_testnet else API_CONFIG.BINANCE_API_KEY)
                api_secret = self.api_secret or (API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET if self.is_testnet else API_CONFIG.BINANCE_API_SECRET)
            else:
                api_key = self.api_key or API_CONFIG.BINANCE_API_KEY
                api_secret = self.api_secret or API_CONFIG.BINANCE_API_SECRET

            if not api_key or not api_secret or 'your_' in api_key or 'your_' in api_secret:
                raise KeyError("Valid API_KEY and API_SECRET are required for this mode.")

            if FUTURES_AVAILABLE and self.use_futures:
                base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
                self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
                self.logger.info(f"BinanceExecutor initialized with Futures connector ({'testnet' if self.is_testnet else 'live'} mode).")
                try:
                    self.set_leverage('BTCUSDT', self.default_leverage)
                except Exception as e:
                    self.logger.warning(f"Could not set leverage for 'BTCUSDT': {e}")
            else:
                self.client = Client(api_key, api_secret, testnet=self.is_testnet)
                self.logger.info(f"BinanceExecutor initialized with Spot client ({'testnet' if self.is_testnet else 'live'} mode).")
            
            return True  # Successful initialization
            
        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            return False  # Failed due to API config
        except Exception as e:
            self.logger.error(f"Failed to initialize BinanceExecutor: {e}")
            return False  # Failed due to other error

    def get_balance(self) -> Dict[str, float]:
        if self.client is None or (self.is_testnet and not self.use_futures):
            return self._calculate_paper_balance()
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):
                account = self.client.balance()
                balances = {item['asset']: float(item['balance']) for item in account if float(item['balance']) > 0}
                account_info = self.client.account()
                wallet_balance = float(account_info['totalWalletBalance'])
                self.logger.info(f"Futures account - Wallet Balance: ${wallet_balance:.2f}")
                return {'USDT': wallet_balance, **balances}
            else:
                account = self.client.get_account()
                return {item['asset']: float(item['free']) for item in account['balances']}
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting balance: {e}")
            return {'USDT': 10000.0, 'BTC': 0.0}

    def get_position(self, symbol: str) -> Dict[str, Any]:
        if self.client is None or not self.use_futures:
            return {}
        try:
            positions = self.client.get_position_risk(symbol=symbol)
            return positions[0] if positions else {}
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        if self.client is None:
            return self._get_mock_price(symbol)
        try:
            if self.use_futures:
                return float(self.client.ticker_price(symbol=symbol)['price'])
            else:
                return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except (ClientError if FUTURES_AVAILABLE else BinanceAPIException) as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.client is None or not self.use_futures:
            self.logger.info(f"Paper trading: Leverage set to {leverage}x for {symbol}")
            return
        try:
            self.client.change_leverage(symbol=symbol, leverage=leverage)
            self.logger.info(f"Leverage for {symbol} set to {leverage}x.")
        except BinanceAPIException as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")

    def execute_buy(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        return self._execute_paper_trade(symbol, 'BUY', quantity, price)

    def execute_sell(self, symbol: str, quantity: float, price: float = None, **kwargs) -> Dict[str, Any]:
        return self._execute_paper_trade(symbol, 'SELL', quantity, price)

    def _execute_paper_trade(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict[str, Any]:
        try:
            current_price = price if price else self._get_mock_price(symbol)
            fee = self.fee_calculator.calculate_trading_fee(current_price, quantity, is_maker=False, is_futures=True)
            
            open_positions = get_open_positions()
            opposite_side = 'SELL' if side == 'BUY' else 'BUY'
            existing_position = next((p for p in open_positions if p[1] == symbol and p[2] == side), None)
            opposite_position = next((p for p in open_positions if p[1] == symbol and p[2] == opposite_side), None)

            pnl = 0.0
            if opposite_position:
                self.logger.info(f"[PAPER TRADE] Executing {side} to close {opposite_side} position for {symbol}")
                pnl = self._calculate_position_pnl(symbol, opposite_side, current_price, quantity)
                if self.db_available:
                    record_trade(symbol, side, current_price, quantity, pnl, fee)
                    
                    existing_quantity = float(opposite_position[4])
                    new_quantity = existing_quantity - quantity
                    
                    if new_quantity <= 0.000001:
                        close_open_position(symbol, opposite_side)
                    else:
                        close_open_position(symbol, opposite_side)
                        add_open_position(symbol, opposite_side, float(opposite_position[3]), new_quantity, self.default_leverage)
            else:
                self.logger.info(f"[PAPER TRADE] Opening new {side} position for {symbol}")
                if self.db_available:
                    record_trade(symbol, side, current_price, quantity, 0.0, fee)
                    add_open_position(symbol, side, current_price, quantity, self.default_leverage)

            mock_order = {
                'symbol': symbol, 'orderId': f"MOCK_{symbol}_{int(time.time())}", 'side': side,
                'quantity': str(quantity), 'price': str(current_price), 'status': 'FILLED',
                'type': 'LIMIT' if price else 'MARKET', 'leverage': self.default_leverage
            }
            self.logger.info(f"[PAPER TRADE] {side} order completed successfully: {mock_order} | PnL: ${pnl:.2f}")
            return mock_order
        except Exception as e:
            self.logger.error(f"[PAPER TRADE] Error executing {side} order: {e}", exc_info=True)
            return {}

    def _calculate_position_pnl(self, symbol: str, side: str, current_price: float, quantity: float) -> float:
        try:
            positions = get_open_positions()
            position = next((p for p in positions if p[1] == symbol and p[2] == side), None)
            if position:
                entry_price = float(position[3])
                position_quantity = float(position[4])
                pnl_multiplier = 1 if side == 'BUY' else -1
                simple_pnl = (current_price - entry_price) * min(quantity, position_quantity) * pnl_multiplier
                return simple_pnl
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict[str, Any]:
        if side.upper() == 'BUY':
            return self.execute_buy(symbol, quantity, price)
        elif side.upper() == 'SELL':
            return self.execute_sell(symbol, quantity, price)
        else:
            raise ValueError(f"Invalid order side: {side}")

    def _get_mock_price(self, symbol: str) -> float:
        """Get mock prices for paper trading - use realistic values"""
        if symbol == 'BTCUSDT': 
            return 97000.0
        elif symbol == 'BNBBTC':
            return 0.00665  # Realistic BNB/BTC rate
        elif symbol == 'BNBUSDT':
            return 757.0    # Realistic BNB/USDT rate
        else: 
            return 100.0

    def _init_paper_trading_db(self):
        try:
            from utils.paper_trade_db import init_db
            init_db()
            self.logger.info("📊 Paper trading database initialized - preserving existing trades")
            self.db_available = True
        except Exception as e:
            self.logger.error(f"Failed to initialize paper trading database: {e}")
            self.db_available = False

    def _calculate_paper_balance(self) -> Dict[str, float]:
        """Calculate paper trading balance with multi-asset support"""
        if not self.db_available:
            return {'USDT': 1000.0, 'BNB': 0.0}
        
        try:
            from utils.paper_trade_db import get_all_balances, get_all_trades
            
            # Get stored balances
            balances = get_all_balances()
            
            # Calculate USDT balance from trades (legacy method)
            usdt_balance = balances.get('USDT', 1000.0)
            trades = get_all_trades(limit=10000, exclude_test=True)
            for trade in trades:
                if trade[2] == 'BTCUSDT':  # Only apply trade PnL to USDT for BTC trades
                    pnl = float(trade[6]) if trade[6] is not None else 0.0
                    fee = float(trade[7]) if trade[7] is not None else 0.0
                    usdt_balance += pnl - fee
            
            # Update USDT balance
            balances['USDT'] = usdt_balance
            
            # Ensure we have minimum balances
            if 'BNB' not in balances:
                balances['BNB'] = 0.0
            
            return balances
            
        except Exception as e:
            self.logger.error(f"Error calculating paper balance: {e}", exc_info=True)
            return {'USDT': 1000.0, 'BNB': 0.0}

    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        self.logger.info(f"Paper trading: Stop loss not implemented.")
        return {}

    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        self.logger.info(f"Paper trading: Take profit not implemented.")
        return {}

    def cancel_order(self, order_id: str) -> bool:
        self.logger.info(f"Paper trading: Cancel order not implemented.")
        return True

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        self.logger.info(f"Paper trading: Get order status not implemented.")
        return {}
        
    def get_account_balance(self) -> float:
        """Get total portfolio value in USDT for trading calculations."""
        try:
            balance = self.get_balance()
            usdt_balance = balance.get('USDT', 10000.0)
            return usdt_balance
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 10000.0
            
    def calculate_open_position_pnl(self, symbol: str, side: str, current_price: float, entry_price: float, quantity: float, leverage: int, entry_time) -> Dict[str, float]:
        """
        Calculate comprehensive P&L for an open position including all fees and costs.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            current_price: Current market price
            entry_price: Position entry price
            quantity: Position quantity (always positive)
            leverage: Leverage used
            entry_time: When the position was opened
            
        Returns:
            Dict containing detailed P&L breakdown
        """
        try:
            # Determine PNL direction
            pnl_multiplier = 1 if side.upper() == 'BUY' else -1

            # Calculate basic P&L
            gross_pnl = (current_price - entry_price) * quantity * pnl_multiplier
            
            # Calculate time-based costs
            from datetime import datetime, timezone
            if isinstance(entry_time, str):
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                except:
                    entry_dt = datetime.now(timezone.utc)
            else:
                entry_dt = entry_time if entry_time else datetime.now(timezone.utc)
            
            # Ensure both datetimes are timezone-aware
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            hours_held = max(0, (now - entry_dt).total_seconds() / 3600)
            
            # Trading fees (approximate)
            position_value = abs(quantity) * current_price
            trading_fee_rate = 0.0004  # 0.04% for futures
            entry_fee = position_value * trading_fee_rate
            estimated_exit_fee = position_value * trading_fee_rate
            
            # Funding fees (approximate, every 8 hours)
            funding_intervals = max(0, int(hours_held / 8))
            funding_rate = 0.0001  # 0.01% typical funding rate
            funding_fee = position_value * funding_rate * funding_intervals
            
            # Borrowing costs for margin (approximate)
            borrowing_rate = 0.0002  # 0.02% daily
            borrowing_cost = (position_value / leverage) * borrowing_rate * (hours_held / 24)
            
            # Total ongoing costs
            ongoing_costs = entry_fee + funding_fee + borrowing_cost
            
            # Net P&L after all costs
            net_pnl = gross_pnl - ongoing_costs - estimated_exit_fee
            
            return {
                'gross_pnl': float(gross_pnl),
                'net_pnl': float(net_pnl),
                'entry_fee': float(entry_fee),
                'funding_fee': float(funding_fee),
                'borrowing_cost': float(borrowing_cost),
                'estimated_exit_fee': float(estimated_exit_fee),
                'ongoing_costs': float(ongoing_costs),
                'hours_held': float(hours_held),
                'position_value': float(position_value)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating comprehensive P&L: {e}")
            # Return safe fallback values
            pnl_multiplier = 1 if 'side' in locals() and side.upper() == 'BUY' else -1
            simple_pnl = (current_price - entry_price) * quantity * pnl_multiplier
            return {
                'gross_pnl': float(simple_pnl),
                'net_pnl': float(simple_pnl),
                'entry_fee': 0.0,
                'funding_fee': 0.0,
                'borrowing_cost': 0.0,
                'estimated_exit_fee': 0.0,
                'ongoing_costs': 0.0,
                'hours_held': 0.0,
                'position_value': float(abs(quantity) * current_price)
            }
