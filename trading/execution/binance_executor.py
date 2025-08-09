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
            
            # 🛑 CRITICAL: Check existing positions for risk management
            open_positions = get_open_positions()
            opposite_side = 'SELL' if side == 'BUY' else 'BUY'
            existing_position = next((p for p in open_positions if p[1] == symbol and p[2] == side), None)
            opposite_position = next((p for p in open_positions if p[1] == symbol and p[2] == opposite_side), None)

            pnl = 0.0
            should_execute = True
            
            if opposite_position:
                # Calculate potential PnL before executing
                potential_pnl = self._calculate_position_pnl(symbol, opposite_side, current_price, quantity)
                
                # 🛑 STOP LOSS: If losing more than $5, force close
                if potential_pnl < -5.0:
                    self.logger.warning(f"🛑 STOP LOSS: {symbol} {opposite_side} position losing ${abs(potential_pnl):.2f} - FORCE CLOSING")
                    pnl = potential_pnl
                    should_execute = True  # Force execute to close losing position
                    
                # 💰 PROFIT TAKING: If profit >= $1, close position  
                elif potential_pnl >= 1.0:
                    self.logger.info(f"💰 PROFIT TAKING: {symbol} {opposite_side} position profit ${potential_pnl:.2f} - CLOSING")
                    pnl = potential_pnl
                    should_execute = True
                    
                else:
                    pnl = potential_pnl
                    
                if should_execute:
                    self.logger.info(f"[PAPER TRADE] Executing {side} to close {opposite_side} position for {symbol}")
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
                # Opening new position - check if we should (risk management)
                current_balance = self._calculate_paper_balance()
                usdt_balance = current_balance.get('USDT', 0)
                
                # 🛑 RISK MANAGEMENT: Don't open new positions if balance < $500
                if usdt_balance < 500:
                    self.logger.warning(f"🛑 RISK LIMIT: USDT balance ${usdt_balance:.2f} too low - NOT opening new {side} position")
                    should_execute = False
                    
                if should_execute:
                    self.logger.info(f"[PAPER TRADE] Opening new {side} position for {symbol}")
                    if self.db_available:
                        record_trade(symbol, side, current_price, quantity, 0.0, fee)
                        add_open_position(symbol, side, current_price, quantity, self.default_leverage)

            if should_execute:
                mock_order = {
                    'symbol': symbol, 'orderId': f"MOCK_{symbol}_{int(time.time())}", 'side': side,
                    'quantity': str(quantity), 'price': str(current_price), 'status': 'FILLED',
                    'type': 'LIMIT' if price else 'MARKET', 'leverage': self.default_leverage
                }
                self.logger.info(f"[PAPER TRADE] {side} order completed successfully: {mock_order} | PnL: ${pnl:.2f}")
                return mock_order
            else:
                self.logger.info(f"[PAPER TRADE] {side} order BLOCKED by risk management")
                return {'status': 'BLOCKED', 'reason': 'Risk management'}
                
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
            return 116500.0  # Current BTC price in USDT
        elif symbol == 'BNBUSDT':
            return 600.0  # BNB price in USDT
        elif symbol == 'BNBBTC':
            return 0.00515  # BNB price in BTC (realistic rate: 600/116500)
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
        """Calculate paper trading balance including realized P&L from RECENT trades only"""
        # FIXED: Get total realized P&L from RECENT trades only (not all historical data)
        total_pnl = 0.0
        try:
            import psycopg2
            from utils.paper_trade_db import get_conn
            
            # Only get trades after the latest trading session reset
            conn = get_conn()
            if conn:
                with conn.cursor() as c:
                    # Get the latest reset timestamp
                    c.execute("""
                        SELECT reset_timestamp FROM trading_session_resets 
                        ORDER BY reset_timestamp DESC LIMIT 1
                    """)
                    reset_result = c.fetchone()
                    
                    if reset_result:
                        reset_timestamp = reset_result[0]
                        # Get P&L from trades after reset only
                        c.execute("""
                            SELECT SUM(pnl) FROM trades 
                            WHERE timestamp >= %s
                            AND pnl BETWEEN -100 AND 100
                        """, (reset_timestamp,))
                        result = c.fetchone()
                        total_pnl = float(result[0]) if result and result[0] is not None else 0.0
                    else:
                        # No reset found, default to 0 (fresh start)
                        total_pnl = 0.0
                conn.close()
            
            # Cap P&L to reasonable range (-$1000 to +$1000) to prevent bugs
            total_pnl = max(-1000.0, min(1000.0, total_pnl))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate recent P&L: {e}")
            total_pnl = 0.0
        
        # Starting balances - $1000 USD equivalent each in USDT, BNB, and BTC
        # USDT is pegged to USD (1 USDT ≈ 1 USD)
        # BNB and BTC amounts calculated to equal $1000 USD each
        
        bnb_price_in_usdt = self._get_mock_price('BNBUSDT') or 600.0  # BNB price in USDT
        btc_price_in_usdt = self._get_mock_price('BTCUSDT') or 116500.0  # BTC price in USDT
        
        # Calculate crypto amounts that equal $1000 USD each
        initial_bnb_amount = 1000.0 / bnb_price_in_usdt  # Amount of BNB worth $1000 USD
        initial_btc_amount = 1000.0 / btc_price_in_usdt  # Amount of BTC worth $1000 USD
        
        # FIXED: Include P&L in balance calculation (applied to USDT since it's the base currency)
        # If P&L is negative (losses), it reduces the USDT balance
        adjusted_usdt = 1000.0 + total_pnl  # Add P&L to USDT (negative P&L reduces balance)
        
        balance_with_pnl = {
            'USDT': max(0.0, adjusted_usdt),  # USDT balance (USD equivalent)
            'BNB': initial_bnb_amount,        # BNB amount (worth ~$1000 USD)
            'BTC': initial_btc_amount         # BTC amount (worth ~$1000 USD)
        }
        
        # 🚨 EMERGENCY SHUTDOWN if losses are catastrophic
        if total_pnl < -500.0:  # If losses exceed $500
            self.logger.error(f"🚨 EMERGENCY: Catastrophic losses detected! P&L: ${total_pnl:.2f}")
            self.logger.error(f"🛑 TRADING SHOULD BE STOPPED IMMEDIATELY!")
            # Could implement automatic shutdown here
        
        # Calculate current USD values for logging
        current_bnb_usd = initial_bnb_amount * bnb_price_in_usdt
        current_btc_usd = initial_btc_amount * btc_price_in_usdt
        total_portfolio_usd = adjusted_usdt + current_bnb_usd + current_btc_usd
        
        self.logger.info(f"💰 PAPER BALANCE: ${adjusted_usdt:.2f} USDT + {initial_bnb_amount:.6f} BNB (${current_bnb_usd:.2f}) + {initial_btc_amount:.8f} BTC (${current_btc_usd:.2f}) = ${total_portfolio_usd:.2f} total (P&L since reset: ${total_pnl:.2f})")
        return balance_with_pnl

    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        """Execute stop loss - force close losing position"""
        self.logger.warning(f"🛑 STOP LOSS: Force closing {symbol} position at ${stop_price:.2f}")
        current_price = self._get_mock_price(symbol)
        
        # Determine which side to close (opposite of current losing position)
        open_positions = get_open_positions()
        position = next((p for p in open_positions if p[1] == symbol), None)
        
        if position:
            position_side = position[2]  # BUY or SELL
            close_side = 'SELL' if position_side == 'BUY' else 'BUY'
            return self._execute_paper_trade(symbol, close_side, quantity, current_price)
        
        return {'status': 'NO_POSITION'}

    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        """Execute take profit - close profitable position"""
        self.logger.info(f"💰 TAKE PROFIT: Closing {symbol} position at ${take_profit_price:.2f}")
        current_price = self._get_mock_price(symbol)
        
        # Determine which side to close
        open_positions = get_open_positions()
        position = next((p for p in open_positions if p[1] == symbol), None)
        
        if position:
            position_side = position[2]  # BUY or SELL
            close_side = 'SELL' if position_side == 'BUY' else 'BUY'
            return self._execute_paper_trade(symbol, close_side, quantity, current_price)
        
        return {'status': 'NO_POSITION'}
    
    def check_and_manage_positions(self) -> None:
        """Automatically check all positions for stop loss and take profit"""
        try:
            open_positions = get_open_positions()
            current_time = time.time()
            
            for position in open_positions:
                symbol = position[1]
                side = position[2]
                entry_price = float(position[3])
                quantity = float(position[4])
                
                current_price = self._get_mock_price(symbol)
                pnl = self._calculate_position_pnl(symbol, side, current_price, quantity)
                
                # Auto stop loss at -$5
                if pnl < -5.0:
                    self.logger.warning(f"🛑 AUTO STOP LOSS: {symbol} {side} losing ${abs(pnl):.2f}")
                    close_side = 'SELL' if side == 'BUY' else 'BUY' 
                    self._execute_paper_trade(symbol, close_side, quantity, current_price)
                    
                # Auto take profit at +$1
                elif pnl >= 1.0:
                    self.logger.info(f"💰 AUTO TAKE PROFIT: {symbol} {side} profit ${pnl:.2f}")
                    close_side = 'SELL' if side == 'BUY' else 'BUY'
                    self._execute_paper_trade(symbol, close_side, quantity, current_price)
                    
        except Exception as e:
            self.logger.error(f"Error in position management: {e}")
    

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

    def close_all_positions(self, reason: str = "Manual close") -> dict:
        """LIQUIDATE all open positions immediately - used during shutdown"""
        results = {'liquidated': [], 'errors': [], 'total_pnl': 0.0}
        
        try:
            open_positions = get_open_positions()
            self.logger.warning(f"🚨 LIQUIDATING ALL POSITIONS: {len(open_positions)} positions ({reason})...")
            
            if len(open_positions) == 0:
                self.logger.info("💼 LIQUIDATION: No positions to liquidate - portfolio is clean")
                return results
            
            for position in open_positions:
                try:
                    position_id = position[0]
                    symbol = position[1]
                    side = position[2]
                    entry_price = float(position[3])
                    quantity = float(position[4])
                    leverage = float(position[5]) if len(position) > 5 else 1.0
                    
                    # Get current market price for liquidation
                    current_price = self._get_mock_price(symbol)
                    
                    # Calculate final P&L before liquidation
                    pnl = self._calculate_position_pnl(symbol, side, current_price, quantity)
                    results['total_pnl'] += pnl
                    
                    # LIQUIDATE: Force close at market price
                    liquidation_side = 'SELL' if side == 'BUY' else 'BUY'
                    
                    self.logger.warning(f"🔥 LIQUIDATING {symbol} {side} position: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Quantity={quantity}")
                    
                    # Force execution - bypass risk management for liquidation
                    result = self._execute_paper_trade(symbol, liquidation_side, quantity, current_price)
                    
                    if result and result.get('status') != 'BLOCKED':
                        results['liquidated'].append({
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'liquidation_price': current_price,
                            'pnl': pnl,
                            'leverage': leverage,
                            'liquidation_type': 'Forced Exit'
                        })
                        self.logger.warning(f"🔥 LIQUIDATED {symbol} {side}: ${entry_price:.2f} → ${current_price:.2f} = P&L ${pnl:.2f}")
                        
                        # Record liquidation in database for tracking
                        try:
                            from utils.paper_trade_db import record_liquidation
                            record_liquidation(symbol, entry_price, current_price, quantity, leverage, 0.0)
                        except Exception as db_e:
                            self.logger.warning(f"Could not record liquidation in DB: {db_e}")
                        
                    else:
                        # Force close in database even if trade execution failed
                        try:
                            close_open_position(symbol, side)
                            self.logger.warning(f"🔥 FORCE CLOSED {symbol} {side} in database (trade execution failed)")
                            results['liquidated'].append({
                                'symbol': symbol,
                                'side': side,
                                'quantity': quantity,
                                'entry_price': entry_price,
                                'liquidation_price': current_price,
                                'pnl': pnl,
                                'leverage': leverage,
                                'liquidation_type': 'Force Database Close'
                            })
                        except Exception as db_e:
                            error_msg = f"CRITICAL: Could not force close {symbol} {side} - position may remain open! Error: {db_e}"
                            results['errors'].append(error_msg)
                            self.logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"CRITICAL ERROR liquidating position {position}: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
                    # Try to force close in database as last resort
                    try:
                        close_open_position(position[1], position[2])
                        self.logger.warning(f"🔥 EMERGENCY: Force closed {position[1]} {position[2]} in database")
                    except:
                        self.logger.error(f"💥 EMERGENCY FAILED: Position {position[1]} {position[2]} may remain open!")
            
            # Final verification - check if any positions remain
            remaining_positions = get_open_positions()
            if remaining_positions:
                self.logger.error(f"💥 LIQUIDATION INCOMPLETE: {len(remaining_positions)} positions still open!")
                for pos in remaining_positions:
                    self.logger.error(f"💥 REMAINING: {pos[1]} {pos[2]} - Manual intervention required")
            
            # Summary report
            if results['liquidated']:
                liquidation_count = len(results['liquidated'])
                self.logger.warning(f"🔥 LIQUIDATION COMPLETE: {liquidation_count} positions liquidated")
                self.logger.warning(f"💰 FINAL P&L FROM LIQUIDATIONS: ${results['total_pnl']:.2f}")
                
                # Show breakdown by symbol
                symbols_liquidated = {}
                for liq in results['liquidated']:
                    symbol = liq['symbol']
                    if symbol not in symbols_liquidated:
                        symbols_liquidated[symbol] = {'count': 0, 'pnl': 0.0}
                    symbols_liquidated[symbol]['count'] += 1
                    symbols_liquidated[symbol]['pnl'] += liq['pnl']
                
                for symbol, data in symbols_liquidated.items():
                    self.logger.warning(f"📊 {symbol}: {data['count']} positions liquidated, P&L: ${data['pnl']:.2f}")
            else:
                self.logger.info("💼 LIQUIDATION: Portfolio was already clean")
            
            if results['errors']:
                self.logger.error(f"⚠️ LIQUIDATION ERRORS: {len(results['errors'])} errors occurred - check logs")
                
        except Exception as e:
            critical_error = f"💥 CRITICAL LIQUIDATION ERROR: {e}"
            results['errors'].append(critical_error)
            self.logger.error(critical_error)
            self.logger.error("💥 EMERGENCY: Manual position cleanup may be required!")
        
        return results
