import logging
import os
import time

from binance.exceptions import BinanceAPIException

try:
    from binance.error import ClientError
    from binance.um_futures import UMFutures

    FUTURES_AVAILABLE = True
except ImportError:
    FUTURES_AVAILABLE = False

from datetime import datetime
from typing import Any, Dict

from binance.client import Client

from config.config import API_CONFIG
from trading.execution.base_executor import BaseExecutor
from trading.fees.binance_fee_calculator import BinanceFeeCalculator

# Issue #12: Import rate-limit utilities (retry decorator + header checker).
from utils.binance_rate_limiter import binance_retry, check_rate_limit_headers
from utils.paper_trade_db import (
    add_open_position,
    close_open_position,
    get_open_positions,
    record_trade,
)


class BinanceExecutor(BaseExecutor):
    # Issue #14: Default leverage reduced from 100x to 3x.  Callers may pass an
    # explicit value, which is validated by validate_leverage_config() below.
    def __init__(
        self,
        logger: logging.Logger = None,
        api_key: str = None,
        api_secret: str = None,
        is_testnet: bool = False,
        use_futures: bool = False,
        default_leverage: int = None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        self.use_futures = use_futures
        self.fee_calculator = BinanceFeeCalculator(logger)
        self.db_available = False

        # Issue #14: Validate leverage before storing it.  Imports here to
        # avoid circular-import issues at module load time.
        from config.config import TRADING_CONFIG, validate_leverage_config

        resolved_leverage = (
            default_leverage
            if default_leverage is not None
            else TRADING_CONFIG["DEFAULT_LEVERAGE"]
        )
        self.default_leverage = validate_leverage_config(resolved_leverage)

        if is_testnet:
            self._init_paper_trading_db()

    def initialize(self) -> bool:
        try:
            if self.is_testnet and not self.use_futures:
                self.logger.info(
                    "Paper trading mode (spot) - no API keys required, using mock execution."
                )
                self.client = None
                return True  # Paper trading always succeeds

            if self.use_futures:
                api_key = self.api_key or (
                    API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY
                    if self.is_testnet
                    else API_CONFIG.BINANCE_API_KEY
                )
                api_secret = self.api_secret or (
                    API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET
                    if self.is_testnet
                    else API_CONFIG.BINANCE_API_SECRET
                )
            else:
                api_key = self.api_key or API_CONFIG.BINANCE_API_KEY
                api_secret = self.api_secret or API_CONFIG.BINANCE_API_SECRET

            if (
                not api_key
                or not api_secret
                or "your_" in api_key
                or "your_" in api_secret
            ):
                raise KeyError(
                    "Valid API_KEY and API_SECRET are required for this mode."
                )

            if FUTURES_AVAILABLE and self.use_futures:
                base_url = (
                    "https://testnet.binancefuture.com"
                    if self.is_testnet
                    else "https://fapi.binance.com"
                )
                self.client = UMFutures(
                    key=api_key, secret=api_secret, base_url=base_url
                )
                self.logger.info(
                    f"BinanceExecutor initialized with Futures connector ({'testnet' if self.is_testnet else 'live'} mode)."
                )
                try:
                    self.set_leverage("BTCUSDT", self.default_leverage)
                except Exception as e:
                    self.logger.warning(f"Could not set leverage for 'BTCUSDT': {e}")
            else:
                self.client = Client(api_key, api_secret, testnet=self.is_testnet)
                self.logger.info(
                    f"BinanceExecutor initialized with Spot client ({'testnet' if self.is_testnet else 'live'} mode)."
                )

            return True  # Successful initialization

        except KeyError as e:
            self.logger.error(f"API configuration error: {e}")
            return False  # Failed due to API config
        except Exception as e:
            self.logger.error(f"Failed to initialize BinanceExecutor: {e}")
            return False  # Failed due to other error

    # Issue #12: Wrap each live Binance API call with @binance_retry so that
    # transient failures (network blips, 429 rate-limit responses) are retried
    # with exponential back-off instead of failing immediately.

    def get_balance(self) -> Dict[str, float]:
        if self.client is None or (self.is_testnet and not self.use_futures):
            return self._calculate_paper_balance()
        try:
            if FUTURES_AVAILABLE and isinstance(self.client, UMFutures):

                @binance_retry
                def _fetch():
                    account = self.client.balance()
                    account_info = self.client.account()
                    return account, account_info

                account, account_info = _fetch()
                balances = {
                    item["asset"]: float(item["balance"])
                    for item in account
                    if float(item["balance"]) > 0
                }
                wallet_balance = float(account_info["totalWalletBalance"])
                self.logger.info(
                    f"Futures account - Wallet Balance: ${wallet_balance:.2f}"
                )
                return {"USDT": wallet_balance, **balances}
            else:

                @binance_retry
                def _fetch():
                    return self.client.get_account()

                account = _fetch()
                return {
                    item["asset"]: float(item["free"]) for item in account["balances"]
                }
        except ClientError if FUTURES_AVAILABLE else BinanceAPIException as e:
            self.logger.error(f"Error getting balance: {e}")
            return {"USDT": 10000.0, "BTC": 0.0}

    def get_position(self, symbol: str) -> Dict[str, Any]:
        if self.client is None or not self.use_futures:
            return {}
        try:

            @binance_retry
            def _fetch():
                return self.client.get_position_risk(symbol=symbol)

            positions = _fetch()
            return positions[0] if positions else {}
        except ClientError if FUTURES_AVAILABLE else BinanceAPIException as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        if self.client is None:
            return self._get_mock_price(symbol)
        try:
            if self.use_futures:

                @binance_retry
                def _fetch():
                    return self.client.ticker_price(symbol=symbol)

                return float(_fetch()["price"])
            else:

                @binance_retry
                def _fetch():
                    return self.client.get_symbol_ticker(symbol=symbol)

                return float(_fetch()["price"])
        except ClientError if FUTURES_AVAILABLE else BinanceAPIException as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Return the latest funding rate for ``symbol``.

        Live futures mode queries the UMFutures ``funding_rate`` endpoint and
        returns its most recent entry.  In paper mode (no client, or a spot
        client) a mock structure ``{'symbol', 'fundingRate', 'ts'}`` with a
        zero rate is returned so callers never hit the network.
        """
        if self.client is None or not (FUTURES_AVAILABLE and self.use_futures):
            return {
                "symbol": symbol,
                "fundingRate": 0.0,
                "ts": int(time.time() * 1000),
            }
        try:

            @binance_retry
            def _fetch():
                return self.client.funding_rate(symbol=symbol, limit=1)

            data = _fetch()
            latest = data[-1] if isinstance(data, list) and data else data
            if not latest:
                return {
                    "symbol": symbol,
                    "fundingRate": 0.0,
                    "ts": int(time.time() * 1000),
                }
            return {
                "symbol": latest.get("symbol", symbol),
                "fundingRate": float(latest.get("fundingRate", 0.0)),
                "ts": int(latest.get("fundingTime", int(time.time() * 1000))),
            }
        except ClientError if FUTURES_AVAILABLE else BinanceAPIException as e:
            self.logger.error(f"Error getting funding rate for {symbol}: {e}")
            return {
                "symbol": symbol,
                "fundingRate": 0.0,
                "ts": int(time.time() * 1000),
            }

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Return the current order book (bids/asks) for ``symbol``.

        Live futures mode queries the UMFutures ``depth`` endpoint; live spot
        mode uses the spot client's ``get_order_book``. In paper mode (no
        client) the REAL public spot depth is fetched (no API key required —
        consistent with paper mode trading on real klines) so order-flow
        analysis works without exchange credentials. Set
        ``ELVIS_PAPER_PUBLIC_BOOK=0`` to force the offline empty book
        (CI / air-gapped runs); any fetch failure also degrades to the empty
        ``{'symbol', 'bids', 'asks', 'timestamp'}`` shape.
        """
        if self.client is None:
            if os.getenv("ELVIS_PAPER_PUBLIC_BOOK", "1") == "1":
                try:
                    import requests

                    resp = requests.get(
                        "https://api.binance.com/api/v3/depth",
                        params={"symbol": symbol, "limit": min(int(limit), 100)},
                        timeout=5,
                    )
                    resp.raise_for_status()
                    book = resp.json()
                    return {
                        "symbol": symbol,
                        "bids": book.get("bids", []),
                        "asks": book.get("asks", []),
                        "timestamp": int(time.time() * 1000),
                    }
                except Exception as exc:
                    self.logger.debug(
                        f"Public depth fetch failed for {symbol} ({exc}); "
                        "returning empty paper book"
                    )
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": int(time.time() * 1000),
            }
        try:
            if FUTURES_AVAILABLE and self.use_futures:

                @binance_retry
                def _fetch():
                    return self.client.depth(symbol=symbol, limit=limit)

                book = _fetch()
            else:

                @binance_retry
                def _fetch():
                    return self.client.get_order_book(symbol=symbol, limit=limit)

                book = _fetch()
            return {
                "symbol": symbol,
                "bids": book.get("bids", []),
                "asks": book.get("asks", []),
                "timestamp": int(book.get("T") or book.get("E") or time.time() * 1000),
            }
        except ClientError if FUTURES_AVAILABLE else BinanceAPIException as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": int(time.time() * 1000),
            }

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if self.client is None or not self.use_futures:
            self.logger.info(f"Paper trading: Leverage set to {leverage}x for {symbol}")
            return
        try:

            @binance_retry
            def _set():
                return self.client.change_leverage(symbol=symbol, leverage=leverage)

            _set()
            self.logger.info(f"Leverage for {symbol} set to {leverage}x.")
        except BinanceAPIException as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")

    def execute_buy(
        self, symbol: str, quantity: float, price: float = None, **kwargs
    ) -> Dict[str, Any]:
        return self._execute_paper_trade(symbol, "BUY", quantity, price)

    def execute_sell(
        self, symbol: str, quantity: float, price: float = None, **kwargs
    ) -> Dict[str, Any]:
        return self._execute_paper_trade(symbol, "SELL", quantity, price)

    def _execute_paper_trade(
        self, symbol: str, side: str, quantity: float, price: float = None
    ) -> Dict[str, Any]:
        try:
            current_price = price if price else self._get_mock_price(symbol)
            fee = self.fee_calculator.calculate_trading_fee(
                current_price, quantity, is_maker=False, is_futures=True
            )

            # 🛑 CRITICAL: Check existing positions for risk management
            open_positions = get_open_positions()
            opposite_side = "SELL" if side == "BUY" else "BUY"
            existing_position = next(
                (p for p in open_positions if p[1] == symbol and p[2] == side), None
            )
            opposite_position = next(
                (p for p in open_positions if p[1] == symbol and p[2] == opposite_side),
                None,
            )

            pnl = 0.0
            should_execute = True

            if opposite_position:
                # Calculate potential PnL for opposite position
                potential_pnl = self._calculate_position_pnl(
                    symbol, opposite_side, current_price, quantity
                )

                # Only close opposite position if stop loss or profit taking conditions are met
                close_opposite = False

                # 🎯 BIGGER TRADES OPTIMIZED RISK MANAGEMENT
                stop_loss_threshold = (
                    -50.0
                )  # Larger stop for bigger positions: $50.00 loss per position
                profit_target = 25.0  # Bigger target for larger positions: $25.00 profit per position

                if potential_pnl < stop_loss_threshold:
                    self.logger.warning(
                        f"🛑 STOP LOSS: {symbol} {opposite_side} position losing ${abs(potential_pnl):.2f} (limit: ${abs(stop_loss_threshold):.2f}) - FORCE CLOSING"
                    )
                    close_opposite = True
                    pnl = potential_pnl
                elif potential_pnl >= profit_target:
                    # 💰 PROFIT TAKING: Dollar-based targets
                    self.logger.info(
                        f"💰 PROFIT TARGET REACHED: {symbol} {opposite_side} position profit ${potential_pnl:.2f} (target: ${profit_target:.2f}) - CLOSING!"
                    )
                    close_opposite = True
                    pnl = potential_pnl
                else:
                    # 🚀 ALLOW MULTIPLE POSITIONS: Don't close opposite if no stop/profit trigger
                    close_opposite = False
                    pnl = 0.0  # New position, no P&L yet

                if close_opposite:
                    self.logger.info(
                        f"[PAPER TRADE] Executing {side} to close {opposite_side} position for {symbol}"
                    )
                    if self.db_available:
                        record_trade(symbol, side, current_price, quantity, pnl, fee)

                        existing_quantity = float(opposite_position[4])
                        new_quantity = existing_quantity - quantity

                        if new_quantity <= 0.000001:
                            close_open_position(symbol, opposite_side)
                        else:
                            close_open_position(symbol, opposite_side)
                            add_open_position(
                                symbol,
                                opposite_side,
                                float(opposite_position[3]),
                                new_quantity,
                                self.default_leverage,
                            )
                else:
                    # 🚀 OPEN NEW POSITION: Allow both BUY and SELL in same symbol
                    # Check balance for new position
                    current_balance = self._calculate_paper_balance()
                    usdt_balance = current_balance.get("USDT", 0)

                    if usdt_balance >= 50:
                        self.logger.info(
                            f"[PAPER TRADE] Opening new {side} position for {symbol} (keeping existing {opposite_side})"
                        )
                        if self.db_available:
                            record_trade(
                                symbol, side, current_price, quantity, 0.0, fee
                            )
                            add_open_position(
                                symbol,
                                side,
                                current_price,
                                quantity,
                                self.default_leverage,
                            )
                    else:
                        self.logger.warning(
                            f"🛑 RISK LIMIT: USDT balance ${usdt_balance:.2f} too low - NOT opening new {side} position"
                        )
                        should_execute = False

            else:
                # Opening new position - check if we should (risk management)
                current_balance = self._calculate_paper_balance()
                usdt_balance = current_balance.get("USDT", 0)

                # 🛑 RISK MANAGEMENT: Don't open new positions if balance < $50 (lowered to allow more trading)
                if usdt_balance < 50:
                    self.logger.warning(
                        f"🛑 RISK LIMIT: USDT balance ${usdt_balance:.2f} too low - NOT opening new {side} position"
                    )
                    should_execute = False

                if should_execute:
                    self.logger.info(
                        f"[PAPER TRADE] Opening new {side} position for {symbol}"
                    )
                    if self.db_available:
                        record_trade(symbol, side, current_price, quantity, 0.0, fee)
                        add_open_position(
                            symbol, side, current_price, quantity, self.default_leverage
                        )

            if should_execute:
                mock_order = {
                    "symbol": symbol,
                    "orderId": f"MOCK_{symbol}_{int(time.time())}",
                    "side": side,
                    "quantity": str(quantity),
                    "price": str(current_price),
                    "status": "FILLED",
                    "type": "LIMIT" if price else "MARKET",
                    "leverage": self.default_leverage,
                }
                self.logger.info(
                    f"[PAPER TRADE] {side} order completed successfully: {mock_order} | PnL: ${pnl:.2f}"
                )
                return mock_order
            else:
                self.logger.info(
                    f"[PAPER TRADE] {side} order BLOCKED by risk management"
                )
                return {"status": "BLOCKED", "reason": "Risk management"}

        except Exception as e:
            self.logger.error(
                f"[PAPER TRADE] Error executing {side} order: {e}", exc_info=True
            )
            return {}

    def _calculate_position_pnl(
        self, symbol: str, side: str, current_price: float, quantity: float
    ) -> float:
        try:
            positions = get_open_positions()
            position = next(
                (p for p in positions if p[1] == symbol and p[2] == side), None
            )
            if position:
                entry_price = float(position[3])
                position_quantity = float(position[4])
                pnl_multiplier = 1 if side == "BUY" else -1
                simple_pnl = (
                    (current_price - entry_price)
                    * min(quantity, position_quantity)
                    * pnl_multiplier
                )
                return simple_pnl
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {e}")
            return 0.0

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float = None
    ) -> Dict[str, Any]:
        if side.upper() == "BUY":
            return self.execute_buy(symbol, quantity, price)
        elif side.upper() == "SELL":
            return self.execute_sell(symbol, quantity, price)
        else:
            raise ValueError(f"Invalid order side: {side}")

    def _get_mock_price(self, symbol: str) -> float:
        """Get mock prices for paper trading - use realistic values"""
        if symbol == "BTCUSDT":
            return 116500.0  # Current BTC price in USDT
        elif symbol == "BNBUSDT":
            return 844.58  # Updated BNB price in USDT (current market price)
        elif symbol == "BNBBTC":
            return 0.00725  # BNB price in BTC (realistic rate: 844.58/116500)
        else:
            return 100.0

    def _init_paper_trading_db(self):
        try:
            from utils.paper_trade_db import init_db

            init_db()
            self.logger.info(
                "📊 Paper trading database initialized - preserving existing trades"
            )
            self.db_available = True
        except Exception as e:
            self.logger.error(f"Failed to initialize paper trading database: {e}")
            self.db_available = False

    def _calculate_paper_balance(self) -> Dict[str, float]:
        """Paper equity = a single USDT deposit plus true cumulative realized P&L.

        Models a real cash account: start from a configurable deposit
        (PAPER_START_USDT env or PAPER_TRADING_CONFIG['INITIAL_USDT_BALANCE'])
        and add every realized-P&L dollar since the last session reset, with
        no per-trade or total clamp. Equity is floored at 0 (liquidation).
        """
        # Configurable starting deposit (default $100).
        try:
            from config.config import PAPER_TRADING_CONFIG

            default_start = float(
                PAPER_TRADING_CONFIG.get("INITIAL_USDT_BALANCE", 100.0)
            )
        except Exception:
            default_start = 100.0
        starting_usdt = float(os.getenv("PAPER_START_USDT", default_start))

        # True cumulative realized P&L since the latest session reset.
        total_pnl = 0.0
        try:
            from utils.paper_trade_db import get_conn

            conn = get_conn()
            if conn:
                with conn.cursor() as c:
                    c.execute("""
                        SELECT reset_timestamp FROM trading_session_resets
                        ORDER BY reset_timestamp DESC LIMIT 1
                    """)
                    reset_result = c.fetchone()
                    if reset_result:
                        c.execute(
                            "SELECT SUM(pnl) FROM trades WHERE timestamp >= %s",
                            (reset_result[0],),
                        )
                    else:
                        c.execute("SELECT SUM(pnl) FROM trades")
                    result = c.fetchone()
                    total_pnl = (
                        float(result[0]) if result and result[0] is not None else 0.0
                    )
                conn.close()
        except Exception as e:
            self.logger.warning(f"Could not calculate realized P&L: {e}")
            total_pnl = 0.0

        # Equity floored at 0 (a paper account cannot go negative / is liquidated).
        equity = max(0.0, starting_usdt + total_pnl)

        # Pure-USDT account: the deposit is cash, not pre-seeded crypto.
        balance_with_pnl = {"USDT": equity, "BNB": 0.0, "BTC": 0.0}

        if equity <= 0.0:
            self.logger.error("🚨 Paper account liquidated (equity hit $0).")

        self.logger.info(
            f"💰 PAPER BALANCE: ${equity:.2f} total "
            f"(deposit ${starting_usdt:.2f} + realized P&L ${total_pnl:.2f})"
        )
        return balance_with_pnl

    def execute_stop_loss(
        self, symbol: str, quantity: float, stop_price: float, **kwargs
    ) -> Dict[str, Any]:
        """Execute stop loss - force close losing position"""
        self.logger.warning(
            f"🛑 STOP LOSS: Force closing {symbol} position at ${stop_price:.2f}"
        )
        current_price = self._get_mock_price(symbol)

        # Determine which side to close (opposite of current losing position)
        open_positions = get_open_positions()
        position = next((p for p in open_positions if p[1] == symbol), None)

        if position:
            position_side = position[2]  # BUY or SELL
            close_side = "SELL" if position_side == "BUY" else "BUY"
            return self._execute_paper_trade(
                symbol, close_side, quantity, current_price
            )

        return {"status": "NO_POSITION"}

    def execute_take_profit(
        self, symbol: str, quantity: float, take_profit_price: float, **kwargs
    ) -> Dict[str, Any]:
        """Execute take profit - close profitable position"""
        self.logger.info(
            f"💰 TAKE PROFIT: Closing {symbol} position at ${take_profit_price:.2f}"
        )
        current_price = self._get_mock_price(symbol)

        # Determine which side to close
        open_positions = get_open_positions()
        position = next((p for p in open_positions if p[1] == symbol), None)

        if position:
            position_side = position[2]  # BUY or SELL
            close_side = "SELL" if position_side == "BUY" else "BUY"
            return self._execute_paper_trade(
                symbol, close_side, quantity, current_price
            )

        return {"status": "NO_POSITION"}

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
                pnl = self._calculate_position_pnl(
                    symbol, side, current_price, quantity
                )

                # 🎯 BIGGER TRADES OPTIMIZED STOPS
                stop_loss_amount = -50.0  # Larger stop for bigger positions: -$50.00
                profit_target = 25.0  # Bigger target for larger positions: +$25.00

                if pnl < stop_loss_amount:
                    self.logger.warning(
                        f"🛑 AUTO STOP LOSS: {symbol} {side} losing ${abs(pnl):.2f} (limit: ${abs(stop_loss_amount):.2f})"
                    )
                    close_side = "SELL" if side == "BUY" else "BUY"
                    self._execute_paper_trade(
                        symbol, close_side, quantity, current_price
                    )

                # 💰 DOLLAR-BASED TAKE PROFIT: $1-$10 range
                elif pnl >= profit_target:
                    # Check if position is very new (less than 30 seconds) - let it show in dashboard briefly
                    try:
                        if len(position) >= 7:  # Has timestamp
                            position_age = time.time() - position[6].timestamp()
                            if position_age < 30:  # Less than 30 seconds old
                                self.logger.info(
                                    f"💰 PROFITABLE POSITION: {symbol} {side} profit ${pnl:.2f} - waiting 30s for dashboard visibility"
                                )
                                continue  # Skip this iteration, let it show in dashboard
                    except Exception as e:
                        self.logger.debug(f"Position age check error: {e}")
                        pass  # If timestamp check fails, proceed with closure

                    self.logger.info(
                        f"💰 AGGRESSIVE AUTO TAKE PROFIT: {symbol} {side} profit ${pnl:.2f} - CLOSING IMMEDIATELY!"
                    )
                    close_side = "SELL" if side == "BUY" else "BUY"
                    self._execute_paper_trade(
                        symbol, close_side, quantity, current_price
                    )

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
            usdt_balance = balance.get("USDT", 10000.0)
            return usdt_balance
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 10000.0

    def calculate_open_position_pnl(
        self,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        quantity: float,
        leverage: int,
        entry_time,
    ) -> Dict[str, float]:
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
            pnl_multiplier = 1 if side.upper() == "BUY" else -1

            # Calculate basic P&L
            gross_pnl = (current_price - entry_price) * quantity * pnl_multiplier

            # Calculate time-based costs
            from datetime import datetime, timezone

            if isinstance(entry_time, str):
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
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
            borrowing_cost = (
                (position_value / leverage) * borrowing_rate * (hours_held / 24)
            )

            # Total ongoing costs
            ongoing_costs = entry_fee + funding_fee + borrowing_cost

            # Net P&L after all costs
            net_pnl = gross_pnl - ongoing_costs - estimated_exit_fee

            return {
                "gross_pnl": float(gross_pnl),
                "net_pnl": float(net_pnl),
                "entry_fee": float(entry_fee),
                "funding_fee": float(funding_fee),
                "borrowing_cost": float(borrowing_cost),
                "estimated_exit_fee": float(estimated_exit_fee),
                "ongoing_costs": float(ongoing_costs),
                "hours_held": float(hours_held),
                "position_value": float(position_value),
            }

        except Exception as e:
            self.logger.warning(f"Error calculating comprehensive P&L: {e}")
            # Return safe fallback values
            pnl_multiplier = 1 if "side" in locals() and side.upper() == "BUY" else -1
            simple_pnl = (current_price - entry_price) * quantity * pnl_multiplier
            return {
                "gross_pnl": float(simple_pnl),
                "net_pnl": float(simple_pnl),
                "entry_fee": 0.0,
                "funding_fee": 0.0,
                "borrowing_cost": 0.0,
                "estimated_exit_fee": 0.0,
                "ongoing_costs": 0.0,
                "hours_held": 0.0,
                "position_value": float(abs(quantity) * current_price),
            }

    def close_all_positions(self, reason: str = "Manual close") -> dict:
        """LIQUIDATE all open positions immediately - used during shutdown"""
        results = {"liquidated": [], "errors": [], "total_pnl": 0.0}

        try:
            open_positions = get_open_positions()
            self.logger.warning(
                f"🚨 LIQUIDATING ALL POSITIONS: {len(open_positions)} positions ({reason})..."
            )

            if len(open_positions) == 0:
                self.logger.info(
                    "💼 LIQUIDATION: No positions to liquidate - portfolio is clean"
                )
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
                    pnl = self._calculate_position_pnl(
                        symbol, side, current_price, quantity
                    )
                    results["total_pnl"] += pnl

                    # LIQUIDATE: Force close at market price
                    liquidation_side = "SELL" if side == "BUY" else "BUY"

                    self.logger.warning(
                        f"🔥 LIQUIDATING {symbol} {side} position: Entry=${entry_price:.2f}, Current=${current_price:.2f}, Quantity={quantity}"
                    )

                    # Force execution - bypass risk management for liquidation
                    result = self._execute_paper_trade(
                        symbol, liquidation_side, quantity, current_price
                    )

                    if result and result.get("status") != "BLOCKED":
                        results["liquidated"].append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "quantity": quantity,
                                "entry_price": entry_price,
                                "liquidation_price": current_price,
                                "pnl": pnl,
                                "leverage": leverage,
                                "liquidation_type": "Forced Exit",
                            }
                        )
                        self.logger.warning(
                            f"🔥 LIQUIDATED {symbol} {side}: ${entry_price:.2f} → ${current_price:.2f} = P&L ${pnl:.2f}"
                        )

                        # Record liquidation in database for tracking
                        try:
                            from utils.paper_trade_db import record_liquidation

                            record_liquidation(
                                symbol,
                                entry_price,
                                current_price,
                                quantity,
                                leverage,
                                0.0,
                            )
                        except Exception as db_e:
                            self.logger.warning(
                                f"Could not record liquidation in DB: {db_e}"
                            )

                    else:
                        # Force close in database even if trade execution failed
                        try:
                            close_open_position(symbol, side)
                            self.logger.warning(
                                f"🔥 FORCE CLOSED {symbol} {side} in database (trade execution failed)"
                            )
                            results["liquidated"].append(
                                {
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": quantity,
                                    "entry_price": entry_price,
                                    "liquidation_price": current_price,
                                    "pnl": pnl,
                                    "leverage": leverage,
                                    "liquidation_type": "Force Database Close",
                                }
                            )
                        except Exception as db_e:
                            error_msg = f"CRITICAL: Could not force close {symbol} {side} - position may remain open! Error: {db_e}"
                            results["errors"].append(error_msg)
                            self.logger.error(error_msg)

                except Exception as e:
                    error_msg = f"CRITICAL ERROR liquidating position {position}: {e}"
                    results["errors"].append(error_msg)
                    self.logger.error(error_msg)

                    # Try to force close in database as last resort
                    try:
                        close_open_position(position[1], position[2])
                        self.logger.warning(
                            f"🔥 EMERGENCY: Force closed {position[1]} {position[2]} in database"
                        )
                    except:
                        self.logger.error(
                            f"💥 EMERGENCY FAILED: Position {position[1]} {position[2]} may remain open!"
                        )

            # Final verification - check if any positions remain
            remaining_positions = get_open_positions()
            if remaining_positions:
                self.logger.error(
                    f"💥 LIQUIDATION INCOMPLETE: {len(remaining_positions)} positions still open!"
                )
                for pos in remaining_positions:
                    self.logger.error(
                        f"💥 REMAINING: {pos[1]} {pos[2]} - Manual intervention required"
                    )

            # Summary report
            if results["liquidated"]:
                liquidation_count = len(results["liquidated"])
                self.logger.warning(
                    f"🔥 LIQUIDATION COMPLETE: {liquidation_count} positions liquidated"
                )
                self.logger.warning(
                    f"💰 FINAL P&L FROM LIQUIDATIONS: ${results['total_pnl']:.2f}"
                )

                # Show breakdown by symbol
                symbols_liquidated = {}
                for liq in results["liquidated"]:
                    symbol = liq["symbol"]
                    if symbol not in symbols_liquidated:
                        symbols_liquidated[symbol] = {"count": 0, "pnl": 0.0}
                    symbols_liquidated[symbol]["count"] += 1
                    symbols_liquidated[symbol]["pnl"] += liq["pnl"]

                for symbol, data in symbols_liquidated.items():
                    self.logger.warning(
                        f"📊 {symbol}: {data['count']} positions liquidated, P&L: ${data['pnl']:.2f}"
                    )
            else:
                self.logger.info("💼 LIQUIDATION: Portfolio was already clean")

            if results["errors"]:
                self.logger.error(
                    f"⚠️ LIQUIDATION ERRORS: {len(results['errors'])} errors occurred - check logs"
                )

        except Exception as e:
            critical_error = f"💥 CRITICAL LIQUIDATION ERROR: {e}"
            results["errors"].append(critical_error)
            self.logger.error(critical_error)
            self.logger.error("💥 EMERGENCY: Manual position cleanup may be required!")

        return results
