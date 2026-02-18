"""
trading/paper_trading.py — Sprint 3: Paper Trading Engine
==========================================================
Provides a self-contained PaperTradingEngine that:
  • Optionally connects to the Binance Testnet Futures API
    (https://testnet.binancefuture.com) for real price feeds
  • Tracks virtual P&L, open positions, and full trade history
  • Persists every trade to a local SQLite database (paper_trading.db)
  • Honours the kill-switch (Redis-backed if available, else in-memory)
  • Activated by setting the env var PAPER_TRADING=true

Usage
-----
    from trading.paper_trading import PaperTradingEngine
    engine = PaperTradingEngine(initial_balance=10_000.0)
    engine.place_order("BTCUSDT", side="BUY", quantity=0.001, price=65_000.0)
    print(engine.get_portfolio_summary())
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully so the engine works standalone
# ---------------------------------------------------------------------------
try:
    from binance.um_futures import UMFutures
    FUTURES_AVAILABLE = True
except ImportError:
    FUTURES_AVAILABLE = False

try:
    import redis as _redis_mod
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESTNET_BASE_URL = "https://testnet.binancefuture.com"
DEFAULT_DB_PATH = os.environ.get("PAPER_TRADING_DB", "paper_trading.db")
TAKER_FEE_RATE = 0.0004   # Binance Futures taker fee (0.04 %)
MAKER_FEE_RATE = 0.0002   # Binance Futures maker fee (0.02 %)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    side: str            # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    leverage: int = 3
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    def unrealised_pnl(self, current_price: float) -> float:
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


@dataclass
class Trade:
    trade_id: int
    symbol: str
    side: str            # "BUY" or "SELL"
    quantity: float
    price: float
    fee: float
    realised_pnl: float
    balance_after: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    note: str = ""


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

_DB_INIT_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id      INTEGER,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    quantity      REAL    NOT NULL,
    price         REAL    NOT NULL,
    fee           REAL    NOT NULL DEFAULT 0,
    realised_pnl  REAL    NOT NULL DEFAULT 0,
    balance_after REAL    NOT NULL,
    timestamp     TEXT    NOT NULL,
    note          TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS positions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL UNIQUE,
    side        TEXT    NOT NULL,
    quantity    REAL    NOT NULL,
    entry_price REAL    NOT NULL,
    leverage    INTEGER NOT NULL DEFAULT 3,
    opened_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DB_INIT_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Kill-switch helper
# ---------------------------------------------------------------------------

class _KillSwitch:
    """Thin wrapper that checks Redis first, falls back to in-memory flag."""

    REDIS_KEY = "elvis:kill_switch"

    def __init__(self):
        self._local = False
        self._redis: Optional[object] = None
        if REDIS_AVAILABLE:
            try:
                host = os.environ.get("REDIS_HOST", "localhost")
                port = int(os.environ.get("REDIS_PORT", 6379))
                db   = int(os.environ.get("REDIS_DB", 0))
                pwd  = os.environ.get("REDIS_PASSWORD") or None
                r = _redis_mod.Redis(host=host, port=port, db=db,
                                     password=pwd, socket_timeout=1)
                r.ping()
                self._redis = r
                logger.info("KillSwitch: Redis connected (%s:%s)", host, port)
            except Exception as exc:
                logger.warning("KillSwitch: Redis unavailable (%s) — using in-memory flag", exc)

    @property
    def active(self) -> bool:
        if self._redis:
            try:
                val = self._redis.get(self.REDIS_KEY)
                return val is not None and val.decode() == "1"
            except Exception:
                pass
        return self._local

    def activate(self):
        self._local = True
        if self._redis:
            try:
                self._redis.set(self.REDIS_KEY, "1")
            except Exception:
                pass
        logger.critical("KillSwitch ACTIVATED — all trading halted")

    def deactivate(self):
        self._local = False
        if self._redis:
            try:
                self._redis.delete(self.REDIS_KEY)
            except Exception:
                pass
        logger.info("KillSwitch deactivated")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Simulated trading engine backed by the Binance Testnet (optional).

    Parameters
    ----------
    initial_balance : float
        Starting USDT balance (virtual).
    db_path : str
        Path to the SQLite database file.
    leverage : int
        Default leverage (capped at 20x for safety).
    use_testnet_prices : bool
        If True and testnet API keys are present, fetch live testnet prices.
    """

    MAX_LEVERAGE = 20   # Hard cap — never exceed this in paper mode

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        db_path: str = DEFAULT_DB_PATH,
        leverage: int = 3,
        use_testnet_prices: bool = True,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.db_path = db_path
        self.leverage = min(max(int(leverage), 1), self.MAX_LEVERAGE)

        self._positions: Dict[str, Position] = {}
        self._trade_counter = 0
        self._trades: List[Trade] = []
        self._lock = threading.Lock()
        self._kill_switch = _KillSwitch()

        # SQLite
        self._db = _get_db(db_path)
        self._restore_state()

        # Binance Testnet client (optional)
        self._testnet_client = None
        if use_testnet_prices and FUTURES_AVAILABLE:
            self._init_testnet_client()

        logger.info(
            "PaperTradingEngine ready | balance=%.2f USDT | leverage=%dx | db=%s",
            self.balance, self.leverage, db_path,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_testnet_client(self):
        api_key    = os.environ.get("BINANCE_TESTNET_API_KEY") or \
                     os.environ.get("BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET") or \
                     os.environ.get("BINANCE_FUTURES_TESTNET_API_SECRET", "")
        if not api_key or "your_" in api_key:
            logger.info("No testnet API keys — price feed disabled (simulation mode)")
            return
        try:
            self._testnet_client = UMFutures(
                key=api_key, secret=api_secret, base_url=TESTNET_BASE_URL
            )
            # Quick connectivity test
            self._testnet_client.ticker_price("BTCUSDT")
            logger.info("Binance Testnet price feed connected")
        except Exception as exc:
            logger.warning("Testnet price feed failed (%s) — simulation mode", exc)
            self._testnet_client = None

    def _restore_state(self):
        """Reload balance + open positions from SQLite on startup."""
        cur = self._db.execute("SELECT value FROM metadata WHERE key='balance'")
        row = cur.fetchone()
        if row:
            self.balance = float(row["value"])

        cur = self._db.execute("SELECT value FROM metadata WHERE key='trade_counter'")
        row = cur.fetchone()
        if row:
            self._trade_counter = int(row["value"])

        for row in self._db.execute("SELECT * FROM positions"):
            pos = Position(
                symbol=row["symbol"],
                side=row["side"],
                quantity=row["quantity"],
                entry_price=row["entry_price"],
                leverage=row["leverage"],
                opened_at=row["opened_at"],
            )
            self._positions[row["symbol"]] = pos

        logger.debug("State restored: balance=%.2f, %d open positions",
                     self.balance, len(self._positions))

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> Optional[float]:
        """Return current price from testnet, or None if unavailable."""
        if self._testnet_client:
            try:
                data = self._testnet_client.ticker_price(symbol)
                return float(data["price"])
            except Exception as exc:
                logger.warning("Price fetch failed for %s: %s", symbol, exc)
        return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,           # "BUY" or "SELL"
        quantity: float,
        price: Optional[float] = None,
        note: str = "",
    ) -> Optional[Trade]:
        """
        Simulate a market order.

        If ``price`` is None, the engine fetches the current testnet price.
        Returns the recorded Trade object, or None if the order was blocked.
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")

        # Kill-switch check
        if self._kill_switch.active:
            logger.warning("Order BLOCKED — kill-switch is active")
            return None

        # Resolve price
        if price is None:
            price = self.get_price(symbol)
        if price is None or price <= 0:
            logger.error("Cannot place order for %s — no valid price", symbol)
            return None

        with self._lock:
            fee = quantity * price * TAKER_FEE_RATE
            realised_pnl = 0.0

            if side == "BUY":
                cost = quantity * price + fee
                if cost > self.balance:
                    logger.warning(
                        "Insufficient balance (%.2f) to buy %s @ %.2f (cost %.2f)",
                        self.balance, symbol, price, cost
                    )
                    return None
                self.balance -= cost
                # Open or add to position
                if symbol in self._positions and self._positions[symbol].side == "LONG":
                    pos = self._positions[symbol]
                    total_qty = pos.quantity + quantity
                    pos.entry_price = (pos.entry_price * pos.quantity + price * quantity) / total_qty
                    pos.quantity = total_qty
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol, side="LONG",
                        quantity=quantity, entry_price=price,
                        leverage=self.leverage,
                    )
                self._persist_position(self._positions[symbol])

            elif side == "SELL":
                proceeds = quantity * price - fee
                # If we have a LONG position, close it (partial or full)
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    close_qty = min(quantity, pos.quantity)
                    if pos.side == "LONG":
                        realised_pnl = (price - pos.entry_price) * close_qty
                    else:
                        realised_pnl = (pos.entry_price - price) * close_qty
                    realised_pnl -= fee  # Deduct fee from PnL

                    pos.quantity -= close_qty
                    if pos.quantity <= 1e-9:
                        self._delete_position(symbol)
                    else:
                        self._persist_position(pos)

                self.balance += proceeds + realised_pnl

            self._trade_counter += 1
            trade = Trade(
                trade_id=self._trade_counter,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                fee=fee,
                realised_pnl=realised_pnl,
                balance_after=self.balance,
                note=note,
            )
            self._trades.append(trade)
            self._persist_trade(trade)

            logger.info(
                "PAPER TRADE #%d %s %s %.6f @ %.2f | fee=%.4f | pnl=%.4f | bal=%.2f",
                trade.trade_id, side, symbol, quantity, price,
                fee, realised_pnl, self.balance,
            )
            return trade

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self, prices: Optional[Dict[str, float]] = None) -> dict:
        """Return a dict with current portfolio state and unrealised P&L."""
        prices = prices or {}
        total_unrealised = 0.0
        positions_summary = []

        for sym, pos in self._positions.items():
            cur_price = prices.get(sym) or self.get_price(sym) or pos.entry_price
            upnl = pos.unrealised_pnl(cur_price)
            total_unrealised += upnl
            positions_summary.append({
                "symbol": sym,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": cur_price,
                "unrealised_pnl": round(upnl, 4),
                "leverage": pos.leverage,
            })

        realised = sum(t.realised_pnl for t in self._trades)
        return {
            "balance_usdt": round(self.balance, 4),
            "initial_balance": self.initial_balance,
            "realised_pnl": round(realised, 4),
            "unrealised_pnl": round(total_unrealised, 4),
            "total_pnl": round(realised + total_unrealised, 4),
            "total_trades": self._trade_counter,
            "open_positions": positions_summary,
            "kill_switch_active": self._kill_switch.active,
        }

    def get_trade_history(self) -> List[dict]:
        """Return all trades as a list of dicts (for JSON serialisation)."""
        return [asdict(t) for t in self._trades]

    # ------------------------------------------------------------------
    # Kill-switch passthrough
    # ------------------------------------------------------------------

    def activate_kill_switch(self):
        self._kill_switch.activate()

    def deactivate_kill_switch(self):
        self._kill_switch.deactivate()

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _persist_trade(self, trade: Trade):
        self._db.execute(
            """INSERT INTO trades
               (trade_id, symbol, side, quantity, price, fee,
                realised_pnl, balance_after, timestamp, note)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (trade.trade_id, trade.symbol, trade.side, trade.quantity,
             trade.price, trade.fee, trade.realised_pnl, trade.balance_after,
             trade.timestamp, trade.note),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('balance', ?)",
            (str(self.balance),),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('trade_counter', ?)",
            (str(self._trade_counter),),
        )
        self._db.commit()

    def _persist_position(self, pos: Position):
        self._db.execute(
            """INSERT OR REPLACE INTO positions
               (symbol, side, quantity, entry_price, leverage, opened_at)
               VALUES (?,?,?,?,?,?)""",
            (pos.symbol, pos.side, pos.quantity, pos.entry_price,
             pos.leverage, pos.opened_at),
        )
        self._db.commit()

    def _delete_position(self, symbol: str):
        if symbol in self._positions:
            del self._positions[symbol]
        self._db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        self._db.commit()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._db.close()
