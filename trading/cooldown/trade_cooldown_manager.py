"""
Trade Cooldown Manager for Fewer, Bigger Trades Strategy
Prevents overtrading and ensures quality trade selection
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TradeRecord:
    """Record of a completed trade for cooldown tracking"""

    symbol: str
    timestamp: datetime
    side: str
    size: float
    confidence: float
    pnl: Optional[float] = None


class TradeCooldownManager:
    """
    Manages trade cooldowns to implement fewer, bigger trades strategy
    """

    def __init__(self, logger: logging.Logger = None, cooldown_minutes: int = 15):
        self.logger = logger or logging.getLogger(__name__)
        self.cooldown_minutes = cooldown_minutes
        self.cooldown_seconds = cooldown_minutes * 60

        # Trade tracking
        self.recent_trades: List[TradeRecord] = []
        self.last_trade_time: Dict[str, datetime] = {}
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Persistence file
        self.state_file = Path.home() / ".elvis" / "trade_cooldown_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load previous state
        self._load_state()

        self.logger.info(f"🕐 Trade Cooldown Manager initialized")
        self.logger.info(f"   - Cooldown period: {cooldown_minutes} minutes")
        self.logger.info(f"   - Strategy: FEWER, BIGGER trades")

    def can_trade(self, symbol: str, max_daily_trades: int = 8) -> Dict[str, any]:
        """
        Check if trading is allowed based on cooldown rules

        Args:
            symbol: Trading symbol
            max_daily_trades: Maximum trades per day

        Returns:
            Dict with trade permission and details
        """
        try:
            self._check_daily_reset()

            # Check daily trade limit
            if self.daily_trade_count >= max_daily_trades:
                return {
                    "can_trade": False,
                    "reason": f"Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})",
                    "cooldown_remaining": None,
                    "next_trade_time": None,
                }

            # Check symbol-specific cooldown
            if symbol in self.last_trade_time:
                last_trade = self.last_trade_time[symbol]
                time_since_last = (datetime.now() - last_trade).total_seconds()

                if time_since_last < self.cooldown_seconds:
                    remaining_cooldown = self.cooldown_seconds - time_since_last
                    next_trade_time = datetime.now() + timedelta(
                        seconds=remaining_cooldown
                    )

                    return {
                        "can_trade": False,
                        "reason": f"Symbol {symbol} in cooldown",
                        "cooldown_remaining": remaining_cooldown,
                        "next_trade_time": next_trade_time,
                        "minutes_remaining": remaining_cooldown / 60,
                    }

            # Check global cooldown (any symbol)
            if self.recent_trades:
                last_any_trade = max(
                    trade.timestamp for trade in self.recent_trades[-10:]
                )
                global_cooldown = 300  # 5 minutes minimum between any trades
                time_since_any = (datetime.now() - last_any_trade).total_seconds()

                if time_since_any < global_cooldown:
                    remaining = global_cooldown - time_since_any
                    return {
                        "can_trade": False,
                        "reason": "Global trade cooldown active",
                        "cooldown_remaining": remaining,
                        "next_trade_time": datetime.now()
                        + timedelta(seconds=remaining),
                        "minutes_remaining": remaining / 60,
                    }

            # Trading allowed
            return {
                "can_trade": True,
                "reason": "No cooldown restrictions",
                "daily_trades_remaining": max_daily_trades - self.daily_trade_count,
                "cooldown_remaining": 0,
            }

        except Exception as e:
            self.logger.error(f"Error checking trade permission: {e}")
            return {
                "can_trade": False,
                "reason": f"Cooldown check error: {e}",
                "cooldown_remaining": None,
            }

    def record_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        confidence: float,
        pnl: Optional[float] = None,
    ):
        """
        Record a completed trade for cooldown tracking

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            size: Position size
            confidence: Signal confidence
            pnl: Profit/loss if available
        """
        try:
            self._check_daily_reset()

            # Create trade record
            trade_record = TradeRecord(
                symbol=symbol,
                timestamp=datetime.now(),
                side=side,
                size=size,
                confidence=confidence,
                pnl=pnl,
            )

            # Add to recent trades (keep last 50)
            self.recent_trades.append(trade_record)
            if len(self.recent_trades) > 50:
                self.recent_trades = self.recent_trades[-50:]

            # Update symbol-specific cooldown
            self.last_trade_time[symbol] = datetime.now()

            # Increment daily counter
            self.daily_trade_count += 1

            # Save state
            self._save_state()

            self.logger.info(
                f"📝 Trade recorded: {symbol} {side} | Daily count: {self.daily_trade_count}"
            )
            self.logger.info(
                f"   Next {symbol} trade available in {self.cooldown_minutes} minutes"
            )

        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")

    def get_cooldown_status(self) -> Dict[str, any]:
        """Get current cooldown status for all symbols"""
        try:
            self._check_daily_reset()

            status = {
                "daily_trade_count": self.daily_trade_count,
                "last_reset_date": self.last_reset_date.isoformat(),
                "symbol_cooldowns": {},
                "recent_trades_count": len(self.recent_trades),
                "cooldown_minutes": self.cooldown_minutes,
            }

            # Check each symbol's cooldown
            for symbol, last_time in self.last_trade_time.items():
                time_since = (datetime.now() - last_time).total_seconds()
                remaining = max(0, self.cooldown_seconds - time_since)

                status["symbol_cooldowns"][symbol] = {
                    "last_trade": last_time.isoformat(),
                    "cooldown_remaining_seconds": remaining,
                    "cooldown_remaining_minutes": remaining / 60,
                    "can_trade": remaining == 0,
                }

            return status

        except Exception as e:
            self.logger.error(f"Error getting cooldown status: {e}")
            return {"error": str(e)}

    def force_reset_cooldown(self, symbol: Optional[str] = None):
        """
        Force reset cooldown for testing or emergency trading

        Args:
            symbol: Specific symbol to reset, or None for all
        """
        try:
            if symbol:
                if symbol in self.last_trade_time:
                    del self.last_trade_time[symbol]
                    self.logger.warning(f"⚠️ FORCED cooldown reset for {symbol}")
            else:
                self.last_trade_time.clear()
                self.recent_trades.clear()
                self.daily_trade_count = 0
                self.logger.warning("⚠️ FORCED cooldown reset for ALL symbols")

            self._save_state()

        except Exception as e:
            self.logger.error(f"Error resetting cooldown: {e}")

    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date

            # Clean old symbol cooldowns (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            symbols_to_remove = [
                symbol
                for symbol, last_time in self.last_trade_time.items()
                if last_time < cutoff_time
            ]
            for symbol in symbols_to_remove:
                del self.last_trade_time[symbol]

            self.logger.info(f"🔄 Daily trade counters reset: {current_date}")
            self._save_state()

    def _save_state(self):
        """Save cooldown state to file"""
        try:
            state = {
                "daily_trade_count": self.daily_trade_count,
                "last_reset_date": self.last_reset_date.isoformat(),
                "last_trade_time": {
                    symbol: time.isoformat()
                    for symbol, time in self.last_trade_time.items()
                },
                "recent_trades": [
                    {
                        "symbol": trade.symbol,
                        "timestamp": trade.timestamp.isoformat(),
                        "side": trade.side,
                        "size": trade.size,
                        "confidence": trade.confidence,
                        "pnl": trade.pnl,
                    }
                    for trade in self.recent_trades[-20:]  # Save last 20 trades
                ],
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save cooldown state: {e}")

    def _load_state(self):
        """Load cooldown state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                self.daily_trade_count = state.get("daily_trade_count", 0)

                # Load last reset date
                reset_date_str = state.get("last_reset_date")
                if reset_date_str:
                    self.last_reset_date = datetime.fromisoformat(reset_date_str).date()

                # Load last trade times
                last_trade_times = state.get("last_trade_time", {})
                for symbol, time_str in last_trade_times.items():
                    self.last_trade_time[symbol] = datetime.fromisoformat(time_str)

                # Load recent trades
                recent_trades_data = state.get("recent_trades", [])
                self.recent_trades = [
                    TradeRecord(
                        symbol=trade["symbol"],
                        timestamp=datetime.fromisoformat(trade["timestamp"]),
                        side=trade["side"],
                        size=trade["size"],
                        confidence=trade["confidence"],
                        pnl=trade.get("pnl"),
                    )
                    for trade in recent_trades_data
                ]

                self.logger.info(
                    f"📁 Loaded cooldown state: {self.daily_trade_count} daily trades"
                )

        except Exception as e:
            self.logger.warning(f"Failed to load cooldown state: {e}")


if __name__ == "__main__":
    # Test the cooldown manager
    logger = logging.getLogger("TradeCooldownManager")
    logging.basicConfig(level=logging.INFO)

    cooldown_mgr = TradeCooldownManager(logger, cooldown_minutes=15)

    # Test trade permission
    permission = cooldown_mgr.can_trade("BTCUSDT", max_daily_trades=8)
    print(f"Can trade BTCUSDT: {permission}")

    # Record a test trade
    if permission["can_trade"]:
        cooldown_mgr.record_trade("BTCUSDT", "BUY", 0.001, 0.87, 5.0)
        print("Trade recorded")

        # Check cooldown after trade
        new_permission = cooldown_mgr.can_trade("BTCUSDT")
        print(f"Can trade after recording: {new_permission}")

    # Get status
    status = cooldown_mgr.get_cooldown_status()
    print(f"Cooldown status: {status}")
