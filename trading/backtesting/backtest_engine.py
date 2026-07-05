"""
Backtesting Engine for ELVIS Trading Bot
Provides historical strategy testing with realistic constraints
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade in backtesting"""

    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    fees: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    status: str = "OPEN"  # 'OPEN', 'CLOSED', 'STOPPED_OUT'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""

    initial_balance: float = 10000.0
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% of balance
    max_open_trades: int = 3
    use_leverage: bool = False
    max_leverage: float = 1.0
    risk_per_trade: float = 0.02  # 2%
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    allow_shorting: bool = False


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset backtesting state"""
        self.balance = self.config.initial_balance
        self.equity = self.balance
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_signals: List[Dict[str, Any]] = []
        self.current_time: Optional[datetime] = None
        self.stats: Dict[str, Any] = {}

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            data: Historical price data with columns: timestamp, open, high, low, close, volume
            strategy: Strategy object with generate_signal method
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Backtest results and statistics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Prepare data
        data = self._prepare_data(data, start_date, end_date)

        if len(data) < 50:
            logger.error("Insufficient data for backtesting")
            return {"error": "Insufficient data"}

        # Initialize strategy
        strategy.reset()

        # Run backtest loop
        for idx, row in data.iterrows():
            self.current_time = row["timestamp"]

            # Update open trades with current prices
            self._update_open_trades(row)

            # Check stop loss and take profit
            self._check_exits(row)

            # Generate trading signal
            signal = strategy.generate_signal(data.loc[:idx])

            if signal:
                self._process_signal(signal, row)

            # Update equity
            self._update_equity(row)

            # Record equity curve
            self.equity_curve.append((self.current_time, self.equity))

        # Close all remaining trades
        self._close_all_trades(data.iloc[-1])

        # Calculate statistics
        self.stats = self._calculate_statistics()

        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")

        return {
            "config": self.config.__dict__,
            "trades": [self._trade_to_dict(t) for t in self.trades],
            "equity_curve": self.equity_curve,
            "statistics": self.stats,
        }

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Prepare and filter data for backtesting"""
        data = data.copy()

        # Ensure timestamp column
        if "timestamp" not in data.columns:
            data["timestamp"] = pd.to_datetime(data.index)
        else:
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Filter by date range
        if start_date:
            data = data[data["timestamp"] >= start_date]
        if end_date:
            data = data[data["timestamp"] <= end_date]

        # Sort by timestamp
        data = data.sort_values("timestamp")

        # Reset index
        data = data.reset_index(drop=True)

        return data

    def _update_open_trades(self, row: pd.Series):
        """Update open trades with current market data"""
        current_price = row["close"]

        for trade in self.open_trades:
            if trade.side == "LONG":
                trade.pnl = (current_price - trade.entry_price) * trade.quantity
            else:  # SHORT
                trade.pnl = (trade.entry_price - current_price) * trade.quantity

            trade.pnl -= trade.fees
            trade.pnl_percentage = (
                trade.pnl / (trade.entry_price * trade.quantity) * 100
            )

    def _check_exits(self, row: pd.Series):
        """Check stop loss and take profit conditions"""
        current_price = row["close"]
        high_price = row["high"]
        low_price = row["low"]

        trades_to_close = []

        for trade in self.open_trades:
            should_close = False
            exit_price = current_price

            if trade.side == "LONG":
                # Check stop loss
                if trade.stop_loss and low_price <= trade.stop_loss:
                    should_close = True
                    exit_price = trade.stop_loss
                    trade.status = "STOPPED_OUT"
                # Check take profit
                elif trade.take_profit and high_price >= trade.take_profit:
                    should_close = True
                    exit_price = trade.take_profit

            else:  # SHORT
                # Check stop loss
                if trade.stop_loss and high_price >= trade.stop_loss:
                    should_close = True
                    exit_price = trade.stop_loss
                    trade.status = "STOPPED_OUT"
                # Check take profit
                elif trade.take_profit and low_price <= trade.take_profit:
                    should_close = True
                    exit_price = trade.take_profit

            if should_close:
                trades_to_close.append((trade, exit_price))

        # Close trades
        for trade, exit_price in trades_to_close:
            self._close_trade(trade, exit_price, row["timestamp"])

    def _process_signal(self, signal: Dict[str, Any], row: pd.Series):
        """Process trading signal"""
        signal_type = signal.get("type")
        confidence = signal.get("confidence", 0.5)

        # Check if we can open new trades
        if len(self.open_trades) >= self.config.max_open_trades:
            return

        # Determine trade side
        if signal_type == "BUY" and (self.config.allow_shorting or True):
            side = "LONG"
        elif signal_type == "SELL" and self.config.allow_shorting:
            side = "SHORT"
        else:
            return

        # Calculate position size
        position_size = self._calculate_position_size(row["close"], confidence)

        if position_size > 0:
            self._open_trade(
                side, row["close"], position_size, row["timestamp"], signal
            )

    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size based on risk management"""
        # Base position size on risk per trade
        risk_amount = self.equity * self.config.risk_per_trade

        # Adjust by confidence
        risk_amount *= confidence

        # Calculate quantity based on stop loss
        stop_distance = price * self.config.stop_loss_pct
        quantity = risk_amount / stop_distance

        # Apply maximum position size limit
        max_quantity = (self.equity * self.config.max_position_size) / price
        quantity = min(quantity, max_quantity)

        # Apply leverage if enabled
        if self.config.use_leverage:
            quantity *= self.config.max_leverage

        return quantity

    def _open_trade(
        self,
        side: str,
        price: float,
        quantity: float,
        timestamp: datetime,
        signal: Dict[str, Any],
    ):
        """Open a new trade"""
        # Calculate fees
        fees = price * quantity * self.config.taker_fee

        # Check if we have enough balance
        required_balance = price * quantity + fees
        if required_balance > self.balance:
            return

        # Calculate stop loss and take profit
        if side == "LONG":
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:  # SHORT
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)

        # Create trade
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side=side,
            fees=fees,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=signal,
        )

        # Update balance
        self.balance -= required_balance

        # Add to trades
        self.trades.append(trade)
        self.open_trades.append(trade)

        logger.debug(f"Opened {side} trade at {price} with quantity {quantity}")

    def _close_trade(self, trade: Trade, exit_price: float, timestamp: datetime):
        """Close a trade"""
        trade.exit_time = timestamp
        trade.exit_price = exit_price

        # Apply slippage
        if trade.side == "LONG":
            exit_price *= 1 - self.config.slippage
        else:  # SHORT
            exit_price *= 1 + self.config.slippage

        # Calculate fees
        exit_fees = exit_price * trade.quantity * self.config.taker_fee
        trade.fees += exit_fees

        # Calculate PnL
        if trade.side == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.pnl -= trade.fees
        trade.pnl_percentage = trade.pnl / (trade.entry_price * trade.quantity) * 100

        # Update balance
        self.balance += exit_price * trade.quantity - exit_fees

        # Update status
        if trade.status == "OPEN":
            trade.status = "CLOSED"

        # Remove from open trades
        self.open_trades.remove(trade)

        logger.debug(
            f"Closed {trade.side} trade at {exit_price} with PnL: {trade.pnl:.2f}"
        )

    def _close_all_trades(self, row: pd.Series):
        """Close all remaining open trades"""
        trades_to_close = list(self.open_trades)
        for trade in trades_to_close:
            self._close_trade(trade, row["close"], row["timestamp"])

    def _update_equity(self, row: pd.Series):
        """Update current equity value"""
        # Start with cash balance
        self.equity = self.balance

        # Add unrealized PnL from open trades
        current_price = row["close"]
        for trade in self.open_trades:
            if trade.side == "LONG":
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:  # SHORT
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

            self.equity += trade.entry_price * trade.quantity + unrealized_pnl

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "total_return_pct": 0.0,
            }

        # Filter closed trades
        closed_trades = [
            t for t in self.trades if t.status in ["CLOSED", "STOPPED_OUT"]
        ]

        # Calculate basic statistics
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))

        # Calculate metrics
        stats = {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(closed_trades) if closed_trades else 0
            ),
            "profit_factor": (
                total_profit / total_loss if total_loss > 0 else float("inf")
            ),
            "average_win": total_profit / len(winning_trades) if winning_trades else 0,
            "average_loss": total_loss / len(losing_trades) if losing_trades else 0,
            "largest_win": (
                max([t.pnl for t in winning_trades]) if winning_trades else 0
            ),
            "largest_loss": min([t.pnl for t in losing_trades]) if losing_trades else 0,
            "total_fees": sum(t.fees for t in closed_trades),
            "total_return": self.equity - self.config.initial_balance,
            "total_return_pct": (
                (self.equity - self.config.initial_balance)
                / self.config.initial_balance
                * 100
            ),
        }

        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = pd.Series(
                [
                    self.equity_curve[i][1] / self.equity_curve[i - 1][1] - 1
                    for i in range(1, len(self.equity_curve))
                ]
            )
            stats["sharpe_ratio"] = (
                np.sqrt(252) * returns.mean() / returns.std()
                if returns.std() > 0
                else 0
            )
        else:
            stats["sharpe_ratio"] = 0

        # Calculate max drawdown
        equity_values = [e[1] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            stats["max_drawdown"] = max_dd * 100
        else:
            stats["max_drawdown"] = 0

        return stats

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "side": trade.side,
            "fees": trade.fees,
            "pnl": trade.pnl,
            "pnl_percentage": trade.pnl_percentage,
            "status": trade.status,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "metadata": trade.metadata,
        }

    def save_results(self, filepath: str):
        """Save backtest results to file"""
        results = {
            "config": self.config.__dict__,
            "trades": [self._trade_to_dict(t) for t in self.trades],
            "equity_curve": [(t[0].isoformat(), t[1]) for t in self.equity_curve],
            "statistics": self.stats,
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Backtest results saved to {filepath}")

    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Equity curve
            equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
            axes[0, 0].plot(equity_df["timestamp"], equity_df["equity"])
            axes[0, 0].set_title("Equity Curve")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Equity ($)")

            # Returns distribution
            returns = [
                t.pnl_percentage
                for t in self.trades
                if t.status in ["CLOSED", "STOPPED_OUT"]
            ]
            if returns:
                axes[0, 1].hist(returns, bins=30, alpha=0.7)
                axes[0, 1].set_title("Returns Distribution")
                axes[0, 1].set_xlabel("Return (%)")
                axes[0, 1].set_ylabel("Frequency")

            # Cumulative returns
            cumulative_returns = []
            cumulative = 0
            for t in sorted(self.trades, key=lambda x: x.exit_time or x.entry_time):
                if t.status in ["CLOSED", "STOPPED_OUT"]:
                    cumulative += t.pnl
                    cumulative_returns.append(cumulative)

            if cumulative_returns:
                axes[1, 0].plot(range(len(cumulative_returns)), cumulative_returns)
                axes[1, 0].set_title("Cumulative PnL")
                axes[1, 0].set_xlabel("Trade Number")
                axes[1, 0].set_ylabel("Cumulative PnL ($)")

            # Statistics text
            stats_text = "\n".join(
                [
                    f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in self.stats.items()
                ]
            )
            axes[1, 1].text(
                0.1,
                0.5,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="center",
            )
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Backtest Statistics")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Backtest plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available. Cannot plot results.")
