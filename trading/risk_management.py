import logging
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import ta

from core.metrics.performance_monitor import PerformanceMonitor
from trading.execution.base_executor import BaseExecutor


def kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    cap: float = 0.2,
) -> float:
    """
    Compute the Kelly-criterion optimal fraction of capital to risk.

    Uses the standard formula ``f* = W - (1 - W) / R`` where ``W`` is the
    probability of a winning trade and ``R`` is the payoff ratio (average
    win / average loss). The result is clamped to ``[0, cap]`` so an edge is
    never sized negative (skip the trade instead) and never exceeds a safety
    cap (fractional Kelly).

    Args:
        win_rate: Probability of a winning trade, in [0, 1].
        payoff_ratio: Average win divided by average loss (must be > 0).
        cap: Upper bound on the returned fraction (default 0.2 = 20%).

    Returns:
        The capped Kelly fraction in [0, cap].
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    if payoff_ratio <= 0.0:
        raise ValueError("payoff_ratio must be > 0")
    if cap < 0.0:
        raise ValueError("cap must be >= 0")

    raw = win_rate - (1.0 - win_rate) / payoff_ratio
    return max(0.0, min(raw, cap))


class RiskManager:
    """
    The RiskManager is responsible for managing risk across all open positions.
    It handles trailing stop-losses and partial take-profits.
    """

    def __init__(
        self,
        executor: BaseExecutor,
        logger: logging.Logger = None,
        performance_monitor: PerformanceMonitor = None,
    ):
        """
        Initialize the RiskManager.

        Args:
            executor (BaseExecutor): The executor to use for placing orders.
            logger (logging.Logger): The logger to use.
            performance_monitor (PerformanceMonitor): The performance monitor instance.
        """
        self.executor = executor
        self.logger = logger or logging.getLogger(__name__)
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        # DYNAMIC RISK MANAGEMENT PARAMETERS
        self.base_stop_loss_pct = 0.01  # Base 1% stop loss
        self.base_take_profit_pct = 0.02  # Base 2% take profit
        self.dynamic_stop_loss_pct = 0.01
        self.dynamic_take_profit_pct = 0.02
        self.last_trade_time = None
        self.trades_today = 0
        self.daily_pnl = 0.0

        # AGGRESSIVE TARGETS - Optimize for 100x leverage profits
        self.daily_profit_target_usd = 5000.0  # Aggressive target with 100x leverage
        self.daily_loss_limit_usd = -1000.0  # Higher tolerance for 100x volatility
        self.cooldown_period = 0  # No cooldown - maximum speed for scalping profits
        self.max_trades_per_day = 1000  # High frequency for 100x leverage opportunities

        # DYNAMIC RISK FACTORS
        self.volatility_factor = 1.0  # Adjusts based on market volatility
        self.performance_factor = 1.0  # Adjusts based on recent performance
        self.leverage_target = 100.0  # Maximum 100x leverage for profit amplification
        self.market_sentiment = "neutral"  # bullish, bearish, neutral
        self.risk_appetite = "aggressive"  # conservative, moderate, aggressive

        # PERFORMANCE TRACKING
        self.recent_trades = []  # Track last 10 trades for dynamic adjustment
        self.win_rate = 0.5  # Current win rate
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def add_position(self, symbol: str, position_data: Dict[str, Any]):
        """
        Add a new position to be managed by the RiskManager.

        Args:
            symbol (str): The symbol of the position.
            position_data (Dict[str, Any]): The position data.
        """
        self.open_positions[symbol] = position_data
        self.logger.info(f"Added position {symbol} to RiskManager.")

    def remove_position(self, symbol: str):
        """
        Remove a position from the RiskManager.

        Args:
            symbol (str): The symbol of the position to remove.
        """
        if symbol in self.open_positions:
            position = self.open_positions.pop(symbol)
            pnl = position.get("pnl", 0.0)
            # Update realized PnL when position is closed
            self.realized_pnl += pnl
            self.performance_monitor.add_return(pnl)
            self.logger.info(
                f"Removed position {symbol} from RiskManager. Realized PnL: {self.realized_pnl:.2f}"
            )

    def update_dynamic_parameters(self, market_data: pd.DataFrame = None):
        """
        Dynamically adjust risk parameters based on market conditions and performance.
        """
        try:
            # Update volatility factor
            if market_data is not None and len(market_data) > 14:
                atr = (
                    ta.volatility.AverageTrueRange(
                        market_data["high"],
                        market_data["low"],
                        market_data["close"],
                        window=14,
                    )
                    .average_true_range()
                    .iloc[-1]
                )

                price = market_data["close"].iloc[-1]
                volatility_pct = (atr / price) * 100

                # Adjust volatility factor (lower volatility = tighter stops, higher volatility = wider stops)
                if volatility_pct < 1.0:  # Low volatility
                    self.volatility_factor = 0.7  # Tighter stops
                    self.risk_appetite = "aggressive"
                elif volatility_pct > 3.0:  # High volatility
                    self.volatility_factor = 1.5  # Wider stops
                    self.risk_appetite = "moderate"
                else:
                    self.volatility_factor = 1.0
                    self.risk_appetite = "aggressive"

                self.logger.info(
                    f"📊 Volatility: {volatility_pct:.2f}%, Factor: {self.volatility_factor:.2f}, Appetite: {self.risk_appetite}"
                )

            # Update performance factor based on recent trades
            if len(self.recent_trades) >= 5:
                recent_pnl = [trade["pnl"] for trade in self.recent_trades[-10:]]
                winning_trades = [pnl for pnl in recent_pnl if pnl > 0]
                losing_trades = [pnl for pnl in recent_pnl if pnl < 0]

                self.win_rate = (
                    len(winning_trades) / len(recent_pnl) if recent_pnl else 0.5
                )
                self.avg_win = np.mean(winning_trades) if winning_trades else 0.0
                self.avg_loss = np.mean(losing_trades) if losing_trades else 0.0

                # Adjust performance factor
                if self.win_rate > 0.6:  # Good performance
                    self.performance_factor = 0.8  # Tighter stops, take profits faster
                elif self.win_rate < 0.4:  # Poor performance
                    self.performance_factor = 1.3  # Wider stops, more patient
                else:
                    self.performance_factor = 1.0

                self.logger.info(
                    f"🎯 Win Rate: {self.win_rate:.1%}, Performance Factor: {self.performance_factor:.2f}"
                )

            # Calculate dynamic stop loss and take profit
            self.dynamic_stop_loss_pct = (
                self.base_stop_loss_pct
                * self.volatility_factor
                * self.performance_factor
            )
            self.dynamic_take_profit_pct = (
                self.base_take_profit_pct
                * self.volatility_factor
                * (2 - self.performance_factor)
            )

            # Ensure reasonable thresholds for reduced leverage
            self.dynamic_stop_loss_pct = max(
                0.01, min(0.05, self.dynamic_stop_loss_pct)
            )  # 1% to 5%
            self.dynamic_take_profit_pct = max(
                0.02, min(0.08, self.dynamic_take_profit_pct)
            )  # 2% to 8%

            self.logger.info(
                f"🔧 Dynamic Stop Loss: {self.dynamic_stop_loss_pct:.1%}, Take Profit: {self.dynamic_take_profit_pct:.1%}"
            )

        except Exception as e:
            self.logger.error(f"Error updating dynamic parameters: {e}")

    def enforce_minimum_leverage(self, current_leverage: float) -> float:
        """
        Ensure leverage is at least x50, adjust if necessary.
        """
        if current_leverage < self.leverage_target:
            self.logger.warning(
                f"⚡ Leverage {current_leverage}x below target. Enforcing {self.leverage_target}x"
            )
            return self.leverage_target
        return min(current_leverage, 125.0)  # Cap at 125x for safety

    def calculate_dynamic_position_size(
        self,
        available_capital: float,
        current_price: float,
        signal_confidence: float,
        leverage: float = 50.0,
    ) -> float:
        """
        Calculate position size dynamically based on multiple factors.
        """
        # Ensure minimum leverage
        effective_leverage = self.enforce_minimum_leverage(leverage)

        # Base position size (conservative due to low win rate)
        base_risk = (
            0.01 if self.risk_appetite == "aggressive" else 0.005
        )  # 0.5-1% of capital (REDUCED)

        # Adjust for confidence
        confidence_multiplier = signal_confidence if signal_confidence > 0.5 else 0.5

        # Adjust for volatility
        volatility_multiplier = (
            (2 - self.volatility_factor) if self.volatility_factor < 1.5 else 0.7
        )

        # Adjust for performance
        performance_multiplier = (
            1.2 if self.win_rate > 0.6 else 0.8 if self.win_rate < 0.4 else 1.0
        )

        # Calculate final position size
        risk_amount = (
            available_capital
            * base_risk
            * confidence_multiplier
            * volatility_multiplier
            * performance_multiplier
        )
        position_value = risk_amount * effective_leverage
        position_size = position_value / current_price

        # Ensure reasonable bounds
        min_size = 0.0001  # Minimum BTC
        max_size = (
            available_capital * 0.8 * effective_leverage
        ) / current_price  # 80% of capital max

        final_size = max(min_size, min(position_size, max_size))

        self.logger.info(
            f"💰 Dynamic sizing: {final_size:.6f} BTC (Lev: {effective_leverage}x, Conf: {signal_confidence:.1%}, Risk: {base_risk:.1%})"
        )
        return final_size

    def calculate_kelly_position_size(
        self,
        available_capital: float,
        current_price: float,
        payoff_ratio: float,
        win_rate: float = None,
        leverage: float = 50.0,
        cap: float = 0.2,
    ) -> float:
        """
        Size a position using the Kelly criterion.

        This is an opt-in alternative to ``calculate_dynamic_position_size``.
        The fraction of capital to risk is derived from ``kelly_fraction`` and
        the resulting notional is scaled by ``leverage`` and converted to a
        BTC quantity at ``current_price``.

        Args:
            available_capital: Capital available to allocate.
            current_price: Current asset price (must be > 0).
            payoff_ratio: Average win / average loss (must be > 0).
            win_rate: Win probability; defaults to the manager's tracked
                ``self.win_rate`` when not provided.
            leverage: Leverage applied to the risked notional.
            cap: Upper bound on the Kelly fraction (fractional Kelly).

        Returns:
            Position size in BTC.
        """
        if current_price <= 0.0:
            raise ValueError("current_price must be > 0")

        w = self.win_rate if win_rate is None else win_rate
        fraction = kelly_fraction(w, payoff_ratio, cap=cap)

        effective_leverage = self.enforce_minimum_leverage(leverage)
        position_value = available_capital * fraction * effective_leverage
        position_size = position_value / current_price

        self.logger.info(
            f"💰 Kelly sizing: {position_size:.6f} BTC "
            f"(f*: {fraction:.3f}, W: {w:.1%}, R: {payoff_ratio:.2f}, "
            f"Lev: {effective_leverage}x)"
        )
        return position_size

    def manage_positions(self, data: Dict[str, pd.DataFrame] = None):
        """
        Iterate through all open positions and manage their risk.
        This method should be called on each price tick.
        """
        # Update dynamic parameters first
        if data:
            primary_symbol = list(data.keys())[0]
            self.update_dynamic_parameters(data[primary_symbol])

        current_unrealized_pnl = 0.0
        for symbol, position in list(self.open_positions.items()):
            current_price = self.executor.get_current_price(symbol)

            # Calculate unrealized PnL
            entry_price = position.get("entry_price", current_price)
            quantity = position.get("quantity", 0)
            side = position.get("side", "BUY")

            # Calculate PnL based on position side
            if side.upper() == "BUY":
                pnl = (current_price - entry_price) * quantity
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL/SHORT
                pnl = (entry_price - current_price) * quantity
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            position["pnl"] = pnl
            position["pnl_pct"] = pnl_pct
            current_unrealized_pnl += pnl

            # DYNAMIC STOP LOSS - Based on current parameters
            if pnl_pct < -(self.dynamic_stop_loss_pct * 100):
                self.logger.warning(
                    f"🛑 DYNAMIC STOP LOSS: {symbol} at {pnl_pct:.2f}% (threshold: {self.dynamic_stop_loss_pct:.1%})"
                )
                self._execute_emergency_close(
                    symbol, position, current_price, "STOP_LOSS"
                )
                continue

            # DYNAMIC TAKE PROFIT - Based on current parameters
            if pnl_pct > (self.dynamic_take_profit_pct * 100):
                self.logger.info(
                    f"💰 DYNAMIC TAKE PROFIT: {symbol} at {pnl_pct:.2f}% (threshold: {self.dynamic_take_profit_pct:.1%})"
                )
                self._execute_emergency_close(
                    symbol, position, current_price, "TAKE_PROFIT"
                )
                continue

            # Manage trailing stop-loss
            self._manage_dynamic_trailing_stop(symbol, position, current_price)

            # Manage partial take-profit
            self._manage_partial_take_profit(symbol, position, current_price)

        self.unrealized_pnl = current_unrealized_pnl

        if data:
            self.apply_correlation_based_position_limits(data)

        self.check_max_drawdown()

        if data and not data.empty:
            # Assuming data contains ohlcv for the primary symbol
            primary_symbol = list(data.keys())[0]
            self.check_circuit_breakers(data[primary_symbol])

    def _execute_emergency_close(
        self, symbol: str, position: Dict[str, Any], current_price: float, reason: str
    ):
        """
        Execute emergency position close with dynamic parameters.
        """
        try:
            side = position.get("side", "BUY")
            quantity = abs(position.get("quantity", 0))

            # Determine close direction
            close_side = "SELL" if side.upper() == "BUY" else "BUY"

            success = self.executor.place_order(
                symbol, close_side.lower(), quantity, current_price
            )
            if success:
                pnl = position.get("pnl", 0)
                self.logger.info(
                    f"✅ {reason}: Closed {side} position of {quantity:.6f} {symbol} with PnL: ${pnl:.2f}"
                )

                # Track trade for dynamic adjustment
                trade_record = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": position.get("entry_price"),
                    "exit_price": current_price,
                    "pnl": pnl,
                    "reason": reason,
                }
                self.recent_trades.append(trade_record)

                # Keep only last 20 trades
                if len(self.recent_trades) > 20:
                    self.recent_trades.pop(0)

                self.remove_position(symbol)
            else:
                self.logger.error(f"❌ Failed to execute {reason} for {symbol}")

        except Exception as e:
            self.logger.error(f"Error executing emergency close for {symbol}: {e}")

    def _manage_dynamic_trailing_stop(
        self, symbol: str, position: Dict[str, Any], current_price: float
    ):
        """
        Manage the trailing stop-loss for a single position.
        """
        # Initialize dynamic trailing stop if not present
        if "dynamic_trailing_stop" not in position:
            entry_price = position.get("entry_price", current_price)
            position["dynamic_trailing_stop"] = {
                "stop_price": entry_price * (1 - self.dynamic_stop_loss_pct),
                "trail_distance_pct": self.dynamic_stop_loss_pct,
                "highest_price": current_price,
                "activated": False,
            }
            return

        trailing_info = position["dynamic_trailing_stop"]
        side = position.get("side", "BUY")

        if side.upper() == "BUY":
            # For BUY positions, trail upward
            if current_price > trailing_info["highest_price"]:
                trailing_info["highest_price"] = current_price

                # Activate trailing after 1% profit
                if (
                    not trailing_info["activated"]
                    and current_price > position.get("entry_price", 0) * 1.01
                ):
                    trailing_info["activated"] = True
                    self.logger.info(
                        f"🎯 Trailing stop activated for {symbol} at 1% profit"
                    )

                # Update stop price if activated
                if trailing_info["activated"]:
                    new_stop_price = current_price * (
                        1 - trailing_info["trail_distance_pct"]
                    )
                    if new_stop_price > trailing_info["stop_price"]:
                        trailing_info["stop_price"] = new_stop_price
                        self.logger.debug(
                            f"📈 Updated trailing stop for {symbol}: ${new_stop_price:.2f}"
                        )

            # Check if stop hit
            if (
                trailing_info["activated"]
                and current_price <= trailing_info["stop_price"]
            ):
                self.logger.info(
                    f"🛑 Dynamic trailing stop triggered for {symbol} at ${current_price:.2f}"
                )
                self._execute_emergency_close(
                    symbol, position, current_price, "TRAILING_STOP"
                )

        else:  # SHORT position
            if current_price < trailing_info["highest_price"]:
                trailing_info["highest_price"] = current_price

                if (
                    not trailing_info["activated"]
                    and current_price < position.get("entry_price", 0) * 0.99
                ):
                    trailing_info["activated"] = True
                    self.logger.info(
                        f"🎯 Trailing stop activated for SHORT {symbol} at 1% profit"
                    )

                if trailing_info["activated"]:
                    new_stop_price = current_price * (
                        1 + trailing_info["trail_distance_pct"]
                    )
                    if new_stop_price < trailing_info["stop_price"]:
                        trailing_info["stop_price"] = new_stop_price
                        self.logger.debug(
                            f"📉 Updated trailing stop for SHORT {symbol}: ${new_stop_price:.2f}"
                        )

            if (
                trailing_info["activated"]
                and current_price >= trailing_info["stop_price"]
            ):
                self.logger.info(
                    f"🛑 Dynamic trailing stop triggered for SHORT {symbol} at ${current_price:.2f}"
                )
                self._execute_emergency_close(
                    symbol, position, current_price, "TRAILING_STOP"
                )

    def _manage_trailing_stop(
        self, symbol: str, position: Dict[str, Any], current_price: float
    ):
        """
        Legacy trailing stop method - now redirects to dynamic version.
        """
        return self._manage_dynamic_trailing_stop(symbol, position, current_price)

    def _manage_partial_take_profit(
        self, symbol: str, position: Dict[str, Any], current_price: float
    ):
        """
        Manage the partial take-profit for a single position.
        """
        if "partial_take_profits" not in position:
            return

        for i, tp in enumerate(position["partial_take_profits"]):
            if not tp.get("executed") and current_price >= tp["price"]:
                self.logger.info(
                    f"Partial take-profit {i+1} triggered for {symbol} at {current_price}"
                )
                self.executor.execute_partial_take_profit(
                    symbol, position["quantity"], tp["percentage"], tp["price"]
                )
                tp["executed"] = True

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate the Value at Risk (VaR) for the portfolio.
        """
        if not self.performance_monitor.returns:
            return 0.0

        returns = np.array(self.performance_monitor.returns)
        return np.percentile(returns, 100 * (1 - confidence_level))

    def calculate_cross_pair_correlation(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the close prices of all symbols.
        """
        close_prices = pd.DataFrame(
            {symbol: df["close"] for symbol, df in data.items()}
        )
        return close_prices.corr()

    def apply_correlation_based_position_limits(self, data: Dict[str, pd.DataFrame]):
        """
        Adjust position sizes based on the correlation between assets.
        """
        correlation_matrix = self.calculate_cross_pair_correlation(data)

        for symbol, position in self.open_positions.items():
            # Example: Reduce position size if highly correlated with other positions
            for other_symbol, other_position in self.open_positions.items():
                if symbol != other_symbol:
                    correlation = correlation_matrix.loc[symbol, other_symbol]
                    if correlation > 0.8:  # High correlation threshold
                        # Reduce position size by a factor of the correlation
                        position["quantity"] *= 1 - correlation * 0.5
                        self.logger.info(
                            f"Reduced position size for {symbol} due to high correlation with {other_symbol}"
                        )

    def check_max_drawdown(self, max_drawdown_threshold: float = 0.2):
        """
        Check if the maximum drawdown has been exceeded and take action.
        """
        drawdown = self.performance_monitor.calculate_rolling_drawdown()
        if drawdown < -max_drawdown_threshold:
            self.logger.warning(
                f"Maximum drawdown of {max_drawdown_threshold:.2%} exceeded. Current drawdown: {drawdown:.2%}"
            )
            # Liquidate all positions
            for symbol, position in list(self.open_positions.items()):
                self.executor.execute_sell(symbol, position["quantity"])
                self.remove_position(symbol)
            self.logger.warning("All positions liquidated due to excessive drawdown.")

    def check_circuit_breakers(
        self, data: pd.DataFrame, volatility_threshold: float = 0.1
    ):
        """
        Check for extreme market conditions and halt trading if necessary.
        """
        # Example: Check for extreme volatility
        atr = ta.volatility.AverageTrueRange(
            data["high"], data["low"], data["close"]
        ).average_true_range()
        if atr.iloc[-1] > volatility_threshold:
            self.logger.warning(
                f"Extreme volatility detected. Halting trading. ATR: {atr.iloc[-1]}"
            )
            for symbol, position in list(self.open_positions.items()):
                self.executor.execute_sell(symbol, position["quantity"])
                self.remove_position(symbol)
            self.logger.warning("All positions liquidated due to circuit breaker.")
            # In a real application, you would also want to stop placing new trades.
            # This could be done by setting a flag that is checked by the trading loop.

    def get_position_level_risk(self) -> Dict[str, float]:
        """
        Calculate the risk contribution of each position.
        """
        position_risk = {}
        total_risk = self.calculate_var()

        if total_risk == 0:
            return {}

        portfolio_value = sum(
            pos.get("quantity", 0) * self.executor.get_current_price(s)
            for s, pos in self.open_positions.items()
        )
        if portfolio_value == 0:
            return {}

        for symbol, position in self.open_positions.items():
            # Simplified example: risk contribution is proportional to position size
            position_value = position.get(
                "quantity", 0
            ) * self.executor.get_current_price(symbol)
            position_risk[symbol] = (position_value / portfolio_value) * total_risk

        return position_risk

    def check_capital(self, capital: float) -> bool:
        """
        Check if there is sufficient capital to continue trading.
        """
        return capital > 1000.0

    def calculate_stop_loss(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the dynamic stop loss price.
        """
        return entry_price * (1 - self.dynamic_stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the dynamic take profit price.
        """
        return max(
            entry_price + (3 * volatility),
            entry_price * (1 + self.dynamic_take_profit_pct),
        )

    def check_new_day(self) -> bool:
        """
        Check if a new day has started.
        """
        if self.last_trade_time and self.last_trade_time.date() < datetime.now().date():
            self.reset_daily_stats()
            return True
        return False

    def reset_daily_stats(self):
        """
        Reset the daily statistics.
        """
        self.trades_today = 0
        self.daily_pnl = 0.0

    def update_trade_stats(self, pnl: float):
        """
        Update the trade statistics.
        """
        self.trades_today += 1
        self.daily_pnl += pnl
        self.last_trade_time = datetime.now()

    def check_trade_limits(self) -> bool:
        """
        Check if any trade limits have been reached.
        """
        if self.trades_today >= self.max_trades_per_day:
            self.logger.warning("Max trades per day reached.")
            return False
        if self.daily_pnl >= self.daily_profit_target_usd:
            self.logger.warning("Daily profit target reached.")
            return False
        if self.daily_pnl <= self.daily_loss_limit_usd:
            self.logger.warning("Daily loss limit reached.")
            return False

        # COOLDOWN COMPLETELY REMOVED - No delays whatsoever

        return True
