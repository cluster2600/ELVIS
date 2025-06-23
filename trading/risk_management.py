import logging
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from trading.execution.base_executor import BaseExecutor
from core.metrics.performance_monitor import PerformanceMonitor
import ta

class RiskManager:
    """
    The RiskManager is responsible for managing risk across all open positions.
    It handles trailing stop-losses and partial take-profits.
    """

    def __init__(self, executor: BaseExecutor, logger: logging.Logger = None, performance_monitor: PerformanceMonitor = None):
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
        self.stop_loss_pct = 0.01
        self.take_profit_pct = 0.03
        self.last_trade_time = None
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.daily_profit_target_usd = 1000.0
        self.daily_loss_limit_usd = -500.0
        self.cooldown_period = 3600.0
        self.max_trades_per_day = 5

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
            pnl = position.get('pnl', 0.0)
            # Update realized PnL when position is closed
            self.realized_pnl += pnl
            self.performance_monitor.add_return(pnl)
            self.logger.info(f"Removed position {symbol} from RiskManager. Realized PnL: {self.realized_pnl:.2f}")

    def manage_positions(self, data: Dict[str, pd.DataFrame] = None):
        """
        Iterate through all open positions and manage their risk.
        This method should be called on each price tick.
        """
        current_unrealized_pnl = 0.0
        for symbol, position in list(self.open_positions.items()):
            current_price = self.executor.get_current_price(symbol)
            
            # Calculate unrealized PnL
            entry_price = position.get('entry_price', current_price)
            quantity = position.get('quantity', 0)
            pnl = (current_price - entry_price) * quantity
            position['pnl'] = pnl
            current_unrealized_pnl += pnl

            # Manage trailing stop-loss
            self._manage_trailing_stop(symbol, position, current_price)
            
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


    def _manage_trailing_stop(self, symbol: str, position: Dict[str, Any], current_price: float):
        """
        Manage the trailing stop-loss for a single position.
        """
        if 'trailing_stop' not in position:
            return

        trailing_stop_info = position['trailing_stop']
        stop_price = trailing_stop_info.get('stop_price')

        if current_price < stop_price:
            self.logger.info(f"Trailing stop triggered for {symbol} at {current_price}")
            self.executor.execute_sell(symbol, position['quantity'])
            self.remove_position(symbol)
        else:
            # Update the stop price if the current price has moved in our favor
            new_stop_price = current_price - trailing_stop_info['trail_distance']
            if new_stop_price > stop_price:
                trailing_stop_info['stop_price'] = new_stop_price
                self.logger.info(f"Updated trailing stop for {symbol} to {new_stop_price}")

    def _manage_partial_take_profit(self, symbol: str, position: Dict[str, Any], current_price: float):
        """
        Manage the partial take-profit for a single position.
        """
        if 'partial_take_profits' not in position:
            return

        for i, tp in enumerate(position['partial_take_profits']):
            if not tp.get('executed') and current_price >= tp['price']:
                self.logger.info(f"Partial take-profit {i+1} triggered for {symbol} at {current_price}")
                self.executor.execute_partial_take_profit(
                    symbol,
                    position['quantity'],
                    tp['percentage'],
                    tp['price']
                )
                tp['executed'] = True

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate the Value at Risk (VaR) for the portfolio.
        """
        if not self.performance_monitor.returns:
            return 0.0

        returns = np.array(self.performance_monitor.returns)
        return np.percentile(returns, 100 * (1 - confidence_level))

    def calculate_cross_pair_correlation(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the close prices of all symbols.
        """
        close_prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
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
                    if correlation > 0.8: # High correlation threshold
                        # Reduce position size by a factor of the correlation
                        position['quantity'] *= (1 - correlation * 0.5)
                        self.logger.info(f"Reduced position size for {symbol} due to high correlation with {other_symbol}")

    def check_max_drawdown(self, max_drawdown_threshold: float = 0.2):
        """
        Check if the maximum drawdown has been exceeded and take action.
        """
        drawdown = self.performance_monitor.calculate_rolling_drawdown()
        if drawdown < -max_drawdown_threshold:
            self.logger.warning(f"Maximum drawdown of {max_drawdown_threshold:.2%} exceeded. Current drawdown: {drawdown:.2%}")
            # Liquidate all positions
            for symbol, position in list(self.open_positions.items()):
                self.executor.execute_sell(symbol, position['quantity'])
                self.remove_position(symbol)
            self.logger.warning("All positions liquidated due to excessive drawdown.")

    def check_circuit_breakers(self, data: pd.DataFrame, volatility_threshold: float = 0.1):
        """
        Check for extreme market conditions and halt trading if necessary.
        """
        # Example: Check for extreme volatility
        atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
        if atr.iloc[-1] > volatility_threshold:
            self.logger.warning(f"Extreme volatility detected. Halting trading. ATR: {atr.iloc[-1]}")
            for symbol, position in list(self.open_positions.items()):
                self.executor.execute_sell(symbol, position['quantity'])
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

        portfolio_value = sum(pos.get('quantity', 0) * self.executor.get_current_price(s) for s, pos in self.open_positions.items())
        if portfolio_value == 0:
            return {}

        for symbol, position in self.open_positions.items():
            # Simplified example: risk contribution is proportional to position size
            position_value = position.get('quantity', 0) * self.executor.get_current_price(symbol)
            position_risk[symbol] = (position_value / portfolio_value) * total_risk
                
        return position_risk

    def check_capital(self, capital: float) -> bool:
        """
        Check if there is sufficient capital to continue trading.
        """
        return capital > 1000.0

    def calculate_stop_loss(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the stop loss price.
        """
        return entry_price * (1 - self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the take profit price.
        """
        return max(entry_price + (3 * volatility), entry_price * (1 + self.take_profit_pct))

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
        if self.last_trade_time and (datetime.now() - self.last_trade_time).total_seconds() < self.cooldown_period:
            self.logger.warning("Cooldown period has not elapsed.")
            return False
        return True
