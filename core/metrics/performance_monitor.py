import numpy as np
import pandas as pd


class PerformanceMonitor:
    """
    Calculates and tracks performance metrics such as Sharpe ratio and drawdown.
    """

    def __init__(self, risk_free_rate=0.0, window=252):
        """
        Initialize the PerformanceMonitor.

        Args:
            risk_free_rate (float): The risk-free rate of return.
            window (int): The rolling window for calculations.
        """
        self.risk_free_rate = risk_free_rate
        self.window = window
        self.returns = []

    def add_return(self, pnl):
        """
        Add a new return to the performance monitor.

        Args:
            pnl (float): The profit or loss for the period.
        """
        self.returns.append(pnl)

    def calculate_rolling_sharpe(self):
        """
        Calculate the rolling Sharpe ratio.
        """
        if len(self.returns) < self.window:
            return 0.0

        returns_series = pd.Series(self.returns)
        rolling_returns = returns_series.rolling(window=self.window)

        mean_return = rolling_returns.mean()
        std_dev = rolling_returns.std()

        sharpe_ratio = (mean_return - self.risk_free_rate) / std_dev
        return sharpe_ratio.iloc[-1] * np.sqrt(self.window)  # Annualized

    def calculate_rolling_drawdown(self):
        """
        Calculate the rolling drawdown.
        """
        if len(self.returns) < self.window:
            return 0.0

        returns_series = pd.Series(self.returns)
        cumulative_returns = (1 + returns_series).cumprod()

        rolling_max = cumulative_returns.rolling(window=self.window).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max

        return drawdown.min()

    def calculate_sortino_ratio(self):
        """
        Calculate the Sortino ratio.
        """
        if len(self.returns) < self.window:
            return 0.0

        returns_series = pd.Series(self.returns)
        mean_return = returns_series.rolling(window=self.window).mean().iloc[-1]

        downside_returns = returns_series[returns_series < 0]
        downside_std_dev = downside_returns.rolling(window=self.window).std().iloc[-1]

        if downside_std_dev == 0:
            return 0.0

        sortino_ratio = (mean_return - self.risk_free_rate) / downside_std_dev
        return sortino_ratio * np.sqrt(self.window)  # Annualized

    def calculate_calmar_ratio(self):
        """
        Calculate the Calmar ratio.
        """
        if len(self.returns) < self.window:
            return 0.0

        mean_return = (
            pd.Series(self.returns).rolling(window=self.window).mean().iloc[-1]
        )
        max_drawdown = self.calculate_rolling_drawdown()

        if max_drawdown == 0:
            return 0.0

        return mean_return / abs(max_drawdown)
