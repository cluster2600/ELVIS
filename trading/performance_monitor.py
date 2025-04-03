"""
Performance monitoring for trading strategies.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from datetime import datetime

class PerformanceMonitor:
    """Monitors and calculates trading performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize the performance monitor."""
        self.logger = logger
        self.returns: List[float] = []
        self.trades: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a trade to the performance history."""
        self.trades.append(trade)
        if 'return' in trade:
            self.returns.append(trade['return'])
            
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio for the trading performance.
        
        Args:
            risk_free_rate (float): Annual risk-free rate (default: 2%)
            
        Returns:
            float: Sharpe ratio
        """
        try:
            if not self.returns:
                return 0.0
                
            # Convert daily returns to numpy array
            returns = np.array(self.returns)
            
            # Calculate annualized return
            annualized_return = np.mean(returns) * 252  # 252 trading days
            
            # Calculate annualized volatility
            annualized_volatility = np.std(returns) * np.sqrt(252)
            
            if annualized_volatility == 0:
                return 0.0
                
            # Calculate Sharpe ratio
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
            
    def calculate_win_rate(self) -> float:
        """Calculate the win rate of trades."""
        if not self.trades:
            return 0.0
            
        winning_trades = sum(1 for trade in self.trades if trade.get('return', 0) > 0)
        return winning_trades / len(self.trades)
        
    def calculate_profit_factor(self) -> float:
        """Calculate the profit factor."""
        if not self.trades:
            return 0.0
            
        gross_profit = sum(trade.get('return', 0) for trade in self.trades if trade.get('return', 0) > 0)
        gross_loss = abs(sum(trade.get('return', 0) for trade in self.trades if trade.get('return', 0) < 0))
        
        if gross_loss == 0:
            return float('inf')
            
        return gross_profit / gross_loss
        
    def calculate_max_drawdown(self) -> float:
        """Calculate the maximum drawdown."""
        if not self.returns:
            return 0.0
            
        returns = np.array(self.returns)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        
        return np.max(drawdowns)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get all performance metrics."""
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'max_drawdown': self.calculate_max_drawdown()
        } 