import pandas as pd
from typing import List, Dict, Any

class TradeAnalyzer:
    """
    Analyzes a list of trades to provide insights into trading performance.
    """

    def __init__(self, trades: List[Dict[str, Any]]):
        """
        Initialize the TradeAnalyzer.

        Args:
            trades (List[Dict[str, Any]]): A list of trade dictionaries.
        """
        self.trades = pd.DataFrame(trades)

    def get_win_loss_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of winning and losing trades.
        """
        if self.trades.empty:
            return {'wins': 0, 'losses': 0}
            
        wins = self.trades[self.trades['pnl'] > 0].shape[0]
        losses = self.trades[self.trades['pnl'] <= 0].shape[0]
        return {'wins': wins, 'losses': losses}

    def get_average_pnl(self) -> Dict[str, float]:
        """
        Get the average PnL for winning and losing trades.
        """
        if self.trades.empty:
            return {'avg_win': 0.0, 'avg_loss': 0.0}

        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean()
        avg_loss = self.trades[self.trades['pnl'] <= 0]['pnl'].mean()
        return {'avg_win': avg_win, 'avg_loss': avg_loss}

    def get_trade_duration_distribution(self) -> Dict[str, float]:
        """
        Get the distribution of trade durations.
        """
        if self.trades.empty or 'entry_time' not in self.trades.columns or 'exit_time' not in self.trades.columns:
            return {}
            
        self.trades['duration'] = (self.trades['exit_time'] - self.trades['entry_time']).dt.total_seconds()
        return self.trades['duration'].describe().to_dict()
