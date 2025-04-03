"""
Simple example strategy for testing the validation system.
"""

import numpy as np
import pandas as pd

def strategy(data: pd.DataFrame, initial_capital: float = 100000, params: dict = None) -> dict:
    """
    Simple moving average crossover strategy.
    
    Args:
        data: DataFrame with price data
        initial_capital: Initial capital
        params: Strategy parameters
        
    Returns:
        Dictionary with strategy results
    """
    if params is None:
        params = {
            'short_window': 20,
            'long_window': 50
        }
        
    # Calculate moving averages
    data['short_ma'] = data['close'].rolling(window=params['short_window']).mean()
    data['long_ma'] = data['close'].rolling(window=params['long_window']).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Calculate metrics
    returns = data['strategy_returns'].dropna()
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    # Calculate win rate and profit factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else np.inf
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    
    return {
        'returns': returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'survival': True  # For stress testing
    } 