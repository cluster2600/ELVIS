"""
Metrics utilities for the BTC_BOT project.
This module provides functions for calculating various financial metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.tsa.stattools as sts
import warnings
from typing import Tuple, Union, List, Dict, Any, Optional

# Default number of trading days in a year
TRADING_DAYS = 365

def compute_data_points_per_year(timeframe: str) -> int:
    """
    Compute the number of data points per year based on the given timeframe.
    
    Args:
        timeframe (str): The timeframe (e.g., '1m', '5m', '1h', '1d').
        
    Returns:
        int: The number of data points per year.
        
    Raises:
        ValueError: If the timeframe is not supported.
    """
    if timeframe == '1m':
        data_points_per_year = 60 * 24 * 365
    elif timeframe == '5m':
        data_points_per_year = 12 * 24 * 365
    elif timeframe == '10m':
        data_points_per_year = 6 * 24 * 365
    elif timeframe == '30m':
        data_points_per_year = 2 * 24 * 365
    elif timeframe == '1h':
        data_points_per_year = 24 * 365
    elif timeframe == '1d':
        data_points_per_year = 365
    else:
        raise ValueError('Timeframe not supported yet, please manually add!')
    return data_points_per_year

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = TRADING_DAYS) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns (np.ndarray): The returns.
        risk_free_rate (float, optional): The risk-free rate. Defaults to 0.0.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        float: The Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    return np.sqrt(annualization_factor) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = TRADING_DAYS) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns (np.ndarray): The returns.
        risk_free_rate (float, optional): The risk-free rate. Defaults to 0.0.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        float: The Sortino ratio.
    """
    excess_returns = returns - risk_free_rate
    downside_returns = np.where(returns < 0, returns, 0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    return np.sqrt(annualization_factor) * np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else 0

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        equity_curve (np.ndarray): The equity curve.
        
    Returns:
        float: The maximum drawdown.
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    
    # Calculate the running maximum
    running_max = equity_curve.cummax()
    
    # Calculate the drawdown
    drawdown = (equity_curve / running_max - 1)
    
    # Return the minimum drawdown (maximum loss)
    return drawdown.min()

def calculate_annualized_return(returns: np.ndarray, annualization_factor: int = TRADING_DAYS) -> float:
    """
    Calculate the annualized return.
    
    Args:
        returns (np.ndarray): The returns.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        float: The annualized return.
    """
    # Calculate the cumulative return
    cumulative_return = np.prod(1 + returns) - 1
    
    # Calculate the number of years
    n_years = len(returns) / annualization_factor
    
    # Calculate the annualized return
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    
    return annualized_return

def calculate_annualized_volatility(returns: np.ndarray, annualization_factor: int = TRADING_DAYS) -> float:
    """
    Calculate the annualized volatility.
    
    Args:
        returns (np.ndarray): The returns.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        float: The annualized volatility.
    """
    return np.std(returns, ddof=1) * np.sqrt(annualization_factor)

def calculate_calmar_ratio(returns: np.ndarray, annualization_factor: int = TRADING_DAYS) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns (np.ndarray): The returns.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        float: The Calmar ratio.
    """
    # Calculate the annualized return
    annualized_return = calculate_annualized_return(returns, annualization_factor)
    
    # Calculate the maximum drawdown
    max_drawdown = calculate_max_drawdown(np.cumprod(1 + returns))
    
    # Calculate the Calmar ratio
    return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate the win rate.
    
    Args:
        returns (np.ndarray): The returns.
        
    Returns:
        float: The win rate.
    """
    wins = np.sum(returns > 0)
    total = len(returns)
    return wins / total if total > 0 else 0

def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate the profit factor.
    
    Args:
        returns (np.ndarray): The returns.
        
    Returns:
        float: The profit factor.
    """
    gross_profit = np.sum(np.where(returns > 0, returns, 0))
    gross_loss = np.sum(np.where(returns < 0, returns, 0))
    return abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

def calculate_expectancy(returns: np.ndarray) -> float:
    """
    Calculate the expectancy.
    
    Args:
        returns (np.ndarray): The returns.
        
    Returns:
        float: The expectancy.
    """
    win_rate = calculate_win_rate(returns)
    avg_win = np.mean(np.where(returns > 0, returns, 0))
    avg_loss = np.mean(np.where(returns < 0, returns, 0))
    return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

def calculate_performance_metrics(returns: np.ndarray, annualization_factor: int = TRADING_DAYS) -> Dict[str, float]:
    """
    Calculate various performance metrics.
    
    Args:
        returns (np.ndarray): The returns.
        annualization_factor (int, optional): The annualization factor. Defaults to TRADING_DAYS.
        
    Returns:
        Dict[str, float]: The performance metrics.
    """
    metrics = {
        'annualized_return': calculate_annualized_return(returns, annualization_factor),
        'annualized_volatility': calculate_annualized_volatility(returns, annualization_factor),
        'sharpe_ratio': calculate_sharpe_ratio(returns, 0, annualization_factor),
        'sortino_ratio': calculate_sortino_ratio(returns, 0, annualization_factor),
        'max_drawdown': calculate_max_drawdown(np.cumprod(1 + returns)),
        'calmar_ratio': calculate_calmar_ratio(returns, annualization_factor),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'expectancy': calculate_expectancy(returns)
    }
    return metrics

def plot_equity_curve(equity_curve: np.ndarray, title: str = 'Equity Curve', figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the equity curve.
    
    Args:
        equity_curve (np.ndarray): The equity curve.
        title (str, optional): The title of the plot. Defaults to 'Equity Curve'.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).
    """
    plt.figure(figsize=figsize)
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

def plot_drawdown(equity_curve: np.ndarray, title: str = 'Drawdown', figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the drawdown.
    
    Args:
        equity_curve (np.ndarray): The equity curve.
        title (str, optional): The title of the plot. Defaults to 'Drawdown'.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    
    # Calculate the running maximum
    running_max = equity_curve.cummax()
    
    # Calculate the drawdown
    drawdown = (equity_curve / running_max - 1)
    
    plt.figure(figsize=figsize)
    plt.plot(drawdown)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.show()

def plot_returns_distribution(returns: np.ndarray, title: str = 'Returns Distribution', figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the returns distribution.
    
    Args:
        returns (np.ndarray): The returns.
        title (str, optional): The title of the plot. Defaults to 'Returns Distribution'.
        figsize (Tuple[int, int], optional): The figure size. Defaults to (10, 6).
    """
    plt.figure(figsize=figsize)
    plt.hist(returns, bins=50)
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def write_metrics_to_file(metrics: Dict[str, float], file_path: str, mode: str = 'w') -> None:
    """
    Write metrics to a file.
    
    Args:
        metrics (Dict[str, float]): The metrics to write.
        file_path (str): The file path.
        mode (str, optional): The file mode. Defaults to 'w'.
    """
    with open(file_path, mode) as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
