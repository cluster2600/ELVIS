"""
Base strategy class for the BTC_BOT project.
This module defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
import logging

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger
        self.kwargs = kwargs
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy/sell signals based on the data.
        
        Args:
            data (pd.DataFrame): The data to generate signals from.
            
        Returns:
            Tuple[bool, bool]: A tuple of (buy_signal, sell_signal).
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, data: pd.DataFrame, current_price: float, available_capital: float) -> float:
        """
        Calculate the position size based on the data and available capital.
        
        Args:
            data (pd.DataFrame): The data to calculate position size from.
            current_price (float): The current price.
            available_capital (float): The available capital.
            
        Returns:
            float: The position size.
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate stop loss from.
            entry_price (float): The entry price.
            
        Returns:
            float: The stop loss price.
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate take profit from.
            entry_price (float): The entry price.
            
        Returns:
            float: The take profit price.
        """
        pass
