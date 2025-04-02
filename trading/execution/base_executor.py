"""
Base executor class for the BTC_BOT project.
This module defines the interface that all trading executors must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, Tuple
import logging

class BaseExecutor(ABC):
    """Base class for all trading executors."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the executor.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger
        self.kwargs = kwargs
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the executor.
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """
        Get the account balance.
        
        Returns:
            Dict[str, float]: The account balance.
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current position for the specified symbol.
        
        Args:
            symbol (str): The symbol to get the position for.
            
        Returns:
            Dict[str, Any]: The position.
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for the specified symbol.
        
        Args:
            symbol (str): The symbol to get the price for.
            
        Returns:
            float: The current price.
        """
        pass
    
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Set the leverage for the specified symbol.
        
        Args:
            symbol (str): The symbol to set the leverage for.
            leverage (int): The leverage to set.
        """
        pass
    
    @abstractmethod
    def execute_buy(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a buy order.
        
        Args:
            symbol (str): The symbol to buy.
            quantity (float): The quantity to buy.
            price (float): The price to buy at.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        pass
    
    @abstractmethod
    def execute_sell(self, symbol: str, quantity: float, price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a sell order.
        
        Args:
            symbol (str): The symbol to sell.
            quantity (float): The quantity to sell.
            price (float): The price to sell at.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        pass
    
    @abstractmethod
    def execute_stop_loss(self, symbol: str, quantity: float, stop_price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a stop loss order.
        
        Args:
            symbol (str): The symbol to set the stop loss for.
            quantity (float): The quantity to sell.
            stop_price (float): The stop price.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        pass
    
    @abstractmethod
    def execute_take_profit(self, symbol: str, quantity: float, take_profit_price: float, **kwargs) -> Dict[str, Any]:
        """
        Execute a take profit order.
        
        Args:
            symbol (str): The symbol to set the take profit for.
            quantity (float): The quantity to sell.
            take_profit_price (float): The take profit price.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The order details.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id (str): The order ID to cancel.
            
        Returns:
            bool: True if the order was cancelled successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id (str): The order ID to get the status for.
            
        Returns:
            Dict[str, Any]: The order status.
        """
        pass
