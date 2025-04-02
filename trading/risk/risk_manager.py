"""
Risk manager for the ELVIS project.
This module provides risk management functionality for trading.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from config import TRADING_CONFIG

class RiskManager:
    """
    Risk manager for trading.
    Handles risk management, position sizing, and trade limits.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the risk manager.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger
        
        # Risk parameters
        self.max_position_size_pct = kwargs.get('max_position_size_pct', 0.5)  # 50% of available capital
        self.max_leverage = kwargs.get('max_leverage', TRADING_CONFIG['LEVERAGE_MAX'])
        self.min_leverage = kwargs.get('min_leverage', TRADING_CONFIG['LEVERAGE_MIN'])
        self.stop_loss_pct = kwargs.get('stop_loss_pct', TRADING_CONFIG['STOP_LOSS_PCT'])
        self.take_profit_pct = kwargs.get('take_profit_pct', TRADING_CONFIG['TAKE_PROFIT_PCT'])
        self.max_trades_per_day = kwargs.get('max_trades_per_day', TRADING_CONFIG['MAX_TRADES_PER_DAY'])
        self.daily_profit_target_usd = kwargs.get('daily_profit_target_usd', TRADING_CONFIG['DAILY_PROFIT_TARGET_USD'])
        self.daily_loss_limit_usd = kwargs.get('daily_loss_limit_usd', TRADING_CONFIG['DAILY_LOSS_LIMIT_USD'])
        self.min_capital_usd = kwargs.get('min_capital_usd', TRADING_CONFIG['MIN_CAPITAL_USD'])
        
        # Trade tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.cooldown_period = kwargs.get('cooldown_period', TRADING_CONFIG['COOLDOWN'])  # seconds
    
    def check_capital(self, available_capital: float) -> bool:
        """
        Check if there is enough capital to trade.
        
        Args:
            available_capital (float): The available capital.
            
        Returns:
            bool: True if there is enough capital, False otherwise.
        """
        if available_capital < self.min_capital_usd:
            self.logger.warning(f"Insufficient capital: {available_capital:.2f} USD < {self.min_capital_usd:.2f} USD")
            return False
        
        self.logger.info(f"Sufficient capital: {available_capital:.2f} USD >= {self.min_capital_usd:.2f} USD")
        return True
    
    def check_trade_limits(self) -> bool:
        """
        Check if trade limits have been reached.
        
        Returns:
            bool: True if trade limits have not been reached, False otherwise.
        """
        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            self.logger.warning(f"Max trades per day reached: {self.trades_today} >= {self.max_trades_per_day}")
            return False
        
        # Check daily profit target
        if self.daily_pnl >= self.daily_profit_target_usd:
            self.logger.warning(f"Daily profit target reached: {self.daily_pnl:.2f} USD >= {self.daily_profit_target_usd:.2f} USD")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit_usd:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f} USD <= {self.daily_loss_limit_usd:.2f} USD")
            return False
        
        # Check cooldown period
        if self.last_trade_time is not None:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < self.cooldown_period:
                self.logger.warning(f"Cooldown period not elapsed: {elapsed:.2f}s < {self.cooldown_period:.2f}s")
                return False
        
        self.logger.info("Trade limits not reached")
        return True
    
    def calculate_position_size(self, available_capital: float, current_price: float, volatility: float) -> float:
        """
        Calculate the position size based on available capital and volatility.
        
        Args:
            available_capital (float): The available capital.
            current_price (float): The current price.
            volatility (float): The volatility (e.g., ATR).
            
        Returns:
            float: The position size.
        """
        # Calculate position value as a percentage of available capital
        # Adjust based on volatility
        volatility_factor = 1.0
        if volatility > 0:
            # Reduce position size for higher volatility
            volatility_factor = 1.0 / (1.0 + volatility / current_price)
        
        position_value = available_capital * self.max_position_size_pct * volatility_factor
        
        # Convert to quantity
        quantity = position_value / current_price
        
        # Adjust for minimum quantity
        min_quantity = 0.001  # Minimum BTC quantity
        if quantity < min_quantity:
            self.logger.warning(f"Calculated quantity {quantity:.8f} is below minimum {min_quantity}. Using minimum.")
            quantity = min_quantity
        
        self.logger.info(f"Calculated position size: {quantity:.8f} BTC (value: {quantity * current_price:.2f} USD)")
        
        return quantity
    
    def calculate_leverage(self, volatility: float, signal_strength: float) -> int:
        """
        Calculate the leverage based on volatility and signal strength.
        
        Args:
            volatility (float): The volatility (e.g., ATR).
            signal_strength (float): The signal strength (0.0 to 1.0).
            
        Returns:
            int: The leverage.
        """
        # Base leverage on signal strength
        leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * signal_strength
        
        # Adjust for volatility
        if volatility > 0:
            # Reduce leverage for higher volatility
            volatility_factor = 1.0 / (1.0 + volatility)
            leverage = leverage * volatility_factor
        
        # Ensure leverage is within limits
        leverage = max(self.min_leverage, min(int(leverage), self.max_leverage))
        
        self.logger.info(f"Calculated leverage: {leverage}x (signal strength: {signal_strength:.2f}, volatility factor: {volatility_factor:.2f})")
        
        return leverage
    
    def calculate_stop_loss(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the stop loss price based on entry price and volatility.
        
        Args:
            entry_price (float): The entry price.
            volatility (float): The volatility (e.g., ATR).
            
        Returns:
            float: The stop loss price.
        """
        # Calculate stop loss based on ATR if available
        if volatility > 0:
            # Use 2 * ATR for stop loss
            stop_loss = entry_price - (2 * volatility)
            
            # Ensure stop loss is not more than stop_loss_pct away from entry price
            min_stop_loss = entry_price * (1 - self.stop_loss_pct)
            stop_loss = max(stop_loss, min_stop_loss)
        else:
            # Use fixed percentage stop loss
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        
        self.logger.info(f"Calculated stop loss: {stop_loss:.2f} USD (entry: {entry_price:.2f} USD)")
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, volatility: float) -> float:
        """
        Calculate the take profit price based on entry price and volatility.
        
        Args:
            entry_price (float): The entry price.
            volatility (float): The volatility (e.g., ATR).
            
        Returns:
            float: The take profit price.
        """
        # Calculate take profit based on ATR if available
        if volatility > 0:
            # Use 3 * ATR for take profit (risk-reward ratio of 1.5)
            take_profit = entry_price + (3 * volatility)
            
            # Ensure take profit is not less than take_profit_pct away from entry price
            min_take_profit = entry_price * (1 + self.take_profit_pct)
            take_profit = max(take_profit, min_take_profit)
        else:
            # Use fixed percentage take profit
            take_profit = entry_price * (1 + self.take_profit_pct)
        
        self.logger.info(f"Calculated take profit: {take_profit:.2f} USD (entry: {entry_price:.2f} USD)")
        
        return take_profit
    
    def update_trade_stats(self, pnl: float) -> None:
        """
        Update trade statistics.
        
        Args:
            pnl (float): The profit/loss from the trade.
        """
        self.trades_today += 1
        self.daily_pnl += pnl
        self.last_trade_time = datetime.now()
        
        self.logger.info(f"Updated trade stats: trades_today={self.trades_today}, daily_pnl={self.daily_pnl:.2f} USD")
    
    def reset_daily_stats(self) -> None:
        """
        Reset daily statistics.
        """
        self.trades_today = 0
        self.daily_pnl = 0.0
        
        self.logger.info("Reset daily stats")
    
    def check_new_day(self) -> bool:
        """
        Check if it's a new day and reset stats if needed.
        
        Returns:
            bool: True if it's a new day, False otherwise.
        """
        if self.last_trade_time is None:
            return False
        
        now = datetime.now()
        if now.date() > self.last_trade_time.date():
            self.logger.info("New day detected, resetting daily stats")
            self.reset_daily_stats()
            return True
        
        return False
