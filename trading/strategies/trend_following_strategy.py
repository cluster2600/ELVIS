"""
Trend Following strategy for the ELVIS project.
This module provides a concrete implementation of the BaseStrategy using trend following principles.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import talib

from trading.strategies.base_strategy import BaseStrategy
from config import TRADING_CONFIG

class TrendFollowingStrategy(BaseStrategy):
    """
    Strategy based on trend following principles.
    Uses moving averages, ADX, and other indicators to identify and follow trends.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the trend following strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Strategy parameters
        self.fast_ma_period = kwargs.get('fast_ma_period', 20)
        self.slow_ma_period = kwargs.get('slow_ma_period', 50)
        self.adx_period = kwargs.get('adx_period', 14)
        self.adx_threshold = kwargs.get('adx_threshold', 25)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', TRADING_CONFIG['STOP_LOSS_PCT'])
        self.take_profit_pct = kwargs.get('take_profit_pct', TRADING_CONFIG['TAKE_PROFIT_PCT'])
        self.position_size_pct = kwargs.get('position_size_pct', 0.5)  # 50% of available capital
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy/sell signals based on the data.
        
        Args:
            data (pd.DataFrame): The data to generate signals from.
            
        Returns:
            Tuple[bool, bool]: A tuple of (buy_signal, sell_signal).
        """
        if data.empty:
            self.logger.warning("Empty data provided to generate_signals")
            return False, False
        
        try:
            # Get the latest data point
            latest = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else latest
            
            # Check if required indicators are present
            required_indicators = ['sma_20', 'sma_50', 'adx']
            if not all(indicator in latest for indicator in required_indicators):
                self.logger.warning(f"Missing required indicators: {[ind for ind in required_indicators if ind not in latest]}")
                return False, False
            
            # Extract indicators
            fast_ma = latest['sma_20']
            slow_ma = latest['sma_50']
            adx = latest['adx']
            
            # Previous values
            prev_fast_ma = previous['sma_20']
            prev_slow_ma = previous['sma_50']
            
            # Calculate crossovers
            golden_cross = fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma
            death_cross = fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma
            
            # Generate buy signal
            # Golden cross and strong trend
            buy_signal = (
                golden_cross and 
                adx > self.adx_threshold
            )
            
            # Generate sell signal
            # Death cross or weakening trend
            sell_signal = (
                death_cross or 
                (adx < self.adx_threshold and fast_ma < slow_ma)
            )
            
            self.logger.info(f"Signal Check: Fast MA={fast_ma:.2f}, Slow MA={slow_ma:.2f}, ADX={adx:.2f}, Buy={buy_signal}, Sell={sell_signal}")
            
            return buy_signal, sell_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return False, False
    
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
        try:
            # Calculate position size as a percentage of available capital
            position_value = available_capital * self.position_size_pct
            
            # Adjust position size based on trend strength
            if 'adx' in data.iloc[-1]:
                adx = data.iloc[-1]['adx']
                
                # Increase position size for stronger trends
                trend_factor = min(1.5, max(0.5, adx / self.adx_threshold))
                position_value *= trend_factor
                
                self.logger.info(f"Adjusting position size by trend factor: {trend_factor:.2f} (ADX: {adx:.2f})")
            
            # Convert to quantity
            quantity = position_value / current_price
            
            # Adjust for minimum quantity
            min_quantity = 0.001  # Minimum BTC quantity
            if quantity < min_quantity:
                self.logger.warning(f"Calculated quantity {quantity:.8f} is below minimum {min_quantity}. Using minimum.")
                quantity = min_quantity
            
            self.logger.info(f"Calculated position size: {quantity:.8f} BTC (value: {quantity * current_price:.2f} USD)")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate stop loss from.
            entry_price (float): The entry price.
            
        Returns:
            float: The stop loss price.
        """
        try:
            # For trend following, use recent swing low/high as stop loss
            if len(data) >= 10:
                # Get recent price action
                recent_data = data.iloc[-10:]
                
                # Find recent swing low
                swing_low = recent_data['low'].min()
                
                # Use swing low as stop loss, with a buffer
                buffer = 0.002  # 0.2% buffer
                stop_loss = swing_low * (1 - buffer)
                
                # Ensure stop loss is not more than stop_loss_pct away from entry price
                min_stop_loss = entry_price * (1 - self.stop_loss_pct)
                stop_loss = max(stop_loss, min_stop_loss)
            else:
                # Use fixed percentage stop loss
                stop_loss = entry_price * (1 - self.stop_loss_pct)
            
            self.logger.info(f"Calculated stop loss: {stop_loss:.2f} USD (entry: {entry_price:.2f} USD)")
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price * (1 - self.stop_loss_pct)
    
    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate take profit from.
            entry_price (float): The entry price.
            
        Returns:
            float: The take profit price.
        """
        try:
            # For trend following, use a trailing stop rather than fixed take profit
            # Here we'll calculate a potential take profit level, but in practice
            # a trailing stop would be used to maximize trend profits
            
            # Calculate take profit based on ATR if available
            if 'atr' in data.columns:
                atr = data.iloc[-1]['atr']
                # Use 3 * ATR for take profit (risk-reward ratio of 1.5)
                take_profit = entry_price + (3 * atr)
                
                # Ensure take profit is not less than take_profit_pct away from entry price
                min_take_profit = entry_price * (1 + self.take_profit_pct)
                take_profit = max(take_profit, min_take_profit)
            else:
                # Use fixed percentage take profit
                take_profit = entry_price * (1 + self.take_profit_pct)
            
            self.logger.info(f"Calculated take profit: {take_profit:.2f} USD (entry: {entry_price:.2f} USD)")
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * (1 + self.take_profit_pct)
    
    def calculate_trailing_stop(self, data: pd.DataFrame, entry_price: float, current_price: float, highest_price: float) -> float:
        """
        Calculate the trailing stop price.
        
        Args:
            data (pd.DataFrame): The data to calculate trailing stop from.
            entry_price (float): The entry price.
            current_price (float): The current price.
            highest_price (float): The highest price since entry.
            
        Returns:
            float: The trailing stop price.
        """
        try:
            # Calculate trailing stop based on ATR if available
            if 'atr' in data.columns:
                atr = data.iloc[-1]['atr']
                
                # Initial stop loss
                initial_stop = entry_price - (2 * atr)
                
                # Trailing stop based on highest price
                trailing_stop = highest_price - (2 * atr)
                
                # Use the higher of initial stop and trailing stop
                stop_price = max(initial_stop, trailing_stop)
                
                # Ensure stop is not more than stop_loss_pct away from entry price
                min_stop = entry_price * (1 - self.stop_loss_pct)
                stop_price = max(stop_price, min_stop)
            else:
                # Use fixed percentage trailing stop
                trailing_pct = 0.05  # 5% trailing stop
                stop_price = highest_price * (1 - trailing_pct)
                
                # Ensure stop is not more than stop_loss_pct away from entry price
                min_stop = entry_price * (1 - self.stop_loss_pct)
                stop_price = max(stop_price, min_stop)
            
            self.logger.info(f"Calculated trailing stop: {stop_price:.2f} USD (entry: {entry_price:.2f} USD, highest: {highest_price:.2f} USD)")
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop: {e}")
            return entry_price * (1 - self.stop_loss_pct)
