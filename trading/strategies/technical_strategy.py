"""
Technical strategy for the ELVIS project.
This module provides a concrete implementation of the BaseStrategy using technical indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import talib

from trading.strategies.base_strategy import BaseStrategy
from config import TRADING_CONFIG

class TechnicalStrategy(BaseStrategy):
    """
    Strategy based on technical indicators.
    Uses RSI, MACD, DX, and OBV to generate trading signals.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the technical strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Strategy parameters
        self.rsi_overbought = kwargs.get('rsi_overbought', 70)
        self.rsi_oversold = kwargs.get('rsi_oversold', 30)
        self.dx_threshold = kwargs.get('dx_threshold', 25)
        self.macd_threshold = kwargs.get('macd_threshold', 0)
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
            
            # Check if required indicators are present
            required_indicators = ['rsi', 'macd', 'dx', 'obv']
            if not all(indicator in latest for indicator in required_indicators):
                self.logger.warning(f"Missing required indicators: {[ind for ind in required_indicators if ind not in latest]}")
                return False, False
            
            # Extract indicators
            rsi = latest['rsi']
            macd = latest['macd']
            dx = latest['dx']
            obv = latest['obv']
            
            # Get previous OBV for comparison
            prev_obv = data.iloc[-2]['obv'] if len(data) > 1 else obv
            
            # Generate buy signal
            # Trend strength (DX > threshold), momentum (MACD > 0), RSI not overbought, OBV increasing
            buy_signal = (
                dx > self.dx_threshold and 
                macd > self.macd_threshold and 
                rsi < self.rsi_overbought and 
                obv > prev_obv
            )
            
            # Generate sell signal
            # RSI overbought or trend weakening
            sell_signal = (
                rsi > self.rsi_overbought or 
                (dx < self.dx_threshold and macd < self.macd_threshold)
            )
            
            self.logger.info(f"Signal Check: RSI={rsi:.2f}, MACD={macd:.2f}, DX={dx:.2f}, OBV={obv:.2f}, Buy={buy_signal}, Sell={sell_signal}")
            
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
            # Calculate stop loss based on ATR if available
            if 'atr' in data.columns:
                atr = data.iloc[-1]['atr']
                # Use 2 * ATR for stop loss
                stop_loss = entry_price - (2 * atr)
                
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
