"""
EMA-RSI strategy for the ELVIS project.
This module provides a strategy based on EMA crossovers and RSI.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import talib

from trading.strategies.base_strategy import BaseStrategy
from config import TRADING_CONFIG

class EmaRsiStrategy(BaseStrategy):
    """
    Strategy based on EMA crossovers and RSI.
    Uses EMA-9, EMA-21, RSI, and volatility to generate trading signals.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the EMA-RSI strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Strategy parameters
        self.ema_short_period = kwargs.get('ema_short_period', 9)
        self.ema_long_period = kwargs.get('ema_long_period', 21)
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.rsi_oversold = kwargs.get('rsi_oversold', 30)
        self.rsi_overbought = kwargs.get('rsi_overbought', 70)
        self.volatility_threshold = kwargs.get('volatility_threshold', 0.005)
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
            # Make a copy of the data to avoid SettingWithCopyWarning
            df = data.copy()
            
            # Calculate indicators if they don't exist
            if 'ema_short' not in df.columns:
                df.loc[:, 'ema_short'] = talib.EMA(df['close'].values, timeperiod=self.ema_short_period)
            
            if 'ema_long' not in df.columns:
                df.loc[:, 'ema_long'] = talib.EMA(df['close'].values, timeperiod=self.ema_long_period)
            
            if 'rsi' not in df.columns:
                df.loc[:, 'rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            if 'volatility' not in df.columns:
                # Calculate volatility as the standard deviation of returns
                df.loc[:, 'volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # Get the latest data point
            latest = df.iloc[-1]
            
            # Generate buy signal
            # EMA short > EMA long (bullish trend), RSI < oversold threshold, volatility > threshold
            buy_signal = (
                latest['ema_short'] > latest['ema_long'] and 
                latest['rsi'] < self.rsi_oversold and 
                latest['volatility'] > self.volatility_threshold
            )
            
            # Generate sell signal
            # EMA short < EMA long (bearish trend) or RSI > overbought threshold
            sell_signal = (
                latest['ema_short'] < latest['ema_long'] or 
                latest['rsi'] > self.rsi_overbought
            )
            
            self.logger.info(f"Signal Check: EMA9={latest['ema_short']:.2f}, EMA21={latest['ema_long']:.2f}, RSI={latest['rsi']:.2f}, Volatility={latest['volatility']:.4f}, Buy={buy_signal}, Sell={sell_signal}")
            
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
            # Calculate position value as a percentage of available capital
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
            # Calculate volatility if available
            volatility = 0.0
            if 'volatility' in data.columns:
                volatility = data['volatility'].iloc[-1]
            
            # Calculate stop loss based on volatility if available
            if volatility > 0:
                # Use 1.5 * volatility for stop loss
                stop_loss = entry_price * (1 - 1.5 * volatility)
                
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
            # Calculate volatility if available
            volatility = 0.0
            if 'volatility' in data.columns:
                volatility = data['volatility'].iloc[-1]
            
            # Calculate take profit based on volatility if available
            if volatility > 0:
                # Use 2 * volatility for take profit (risk-reward ratio of 1.33)
                take_profit = entry_price * (1 + 2 * volatility)
                
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
