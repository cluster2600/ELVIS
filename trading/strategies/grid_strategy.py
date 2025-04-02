"""
Grid Trading strategy for the ELVIS project.
This module provides a concrete implementation of the BaseStrategy using grid trading principles.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import talib

from trading.strategies.base_strategy import BaseStrategy
from config import TRADING_CONFIG

class GridStrategy(BaseStrategy):
    """
    Strategy based on grid trading principles.
    Creates a grid of buy and sell orders at predetermined price levels.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the grid trading strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Strategy parameters
        self.grid_levels = kwargs.get('grid_levels', 10)
        self.grid_spacing_pct = kwargs.get('grid_spacing_pct', 0.01)  # 1% spacing
        self.dynamic_spacing = kwargs.get('dynamic_spacing', True)
        self.min_grid_spacing_pct = kwargs.get('min_grid_spacing_pct', 0.005)  # 0.5% minimum spacing
        self.max_grid_spacing_pct = kwargs.get('max_grid_spacing_pct', 0.02)  # 2% maximum spacing
        self.position_size_per_grid = kwargs.get('position_size_per_grid', 0.1)  # 10% of available capital per grid
        self.stop_loss_pct = kwargs.get('stop_loss_pct', TRADING_CONFIG['STOP_LOSS_PCT'])
        self.take_profit_pct = kwargs.get('take_profit_pct', TRADING_CONFIG['TAKE_PROFIT_PCT'])
        
        # Grid state
        self.grid_prices = []
        self.last_update_price = None
        self.grid_orders = {}
    
    def _calculate_grid_spacing(self, data: pd.DataFrame) -> float:
        """
        Calculate the grid spacing based on volatility.
        
        Args:
            data (pd.DataFrame): The data to calculate grid spacing from.
            
        Returns:
            float: The grid spacing percentage.
        """
        if not self.dynamic_spacing:
            return self.grid_spacing_pct
        
        try:
            # Calculate grid spacing based on ATR if available
            if 'atr' in data.columns:
                current_price = data.iloc[-1]['close']
                atr = data.iloc[-1]['atr']
                
                # Calculate spacing as a percentage of price based on ATR
                atr_pct = atr / current_price
                
                # Scale the spacing
                spacing = atr_pct * 0.5  # Use half of ATR as spacing
                
                # Ensure spacing is within bounds
                spacing = max(self.min_grid_spacing_pct, min(spacing, self.max_grid_spacing_pct))
                
                self.logger.info(f"Calculated dynamic grid spacing: {spacing:.4f} (ATR: {atr:.2f}, Price: {current_price:.2f})")
                
                return spacing
            else:
                self.logger.warning("ATR not available for dynamic grid spacing, using default")
                return self.grid_spacing_pct
                
        except Exception as e:
            self.logger.error(f"Error calculating grid spacing: {e}")
            return self.grid_spacing_pct
    
    def _generate_grid_prices(self, current_price: float, grid_spacing: float) -> List[float]:
        """
        Generate grid price levels.
        
        Args:
            current_price (float): The current price.
            grid_spacing (float): The grid spacing percentage.
            
        Returns:
            List[float]: The grid price levels.
        """
        try:
            # Calculate number of levels above and below current price
            levels_above = self.grid_levels // 2
            levels_below = self.grid_levels - levels_above
            
            # Generate grid prices
            grid_prices = []
            
            # Levels below current price
            for i in range(1, levels_below + 1):
                price = current_price * (1 - i * grid_spacing)
                grid_prices.append(price)
            
            # Current price level
            grid_prices.append(current_price)
            
            # Levels above current price
            for i in range(1, levels_above + 1):
                price = current_price * (1 + i * grid_spacing)
                grid_prices.append(price)
            
            # Sort grid prices
            grid_prices.sort()
            
            self.logger.info(f"Generated {len(grid_prices)} grid price levels from {grid_prices[0]:.2f} to {grid_prices[-1]:.2f}")
            
            return grid_prices
            
        except Exception as e:
            self.logger.error(f"Error generating grid prices: {e}")
            return [current_price]
    
    def _update_grid(self, data: pd.DataFrame) -> None:
        """
        Update the grid based on current price.
        
        Args:
            data (pd.DataFrame): The data to update grid from.
        """
        try:
            current_price = data.iloc[-1]['close']
            
            # Check if grid needs to be updated
            if (self.last_update_price is None or 
                abs(current_price - self.last_update_price) / self.last_update_price > self.grid_spacing_pct * 2):
                
                self.logger.info(f"Updating grid (current price: {current_price:.2f}, last update price: {self.last_update_price})")
                
                # Calculate grid spacing
                grid_spacing = self._calculate_grid_spacing(data)
                
                # Generate grid prices
                self.grid_prices = self._generate_grid_prices(current_price, grid_spacing)
                
                # Update last update price
                self.last_update_price = current_price
                
                # Reset grid orders
                self.grid_orders = {}
                
                # Initialize grid orders
                for i, price in enumerate(self.grid_prices):
                    self.grid_orders[i] = {
                        'price': price,
                        'type': 'buy' if price < current_price else 'sell',
                        'status': 'pending'
                    }
            
        except Exception as e:
            self.logger.error(f"Error updating grid: {e}")
    
    def _find_active_grid_levels(self, current_price: float) -> Tuple[int, int]:
        """
        Find the active grid levels based on current price.
        
        Args:
            current_price (float): The current price.
            
        Returns:
            Tuple[int, int]: The indices of the grid levels below and above current price.
        """
        try:
            # Find the grid level below current price
            level_below = None
            for i in range(len(self.grid_prices) - 1, -1, -1):
                if self.grid_prices[i] <= current_price:
                    level_below = i
                    break
            
            # Find the grid level above current price
            level_above = None
            for i in range(len(self.grid_prices)):
                if self.grid_prices[i] >= current_price:
                    level_above = i
                    break
            
            # If no level found, use the closest
            if level_below is None:
                level_below = 0
            if level_above is None:
                level_above = len(self.grid_prices) - 1
            
            return level_below, level_above
            
        except Exception as e:
            self.logger.error(f"Error finding active grid levels: {e}")
            return 0, 0
    
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
            # Get current price
            current_price = data.iloc[-1]['close']
            
            # Update grid if needed
            self._update_grid(data)
            
            # Find active grid levels
            level_below, level_above = self._find_active_grid_levels(current_price)
            
            # Check if price crossed a grid level
            buy_signal = False
            sell_signal = False
            
            # If price is at or below a buy grid level
            if level_below < len(self.grid_prices) - 1:
                buy_price = self.grid_prices[level_below]
                if current_price <= buy_price and self.grid_orders[level_below]['status'] == 'pending':
                    buy_signal = True
                    self.grid_orders[level_below]['status'] = 'executed'
                    self.logger.info(f"Buy signal at grid level {level_below} (price: {buy_price:.2f})")
            
            # If price is at or above a sell grid level
            if level_above > 0:
                sell_price = self.grid_prices[level_above]
                if current_price >= sell_price and self.grid_orders[level_above]['status'] == 'pending':
                    sell_signal = True
                    self.grid_orders[level_above]['status'] = 'executed'
                    self.logger.info(f"Sell signal at grid level {level_above} (price: {sell_price:.2f})")
            
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
            # Calculate position size per grid
            position_value = available_capital * self.position_size_per_grid
            
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
            # For grid trading, use a fixed percentage stop loss
            # This is a global stop loss for risk management
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
            # For grid trading, find the next grid level above entry price
            level_below, level_above = self._find_active_grid_levels(entry_price)
            
            if level_above < len(self.grid_prices) and level_above > level_below:
                # Use the next grid level as take profit
                take_profit = self.grid_prices[level_above]
            else:
                # Use fixed percentage take profit as fallback
                take_profit = entry_price * (1 + self.take_profit_pct)
            
            self.logger.info(f"Calculated take profit: {take_profit:.2f} USD (entry: {entry_price:.2f} USD)")
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * (1 + self.take_profit_pct)
    
    def reset_grid(self) -> None:
        """
        Reset the grid state.
        """
        self.grid_prices = []
        self.last_update_price = None
        self.grid_orders = {}
        self.logger.info("Grid state reset")
