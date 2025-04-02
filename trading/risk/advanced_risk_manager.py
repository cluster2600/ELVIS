"""
Advanced Risk Manager for the ELVIS project.
This module provides enhanced risk management capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading.risk.risk_manager import RiskManager
from config import TRADING_CONFIG

class AdvancedRiskManager(RiskManager):
    """
    Advanced risk manager with enhanced risk management capabilities.
    Implements Kelly Criterion, dynamic risk allocation, and correlation-based position sizing.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the advanced risk manager.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Advanced risk parameters
        self.use_kelly = kwargs.get('use_kelly', True)
        self.kelly_fraction = kwargs.get('kelly_fraction', 0.5)  # Conservative Kelly
        self.max_kelly_allocation = kwargs.get('max_kelly_allocation', 0.2)  # Maximum 20% of capital
        self.market_regime_window = kwargs.get('market_regime_window', 20)  # Days for market regime detection
        self.correlation_window = kwargs.get('correlation_window', 30)  # Days for correlation calculation
        self.circuit_breaker_drawdown = kwargs.get('circuit_breaker_drawdown', 0.1)  # 10% drawdown triggers circuit breaker
        self.volatility_scaling = kwargs.get('volatility_scaling', True)  # Scale position size by volatility
        self.target_volatility = kwargs.get('target_volatility', 0.2)  # Target annual volatility
        
        # State variables
        self.win_history = []
        self.loss_history = []
        self.drawdown_history = []
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.market_regime = 'neutral'  # 'bullish', 'bearish', or 'neutral'
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
        self.circuit_breaker_cooldown = timedelta(hours=24)  # 24-hour cooldown after circuit breaker
        self.asset_correlations = {}
    
    def calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate the Kelly Criterion allocation.
        
        Args:
            win_rate (float): The win rate (0.0 to 1.0).
            win_loss_ratio (float): The win/loss ratio.
            
        Returns:
            float: The Kelly allocation (0.0 to 1.0).
        """
        try:
            # Kelly formula: f* = p - (1-p)/r
            # where p is win rate, r is win/loss ratio
            
            # Ensure win_rate is between 0 and 1
            win_rate = max(0.0, min(1.0, win_rate))
            
            # Ensure win_loss_ratio is positive
            win_loss_ratio = max(0.1, win_loss_ratio)
            
            # Calculate Kelly allocation
            kelly = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply Kelly fraction for conservatism
            kelly *= self.kelly_fraction
            
            # Cap at maximum allocation
            kelly = min(kelly, self.max_kelly_allocation)
            
            # Ensure non-negative
            kelly = max(0.0, kelly)
            
            self.logger.info(f"Calculated Kelly allocation: {kelly:.4f} (win rate: {win_rate:.2f}, win/loss ratio: {win_loss_ratio:.2f})")
            
            return kelly
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.05  # Default to 5% allocation
    
    def update_win_loss_history(self, pnl: float) -> None:
        """
        Update win/loss history.
        
        Args:
            pnl (float): The profit/loss amount.
        """
        try:
            if pnl > 0:
                self.win_history.append(pnl)
            else:
                self.loss_history.append(abs(pnl))
            
            # Keep history limited to recent trades
            max_history = 100
            if len(self.win_history) > max_history:
                self.win_history = self.win_history[-max_history:]
            if len(self.loss_history) > max_history:
                self.loss_history = self.loss_history[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error updating win/loss history: {e}")
    
    def get_win_rate(self) -> float:
        """
        Get the current win rate.
        
        Returns:
            float: The win rate (0.0 to 1.0).
        """
        try:
            total_trades = len(self.win_history) + len(self.loss_history)
            
            if total_trades == 0:
                return 0.5  # Default to 50% win rate
            
            win_rate = len(self.win_history) / total_trades
            
            return win_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.5  # Default to 50% win rate
    
    def get_win_loss_ratio(self) -> float:
        """
        Get the current win/loss ratio.
        
        Returns:
            float: The win/loss ratio.
        """
        try:
            if not self.win_history:
                return 1.0  # Default to 1.0 ratio
            
            if not self.loss_history:
                return 10.0  # High ratio if no losses
            
            avg_win = sum(self.win_history) / len(self.win_history)
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            
            if avg_loss == 0:
                return 10.0  # High ratio if avg_loss is zero
            
            win_loss_ratio = avg_win / avg_loss
            
            return win_loss_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating win/loss ratio: {e}")
            return 1.0  # Default to 1.0 ratio
    
    def update_equity(self, equity: float) -> None:
        """
        Update equity value and track drawdown.
        
        Args:
            equity (float): The current equity value.
        """
        try:
            self.current_equity = equity
            
            # Update peak equity
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            # Calculate drawdown
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                self.drawdown_history.append(drawdown)
                
                # Keep history limited
                max_history = 100
                if len(self.drawdown_history) > max_history:
                    self.drawdown_history = self.drawdown_history[-max_history:]
                
                # Check for circuit breaker
                if drawdown >= self.circuit_breaker_drawdown and not self.circuit_breaker_triggered:
                    self.circuit_breaker_triggered = True
                    self.circuit_breaker_time = datetime.now()
                    self.logger.warning(f"Circuit breaker triggered! Drawdown: {drawdown:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error updating equity: {e}")
    
    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is active.
        
        Returns:
            bool: True if circuit breaker is active, False otherwise.
        """
        try:
            if not self.circuit_breaker_triggered:
                return False
            
            # Check if cooldown period has elapsed
            if datetime.now() - self.circuit_breaker_time > self.circuit_breaker_cooldown:
                self.circuit_breaker_triggered = False
                self.logger.info("Circuit breaker reset after cooldown period")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breaker: {e}")
            return False
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            str: The market regime ('bullish', 'bearish', or 'neutral').
        """
        try:
            if data.empty or len(data) < self.market_regime_window:
                return 'neutral'
            
            # Get recent data
            recent_data = data.iloc[-self.market_regime_window:]
            
            # Calculate returns
            returns = recent_data['close'].pct_change().dropna()
            
            # Calculate metrics
            avg_return = returns.mean()
            volatility = returns.std()
            
            # Check if SMA indicators are available
            if 'sma_20' in recent_data.columns and 'sma_50' in recent_data.columns:
                sma_20 = recent_data['sma_20'].iloc[-1]
                sma_50 = recent_data['sma_50'].iloc[-1]
                price = recent_data['close'].iloc[-1]
                
                # Determine regime based on price relative to SMAs
                if price > sma_20 and sma_20 > sma_50:
                    regime = 'bullish'
                elif price < sma_20 and sma_20 < sma_50:
                    regime = 'bearish'
                else:
                    regime = 'neutral'
            else:
                # Determine regime based on returns
                if avg_return > 0.001 and avg_return > volatility:
                    regime = 'bullish'
                elif avg_return < -0.001 and abs(avg_return) > volatility:
                    regime = 'bearish'
                else:
                    regime = 'neutral'
            
            self.market_regime = regime
            self.logger.info(f"Detected market regime: {regime}")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'neutral'
    
    def calculate_volatility_adjustment(self, data: pd.DataFrame) -> float:
        """
        Calculate volatility adjustment factor.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            float: The volatility adjustment factor.
        """
        try:
            if not self.volatility_scaling or data.empty:
                return 1.0
            
            # Calculate recent volatility
            if 'atr' in data.columns:
                # Use ATR-based volatility
                current_price = data['close'].iloc[-1]
                atr = data['atr'].iloc[-1]
                
                # Convert to percentage volatility
                volatility = atr / current_price
            else:
                # Use return-based volatility
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()
            
            # Annualize volatility (assuming daily data)
            annual_volatility = volatility * np.sqrt(252)
            
            # Calculate adjustment factor
            if annual_volatility > 0:
                adjustment = self.target_volatility / annual_volatility
            else:
                adjustment = 1.0
            
            # Limit adjustment range
            adjustment = max(0.2, min(adjustment, 5.0))
            
            self.logger.info(f"Volatility adjustment factor: {adjustment:.2f} (volatility: {annual_volatility:.2f}, target: {self.target_volatility:.2f})")
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
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
            # Check circuit breaker
            if self.check_circuit_breaker():
                self.logger.warning("Circuit breaker active - no new positions allowed")
                return 0.0
            
            # Detect market regime
            market_regime = self.detect_market_regime(data)
            
            # Calculate base position size
            if self.use_kelly:
                # Use Kelly Criterion
                win_rate = self.get_win_rate()
                win_loss_ratio = self.get_win_loss_ratio()
                kelly_allocation = self.calculate_kelly_criterion(win_rate, win_loss_ratio)
                
                # Adjust based on market regime
                if market_regime == 'bullish':
                    kelly_allocation *= 1.2  # Increase allocation in bullish regime
                elif market_regime == 'bearish':
                    kelly_allocation *= 0.8  # Decrease allocation in bearish regime
                
                position_value = available_capital * kelly_allocation
            else:
                # Use standard position sizing
                position_value = available_capital * self.max_position_size_pct
            
            # Apply volatility adjustment
            volatility_adjustment = self.calculate_volatility_adjustment(data)
            position_value *= volatility_adjustment
            
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
            return super().calculate_position_size(data, current_price, available_capital)
    
    def calculate_correlation_adjustment(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Calculate correlation-based position size adjustment.
        
        Args:
            symbol (str): The symbol to calculate correlation for.
            data (pd.DataFrame): The market data.
            
        Returns:
            float: The correlation adjustment factor.
        """
        try:
            # If no other assets, no adjustment needed
            if not self.asset_correlations:
                return 1.0
            
            # Calculate average correlation with other assets
            correlations = list(self.asset_correlations.values())
            if not correlations:
                return 1.0
            
            avg_correlation = sum(correlations) / len(correlations)
            
            # Adjust position size based on correlation
            # Higher correlation = lower position size
            adjustment = 1.0 - (avg_correlation * 0.5)
            
            # Ensure adjustment is within reasonable bounds
            adjustment = max(0.5, min(adjustment, 1.5))
            
            self.logger.info(f"Correlation adjustment for {symbol}: {adjustment:.2f} (avg correlation: {avg_correlation:.2f})")
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0
    
    def update_correlations(self, symbol: str, data: pd.DataFrame, other_assets: Dict[str, pd.DataFrame]) -> None:
        """
        Update asset correlations.
        
        Args:
            symbol (str): The symbol to update correlations for.
            data (pd.DataFrame): The market data for the symbol.
            other_assets (Dict[str, pd.DataFrame]): The market data for other assets.
        """
        try:
            if data.empty:
                return
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate correlations with other assets
            for other_symbol, other_data in other_assets.items():
                if other_data.empty:
                    continue
                
                other_returns = other_data['close'].pct_change().dropna()
                
                # Ensure returns have the same length
                min_length = min(len(returns), len(other_returns))
                if min_length < 10:  # Need at least 10 data points
                    continue
                
                # Calculate correlation
                correlation = returns.iloc[-min_length:].corr(other_returns.iloc[-min_length:])
                
                # Store correlation
                self.asset_correlations[other_symbol] = correlation
            
            self.logger.info(f"Updated correlations for {symbol}: {self.asset_correlations}")
            
        except Exception as e:
            self.logger.error(f"Error updating correlations: {e}")
    
    def update_trade_stats(self, pnl: float) -> None:
        """
        Update trade statistics.
        
        Args:
            pnl (float): The profit/loss amount.
        """
        # Update base stats
        super().update_trade_stats(pnl)
        
        # Update advanced stats
        self.update_win_loss_history(pnl)
    
    def check_trade_limits(self) -> bool:
        """
        Check if trade limits have been reached.
        
        Returns:
            bool: True if trade limits have not been reached, False otherwise.
        """
        # Check circuit breaker first
        if self.check_circuit_breaker():
            self.logger.warning("Trade rejected: Circuit breaker active")
            return False
        
        # Check base limits
        return super().check_trade_limits()
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get risk metrics.
        
        Returns:
            Dict[str, Any]: The risk metrics.
        """
        try:
            metrics = {
                'win_rate': self.get_win_rate(),
                'win_loss_ratio': self.get_win_loss_ratio(),
                'trades_today': self.trades_today,
                'daily_pnl': self.daily_pnl,
                'market_regime': self.market_regime,
                'circuit_breaker': self.circuit_breaker_triggered
            }
            
            # Add drawdown if available
            if self.drawdown_history:
                metrics['current_drawdown'] = self.drawdown_history[-1]
                metrics['max_drawdown'] = max(self.drawdown_history)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}
