"""
Risk management module for the ELVIS trading system.
Implements advanced position sizing, drawdown protection, and correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"

@dataclass
class PositionSizingParams:
    """Parameters for position sizing."""
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    min_position_size: float = 0.01  # Minimum position size
    kelly_fraction: float = 0.5  # Fraction of Kelly criterion to use
    volatility_scaling: bool = True  # Whether to scale positions based on volatility
    regime_based_scaling: bool = True  # Whether to adjust position sizes based on market regime

@dataclass
class DrawdownParams:
    """Parameters for drawdown protection."""
    max_drawdown: float = 0.2  # Maximum allowed drawdown
    circuit_breaker_levels: List[float] = None  # Drawdown levels for circuit breakers
    volatility_threshold: float = 0.3  # Volatility threshold for position scaling
    pause_threshold: float = 0.4  # Volatility threshold for trading pause

@dataclass
class CorrelationParams:
    """Parameters for correlation analysis."""
    lookback_window: int = 60  # Days to look back for correlation calculation
    correlation_threshold: float = 0.7  # Threshold for high correlation
    min_pairs_distance: float = 2.0  # Minimum standard deviations for pair trading
    rebalance_frequency: int = 5  # Days between correlation updates

class RiskManager:
    """Manages risk for the trading system."""
    
    def __init__(self, 
                 position_params: PositionSizingParams = None,
                 drawdown_params: DrawdownParams = None,
                 correlation_params: CorrelationParams = None):
        """
        Initialize the risk manager.
        
        Args:
            position_params: Position sizing parameters
            drawdown_params: Drawdown protection parameters
            correlation_params: Correlation analysis parameters
        """
        self.position_params = position_params or PositionSizingParams()
        self.drawdown_params = drawdown_params or DrawdownParams()
        self.correlation_params = correlation_params or CorrelationParams()
        
        if self.drawdown_params.circuit_breaker_levels is None:
            self.drawdown_params.circuit_breaker_levels = [
                0.1, 0.15, 0.2
            ]
            
        self.logger = logging.getLogger(__name__)
        self.current_regime = MarketRegime.SIDEWAYS
        self.trading_paused = False
        self.correlations = {}
        self.pairs = {}
        
    def calculate_kelly_position(self, 
                               win_rate: float,
                               win_loss_ratio: float,
                               current_volatility: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning trades
            win_loss_ratio: Ratio of average win to average loss
            current_volatility: Current market volatility
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        try:
            # Basic Kelly formula
            kelly = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Adjust for volatility
            if self.position_params.volatility_scaling:
                volatility_factor = 1 / (1 + current_volatility)
                kelly *= volatility_factor
                
            # Apply regime-based scaling
            if self.position_params.regime_based_scaling:
                regime_factor = self._get_regime_scaling_factor()
                kelly *= regime_factor
                
            # Apply fraction and bounds
            kelly = kelly * self.position_params.kelly_fraction
            kelly = np.clip(
                kelly,
                self.position_params.min_position_size,
                self.position_params.max_position_size
            )
            
            return kelly
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position: {str(e)}")
            return self.position_params.min_position_size
            
    def _get_regime_scaling_factor(self) -> float:
        """Get position scaling factor based on market regime."""
        regime_factors = {
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.LOW_VOLATILITY: 1.2,
            MarketRegime.TRENDING_UP: 1.1,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.SIDEWAYS: 0.9
        }
        return regime_factors[self.current_regime]
        
    def detect_market_regime(self, 
                           prices: pd.Series,
                           volatility: float) -> MarketRegime:
        """
        Detect current market regime based on price action and volatility.
        
        Args:
            prices: Price series
            volatility: Current market volatility
            
        Returns:
            Detected market regime
        """
        try:
            # Calculate trend
            returns = prices.pct_change()
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            # Determine trend
            if sma_20.iloc[-1] > sma_50.iloc[-1] and returns.mean() > 0:
                trend = MarketRegime.TRENDING_UP
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and returns.mean() < 0:
                trend = MarketRegime.TRENDING_DOWN
            else:
                trend = MarketRegime.SIDEWAYS
                
            # Adjust for volatility
            if volatility > self.drawdown_params.volatility_threshold:
                if trend in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                    return MarketRegime.HIGH_VOLATILITY
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < self.drawdown_params.volatility_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY
                
            return trend
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS
            
    def check_drawdown_protection(self,
                                portfolio_value: float,
                                peak_value: float,
                                current_volatility: float) -> Tuple[bool, str]:
        """
        Check if drawdown protection measures should be triggered.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            current_volatility: Current market volatility
            
        Returns:
            Tuple of (should_pause, reason)
        """
        try:
            # Calculate drawdown
            drawdown = (peak_value - portfolio_value) / peak_value
            
            # Check circuit breakers
            for level in sorted(self.drawdown_params.circuit_breaker_levels):
                if drawdown >= level:
                    return True, f"Circuit breaker triggered at {level*100}% drawdown"
                    
            # Check volatility pause
            if current_volatility >= self.drawdown_params.pause_threshold:
                return True, "Volatility exceeds pause threshold"
                
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown protection: {str(e)}")
            return True, "Error in drawdown protection check"
            
    def calculate_correlations(self,
                             asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between assets.
        
        Args:
            asset_returns: DataFrame of asset returns
            
        Returns:
            Correlation matrix
        """
        try:
            # Use Ledoit-Wolf shrinkage for more robust correlation estimation
            cov = LedoitWolf().fit(asset_returns)
            correlations = pd.DataFrame(
                cov.correlation_,
                index=asset_returns.columns,
                columns=asset_returns.columns
            )
            
            self.correlations = correlations
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()
            
    def find_trading_pairs(self,
                          asset_returns: pd.DataFrame,
                          correlations: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find potential pairs for pair trading.
        
        Args:
            asset_returns: DataFrame of asset returns
            correlations: Correlation matrix
            
        Returns:
            List of (asset1, asset2) pairs
        """
        try:
            pairs = []
            assets = asset_returns.columns
            
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    # Check correlation
                    if correlations.loc[asset1, asset2] > self.correlation_params.correlation_threshold:
                        # Calculate spread
                        spread = asset_returns[asset1] - asset_returns[asset2]
                        spread_std = spread.std()
                        
                        # Check if spread is wide enough
                        current_spread = spread.iloc[-1]
                        if abs(current_spread) > self.correlation_params.min_pairs_distance * spread_std:
                            pairs.append((asset1, asset2))
                            
            self.pairs = pairs
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error finding trading pairs: {str(e)}")
            return []
            
    def optimize_portfolio_weights(self,
                                 asset_returns: pd.DataFrame,
                                 target_volatility: float = 0.15) -> pd.Series:
        """
        Optimize portfolio weights for risk-adjusted returns.
        
        Args:
            asset_returns: DataFrame of asset returns
            target_volatility: Target portfolio volatility
            
        Returns:
            Series of optimized weights
        """
        try:
            # Calculate covariance matrix
            cov = LedoitWolf().fit(asset_returns)
            cov_matrix = pd.DataFrame(
                cov.covariance_,
                index=asset_returns.columns,
                columns=asset_returns.columns
            )
            
            # Calculate inverse volatility weights
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1 / vols
            weights = inv_vols / inv_vols.sum()
            
            # Scale to target volatility
            current_vol = np.sqrt(weights @ cov_matrix @ weights)
            scaling_factor = target_volatility / current_vol
            weights = weights * scaling_factor
            
            # Normalize weights
            weights = weights / weights.sum()
            
            return pd.Series(weights, index=asset_returns.columns)
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio weights: {str(e)}")
            return pd.Series(1/len(asset_returns.columns), index=asset_returns.columns)
            
    def get_position_sizes(self,
                          portfolio_value: float,
                          asset_returns: pd.DataFrame,
                          win_rates: Dict[str, float],
                          win_loss_ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate position sizes for all assets.
        
        Args:
            portfolio_value: Current portfolio value
            asset_returns: DataFrame of asset returns
            win_rates: Dictionary of win rates by asset
            win_loss_ratios: Dictionary of win/loss ratios by asset
            
        Returns:
            Dictionary of position sizes by asset
        """
        try:
            # Get market regime
            self.current_regime = self.detect_market_regime(
                asset_returns.mean(axis=1),
                asset_returns.std().mean()
            )
            
            # Calculate correlations
            correlations = self.calculate_correlations(asset_returns)
            
            # Get portfolio weights
            weights = self.optimize_portfolio_weights(asset_returns)
            
            # Calculate position sizes
            position_sizes = {}
            for asset in asset_returns.columns:
                # Get Kelly position
                kelly_size = self.calculate_kelly_position(
                    win_rates[asset],
                    win_loss_ratios[asset],
                    asset_returns[asset].std()
                )
                
                # Adjust for portfolio weight
                position_size = kelly_size * weights[asset]
                
                # Apply bounds
                position_size = np.clip(
                    position_size,
                    self.position_params.min_position_size,
                    self.position_params.max_position_size
                )
                
                position_sizes[asset] = position_size * portfolio_value
                
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {str(e)}")
            return {asset: self.position_params.min_position_size * portfolio_value 
                   for asset in asset_returns.columns} 