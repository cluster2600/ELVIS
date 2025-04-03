"""
Value at Risk (VaR) Calculator for portfolio risk assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

class VaRCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_data = {}
        self.positions = {}
        self.correlation_matrix = None

    def add_position(self, symbol: str, size: float, price: float):
        """Add a position to the portfolio."""
        self.positions[symbol] = {
            'size': size,
            'price': price,
            'value': size * price,
            'last_updated': datetime.now()
        }
        self._update_correlation_matrix()

    def update_position(self, symbol: str, size: Optional[float] = None, price: Optional[float] = None):
        """Update an existing position."""
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} not found")
            
        if size is not None:
            self.positions[symbol]['size'] = size
        if price is not None:
            self.positions[symbol]['price'] = price
            
        self.positions[symbol]['value'] = self.positions[symbol]['size'] * self.positions[symbol]['price']
        self.positions[symbol]['last_updated'] = datetime.now()
        self._update_correlation_matrix()

    def add_historical_data(self, symbol: str, returns: List[float], timestamps: List[datetime]):
        """Add historical returns data for a symbol."""
        self.historical_data[symbol] = pd.DataFrame({
            'timestamp': timestamps,
            'returns': returns
        }).set_index('timestamp')
        self._update_correlation_matrix()

    def _update_correlation_matrix(self):
        """Update the correlation matrix based on available historical data."""
        symbols = list(self.positions.keys())
        if len(symbols) < 2:
            return
            
        # Get common time period for all symbols
        common_period = None
        for symbol in symbols:
            if symbol in self.historical_data:
                if common_period is None:
                    common_period = self.historical_data[symbol].index
                else:
                    common_period = common_period.intersection(self.historical_data[symbol].index)
                    
        if common_period is None or len(common_period) < 2:
            return
            
        # Calculate correlation matrix
        returns_data = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.historical_data:
                returns_data[symbol] = self.historical_data[symbol].loc[common_period, 'returns']
                
        self.correlation_matrix = returns_data.corr()

    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> Dict:
        """Calculate Value at Risk for the portfolio."""
        if not self.positions:
            return {
                'var': 0.0,
                'components': {},
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            
        # Calculate individual position VaR
        position_vars = {}
        total_var = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.historical_data:
                returns = self.historical_data[symbol]['returns']
                position_value = position['value']
                
                # Calculate position VaR using historical simulation
                var = self._calculate_position_var(returns, position_value, confidence_level, time_horizon)
                position_vars[symbol] = var
                total_var += var ** 2
                
        # Adjust for correlations if available
        if self.correlation_matrix is not None and len(position_vars) > 1:
            total_var = self._adjust_for_correlations(position_vars)
            
        return {
            'var': np.sqrt(total_var),
            'components': position_vars,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }

    def _calculate_position_var(self, returns: pd.Series, position_value: float, 
                              confidence_level: float, time_horizon: int) -> float:
        """Calculate VaR for a single position."""
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Calculate VaR using historical simulation
        var = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
        return var * position_value

    def _adjust_for_correlations(self, position_vars: Dict[str, float]) -> float:
        """Adjust portfolio VaR for correlations between positions."""
        symbols = list(position_vars.keys())
        n = len(symbols)
        
        # Create variance-covariance matrix
        var_covar = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    var_covar[i, j] = position_vars[symbols[i]] ** 2
                else:
                    correlation = self.correlation_matrix.loc[symbols[i], symbols[j]]
                    var_covar[i, j] = position_vars[symbols[i]] * position_vars[symbols[j]] * correlation
                    
        return np.sum(var_covar)

    def calculate_expected_shortfall(self, confidence_level: float = 0.95, 
                                   time_horizon: int = 1) -> Dict:
        """Calculate Expected Shortfall (CVaR) for the portfolio."""
        if not self.positions:
            return {
                'expected_shortfall': 0.0,
                'components': {},
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        if portfolio_returns is None or len(portfolio_returns) < 2:
            return {
                'expected_shortfall': 0.0,
                'components': {},
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            
        # Scale returns for time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate Expected Shortfall
        var_threshold = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        tail_returns = scaled_returns[scaled_returns <= var_threshold]
        expected_shortfall = -np.mean(tail_returns)
        
        return {
            'expected_shortfall': expected_shortfall,
            'var_threshold': -var_threshold,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }

    def _calculate_portfolio_returns(self) -> Optional[pd.Series]:
        """Calculate historical portfolio returns."""
        if not self.positions or not self.historical_data:
            return None
            
        # Get common time period
        common_period = None
        for symbol in self.positions:
            if symbol in self.historical_data:
                if common_period is None:
                    common_period = self.historical_data[symbol].index
                else:
                    common_period = common_period.intersection(self.historical_data[symbol].index)
                    
        if common_period is None or len(common_period) < 2:
            return None
            
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=common_period)
        total_value = sum(pos['value'] for pos in self.positions.values())
        
        for symbol, position in self.positions.items():
            if symbol in self.historical_data:
                weight = position['value'] / total_value
                portfolio_returns += weight * self.historical_data[symbol].loc[common_period, 'returns']
                
        return portfolio_returns 