"""
Strategy validation module for the ELVIS trading system.
Implements Monte Carlo simulations, walk-forward analysis, and statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 1000
    confidence_level: float = 0.95
    market_conditions: List[str] = None
    stress_test_scenarios: List[str] = None
    random_seed: int = 42

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    n_splits: int = 5
    train_size: int = 200
    test_size: int = 50
    optimization_window: int = 20
    regime_based: bool = True
    regime_threshold: float = 0.7

@dataclass
class StatisticalConfig:
    """Configuration for statistical validation."""
    tests: List[str] = None
    bootstrap_samples: int = 1000
    significance_level: float = 0.05
    metrics: List[str] = None

class StrategyValidator:
    """Validates trading strategies using various methods."""
    
    def __init__(self,
                 mc_config: MonteCarloConfig = None,
                 wf_config: WalkForwardConfig = None,
                 stats_config: StatisticalConfig = None):
        """
        Initialize the strategy validator.
        
        Args:
            mc_config: Monte Carlo simulation configuration
            wf_config: Walk-forward analysis configuration
            stats_config: Statistical validation configuration
        """
        self.mc_config = mc_config or MonteCarloConfig()
        self.wf_config = wf_config or WalkForwardConfig()
        self.stats_config = stats_config or StatisticalConfig()
        
        if self.mc_config.market_conditions is None:
            self.mc_config.market_conditions = [
                'normal', 'high_volatility', 'low_volatility',
                'trending_up', 'trending_down', 'mean_reverting'
            ]
            
        if self.mc_config.stress_test_scenarios is None:
            self.mc_config.stress_test_scenarios = [
                'flash_crash', 'liquidity_crisis', 'market_crash',
                'volatility_spike', 'correlation_breakdown'
            ]
            
        if self.stats_config.tests is None:
            self.stats_config.tests = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'max_drawdown', 'var', 'cvar', 'white_reality_check'
            ]
            
        if self.stats_config.metrics is None:
            self.stats_config.metrics = [
                'returns', 'volatility', 'sharpe_ratio',
                'max_drawdown', 'win_rate', 'profit_factor'
            ]
            
        self.logger = logging.getLogger(__name__)
        np.random.seed(self.mc_config.random_seed)
        
    def run_monte_carlo_simulation(self,
                                 strategy: callable,
                                 initial_capital: float,
                                 data: pd.DataFrame,
                                 params: Dict) -> Dict:
        """
        Run Monte Carlo simulation for strategy validation.
        
        Args:
            strategy: Strategy function to test
            initial_capital: Initial capital
            data: Historical data
            params: Strategy parameters
            
        Returns:
            Dictionary with simulation results
        """
        results = {
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': [],
            'profit_factors': []
        }
        
        for _ in tqdm(range(self.mc_config.n_simulations), desc="Running Monte Carlo simulations"):
            # Generate random market conditions
            market_condition = np.random.choice(self.mc_config.market_conditions)
            simulated_data = self._simulate_market_data(data, market_condition)
            
            # Run strategy on simulated data
            strategy_results = strategy(simulated_data, initial_capital, params)
            
            # Store results
            results['returns'].append(strategy_results['returns'])
            results['sharpe_ratios'].append(strategy_results['sharpe_ratio'])
            results['max_drawdowns'].append(strategy_results['max_drawdown'])
            results['win_rates'].append(strategy_results['win_rate'])
            results['profit_factors'].append(strategy_results['profit_factor'])
            
        # Calculate statistics
        stats = {
            'mean_returns': np.mean(results['returns']),
            'std_returns': np.std(results['returns']),
            'mean_sharpe': np.mean(results['sharpe_ratios']),
            'mean_drawdown': np.mean(results['max_drawdowns']),
            'mean_win_rate': np.mean(results['win_rates']),
            'mean_profit_factor': np.mean(results['profit_factors']),
            'confidence_intervals': self._calculate_confidence_intervals(results)
        }
        
        return stats
        
    def run_stress_tests(self,
                        strategy: callable,
                        initial_capital: float,
                        data: pd.DataFrame,
                        params: Dict) -> Dict:
        """
        Run stress tests under extreme market conditions.
        
        Args:
            strategy: Strategy function to test
            initial_capital: Initial capital
            data: Historical data
            params: Strategy parameters
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario in tqdm(self.mc_config.stress_test_scenarios, desc="Running stress tests"):
            # Generate extreme market conditions
            stress_data = self._simulate_stress_scenario(data, scenario)
            
            # Run strategy on stress data
            strategy_results = strategy(stress_data, initial_capital, params)
            
            results[scenario] = {
                'returns': strategy_results['returns'],
                'max_drawdown': strategy_results['max_drawdown'],
                'sharpe_ratio': strategy_results['sharpe_ratio'],
                'survival': strategy_results['survival']  # Whether strategy survived the stress test
            }
            
        return results
        
    def run_walk_forward_analysis(self,
                                strategy: callable,
                                data: pd.DataFrame,
                                params: Dict) -> Dict:
        """
        Run walk-forward analysis with adaptive parameter optimization.
        
        Args:
            strategy: Strategy function to test
            data: Historical data
            params: Strategy parameters
            
        Returns:
            Dictionary with walk-forward results
        """
        tscv = TimeSeriesSplit(
            n_splits=self.wf_config.n_splits,
            test_size=self.wf_config.test_size
        )
        
        results = {
            'train_metrics': [],
            'test_metrics': [],
            'optimized_params': []
        }
        
        for train_idx, test_idx in tqdm(tscv.split(data), desc="Running walk-forward analysis"):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Optimize parameters on training data
            if self.wf_config.regime_based:
                optimized_params = self._optimize_parameters_regime_based(
                    strategy, train_data, params
                )
            else:
                optimized_params = self._optimize_parameters(
                    strategy, train_data, params
                )
            
            # Test on out-of-sample data
            train_metrics = strategy(train_data, params=optimized_params)
            test_metrics = strategy(test_data, params=optimized_params)
            
            results['train_metrics'].append(train_metrics)
            results['test_metrics'].append(test_metrics)
            results['optimized_params'].append(optimized_params)
            
        return results
        
    def run_statistical_tests(self,
                            strategy: callable,
                            data: pd.DataFrame,
                            params: Dict) -> Dict:
        """
        Run statistical tests for strategy validation.
        
        Args:
            strategy: Strategy function to test
            data: Historical data
            params: Strategy parameters
            
        Returns:
            Dictionary with statistical test results
        """
        results = {}
        
        # Run strategy on historical data
        strategy_results = strategy(data, params=params)
        returns = strategy_results['returns']
        
        # Calculate various statistical tests
        for test in self.stats_config.tests:
            if test == 'sharpe_ratio':
                results[test] = self._calculate_sharpe_ratio(returns)
            elif test == 'sortino_ratio':
                results[test] = self._calculate_sortino_ratio(returns)
            elif test == 'calmar_ratio':
                results[test] = self._calculate_calmar_ratio(returns, strategy_results['max_drawdown'])
            elif test == 'var':
                results[test] = self._calculate_var(returns)
            elif test == 'cvar':
                results[test] = self._calculate_cvar(returns)
            elif test == 'white_reality_check':
                results[test] = self._run_white_reality_check(returns)
                
        return results
        
    def _simulate_market_data(self,
                             data: pd.DataFrame,
                             market_condition: str) -> pd.DataFrame:
        """Simulate market data for given condition."""
        # Implementation depends on specific market condition
        # This is a placeholder for the actual implementation
        return data
        
    def _simulate_stress_scenario(self,
                                data: pd.DataFrame,
                                scenario: str) -> pd.DataFrame:
        """Simulate stress scenario data."""
        # Implementation depends on specific stress scenario
        # This is a placeholder for the actual implementation
        return data
        
    def _optimize_parameters(self,
                           strategy: callable,
                           data: pd.DataFrame,
                           params: Dict) -> Dict:
        """Optimize strategy parameters."""
        # Implementation of parameter optimization
        # This is a placeholder for the actual implementation
        return params
        
    def _optimize_parameters_regime_based(self,
                                        strategy: callable,
                                        data: pd.DataFrame,
                                        params: Dict) -> Dict:
        """Optimize parameters based on market regime."""
        # Implementation of regime-based parameter optimization
        # This is a placeholder for the actual implementation
        return params
        
    def _calculate_confidence_intervals(self, results: Dict) -> Dict:
        """Calculate confidence intervals for simulation results."""
        intervals = {}
        for metric, values in results.items():
            mean = np.mean(values)
            std = np.std(values)
            z_score = stats.norm.ppf((1 + self.mc_config.confidence_level) / 2)
            margin = z_score * std / np.sqrt(len(values))
            intervals[metric] = {
                'lower': mean - margin,
                'upper': mean + margin
            }
        return intervals
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        return np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        
    def _calculate_calmar_ratio(self,
                              returns: np.ndarray,
                              max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return np.inf
        return np.mean(returns) * 252 / abs(max_drawdown)
        
    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, 5)
        
    def _calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns)
        return np.mean(returns[returns <= var])
        
    def _run_white_reality_check(self, returns: np.ndarray) -> Dict:
        """Run White's Reality Check for strategy significance."""
        # Implementation of White's Reality Check
        # This is a placeholder for the actual implementation
        return {'p_value': 0.05, 'significant': True}
        
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot validation results."""
        # Implementation of plotting functions
        # This is a placeholder for the actual implementation
        pass
        
    def save_results(self, results: Dict, path: str):
        """Save validation results to file."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def load_results(self, path: str) -> Dict:
        """Load validation results from file."""
        with open(path, 'r') as f:
            return json.load(f) 