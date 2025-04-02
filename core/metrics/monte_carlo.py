"""
Monte Carlo simulation for the ELVIS project.
This module provides functionality for Monte Carlo simulations to test strategy robustness.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from config import FILE_PATHS

class MonteCarloSimulator:
    """
    Monte Carlo simulator for testing strategy robustness.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger
        
        # Simulation parameters
        self.num_simulations = kwargs.get('num_simulations', 1000)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.initial_capital = kwargs.get('initial_capital', 10000.0)
        self.transaction_fee = kwargs.get('transaction_fee', 0.001)
        self.parallel = kwargs.get('parallel', True)
        self.num_processes = kwargs.get('num_processes', mp.cpu_count())
        
        # Output parameters
        self.output_dir = kwargs.get('output_dir', FILE_PATHS['METRICS_DIR'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _simulate_returns(self, returns: np.ndarray, win_rate: float, win_loss_ratio: float, num_trades: int) -> np.ndarray:
        """
        Simulate returns for a single Monte Carlo run.
        
        Args:
            returns (np.ndarray): The historical returns.
            win_rate (float): The win rate.
            win_loss_ratio (float): The win/loss ratio.
            num_trades (int): The number of trades to simulate.
            
        Returns:
            np.ndarray: The simulated returns.
        """
        # Generate random trade outcomes
        wins = np.random.random(num_trades) < win_rate
        
        # Calculate average win and loss
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
        avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else -0.01
        
        # If win/loss ratio is provided, adjust avg_win and avg_loss
        if win_loss_ratio > 0:
            avg_loss = -avg_win / win_loss_ratio
        
        # Generate returns based on win/loss
        simulated_returns = np.zeros(num_trades)
        simulated_returns[wins] = np.random.normal(avg_win, avg_win / 2, np.sum(wins))
        simulated_returns[~wins] = np.random.normal(avg_loss, abs(avg_loss) / 2, np.sum(~wins))
        
        return simulated_returns
    
    def _simulate_equity_curve(self, returns: np.ndarray) -> np.ndarray:
        """
        Simulate equity curve from returns.
        
        Args:
            returns (np.ndarray): The returns.
            
        Returns:
            np.ndarray: The equity curve.
        """
        # Apply transaction fee
        net_returns = returns - self.transaction_fee
        
        # Calculate equity curve
        equity_curve = self.initial_capital * np.cumprod(1 + net_returns)
        
        return equity_curve
    
    def _calculate_metrics(self, equity_curve: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics from equity curve.
        
        Args:
            equity_curve (np.ndarray): The equity curve.
            returns (np.ndarray): The returns.
            
        Returns:
            Dict[str, float]: The performance metrics.
        """
        # Calculate metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate Sharpe ratio (assuming daily returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate Sortino ratio (assuming daily returns)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Calculate profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calculate maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for r in returns:
            if r < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Return metrics
        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _run_single_simulation(self, returns: np.ndarray, win_rate: float, win_loss_ratio: float, num_trades: int) -> Dict[str, float]:
        """
        Run a single Monte Carlo simulation.
        
        Args:
            returns (np.ndarray): The historical returns.
            win_rate (float): The win rate.
            win_loss_ratio (float): The win/loss ratio.
            num_trades (int): The number of trades to simulate.
            
        Returns:
            Dict[str, float]: The simulation results.
        """
        # Simulate returns
        simulated_returns = self._simulate_returns(returns, win_rate, win_loss_ratio, num_trades)
        
        # Simulate equity curve
        equity_curve = self._simulate_equity_curve(simulated_returns)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, simulated_returns)
        
        return metrics
    
    def run_simulation(self, returns: np.ndarray, win_rate: Optional[float] = None, win_loss_ratio: Optional[float] = None, num_trades: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns (np.ndarray): The historical returns.
            win_rate (float, optional): The win rate. If None, calculated from returns.
            win_loss_ratio (float, optional): The win/loss ratio. If None, calculated from returns.
            num_trades (int, optional): The number of trades to simulate. If None, uses length of returns.
            
        Returns:
            Dict[str, Any]: The simulation results.
        """
        try:
            self.logger.info(f"Running Monte Carlo simulation with {self.num_simulations} iterations")
            
            # Calculate win rate and win/loss ratio if not provided
            if win_rate is None:
                win_rate = np.sum(returns > 0) / len(returns)
            
            if win_loss_ratio is None:
                avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
                avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.01
                win_loss_ratio = avg_win / avg_loss
            
            # Set number of trades if not provided
            if num_trades is None:
                num_trades = len(returns)
            
            self.logger.info(f"Simulation parameters: win_rate={win_rate:.2f}, win_loss_ratio={win_loss_ratio:.2f}, num_trades={num_trades}")
            
            # Run simulations
            simulation_results = []
            
            if self.parallel and self.num_processes > 1:
                # Parallel simulation
                self.logger.info(f"Running simulations in parallel with {self.num_processes} processes")
                
                # Create partial function
                sim_func = partial(
                    self._run_single_simulation,
                    returns=returns,
                    win_rate=win_rate,
                    win_loss_ratio=win_loss_ratio,
                    num_trades=num_trades
                )
                
                # Run simulations in parallel
                with mp.Pool(processes=self.num_processes) as pool:
                    simulation_results = list(tqdm(
                        pool.imap(lambda _: sim_func(), range(self.num_simulations)),
                        total=self.num_simulations,
                        desc="Running simulations"
                    ))
            else:
                # Sequential simulation
                for i in tqdm(range(self.num_simulations), desc="Running simulations"):
                    result = self._run_single_simulation(returns, win_rate, win_loss_ratio, num_trades)
                    simulation_results.append(result)
            
            # Aggregate results
            aggregated_results = self._aggregate_results(simulation_results)
            
            # Generate plots
            self._generate_plots(simulation_results)
            
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Error running Monte Carlo simulation: {e}")
            return {}
    
    def _aggregate_results(self, simulation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Aggregate simulation results.
        
        Args:
            simulation_results (List[Dict[str, float]]): The simulation results.
            
        Returns:
            Dict[str, Any]: The aggregated results.
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(simulation_results)
        
        # Calculate statistics
        stats = {}
        
        for column in results_df.columns:
            values = results_df[column].values
            
            # Calculate statistics
            mean = np.mean(values)
            median = np.median(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_ci = np.percentile(values, lower_percentile)
            upper_ci = np.percentile(values, upper_percentile)
            
            # Store statistics
            stats[column] = {
                'mean': mean,
                'median': median,
                'std': std,
                'min': min_val,
                'max': max_val,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            }
        
        # Calculate probability of profit
        profit_probability = np.mean(results_df['total_return'] > 0)
        
        # Calculate probability of drawdown exceeding thresholds
        drawdown_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
        drawdown_probabilities = {}
        
        for threshold in drawdown_thresholds:
            probability = np.mean(results_df['max_drawdown'] > threshold)
            drawdown_probabilities[f'drawdown_{int(threshold * 100)}pct'] = probability
        
        # Return aggregated results
        return {
            'statistics': stats,
            'profit_probability': profit_probability,
            'drawdown_probabilities': drawdown_probabilities,
            'num_simulations': self.num_simulations,
            'confidence_level': self.confidence_level
        }
    
    def _generate_plots(self, simulation_results: List[Dict[str, float]]) -> None:
        """
        Generate plots from simulation results.
        
        Args:
            simulation_results (List[Dict[str, float]]): The simulation results.
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(simulation_results)
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'monte_carlo_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot total return distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['total_return'], kde=True)
        plt.title('Total Return Distribution')
        plt.xlabel('Total Return')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'total_return_distribution.png'))
        plt.close()
        
        # Plot max drawdown distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['max_drawdown'], kde=True)
        plt.title('Maximum Drawdown Distribution')
        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'max_drawdown_distribution.png'))
        plt.close()
        
        # Plot Sharpe ratio distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['sharpe_ratio'], kde=True)
        plt.title('Sharpe Ratio Distribution')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.axvline(x=1, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'sharpe_ratio_distribution.png'))
        plt.close()
        
        # Plot profit factor distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['profit_factor'].clip(upper=5), kde=True)  # Clip to avoid extreme values
        plt.title('Profit Factor Distribution')
        plt.xlabel('Profit Factor')
        plt.ylabel('Frequency')
        plt.axvline(x=1, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'profit_factor_distribution.png'))
        plt.close()
        
        # Plot scatter plot of return vs drawdown
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['max_drawdown'], results_df['total_return'], alpha=0.5)
        plt.title('Return vs Drawdown')
        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Total Return')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'return_vs_drawdown.png'))
        plt.close()
        
        # Plot scatter plot of Sharpe ratio vs drawdown
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['max_drawdown'], results_df['sharpe_ratio'], alpha=0.5)
        plt.title('Sharpe Ratio vs Drawdown')
        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'sharpe_vs_drawdown.png'))
        plt.close()
    
    def simulate_strategy(self, strategy_name: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate a strategy based on historical trades.
        
        Args:
            strategy_name (str): The name of the strategy.
            trades (List[Dict[str, Any]]): The historical trades.
            
        Returns:
            Dict[str, Any]: The simulation results.
        """
        try:
            self.logger.info(f"Simulating strategy: {strategy_name}")
            
            # Extract returns from trades
            returns = []
            
            for trade in trades:
                if 'pnl' in trade and 'entry_price' in trade:
                    pnl = trade['pnl']
                    entry_price = trade['entry_price']
                    
                    # Calculate return
                    trade_return = pnl / entry_price
                    returns.append(trade_return)
            
            if not returns:
                self.logger.warning("No returns found in trades")
                return {}
            
            # Convert to numpy array
            returns = np.array(returns)
            
            # Calculate win rate and win/loss ratio
            win_rate = np.sum(returns > 0) / len(returns)
            
            avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.01
            avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.01
            win_loss_ratio = avg_win / avg_loss
            
            # Run simulation
            results = self.run_simulation(returns, win_rate, win_loss_ratio)
            
            # Save results
            self._save_results(strategy_name, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error simulating strategy: {e}")
            return {}
    
    def _save_results(self, strategy_name: str, results: Dict[str, Any]) -> None:
        """
        Save simulation results.
        
        Args:
            strategy_name (str): The name of the strategy.
            results (Dict[str, Any]): The simulation results.
        """
        try:
            # Create results directory
            results_dir = os.path.join(self.output_dir, 'monte_carlo_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results as JSON
            results_file = os.path.join(results_dir, f'{strategy_name}_monte_carlo.json')
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Saved simulation results to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving simulation results: {e}")
    
    def compare_strategies(self, strategies: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategies (Dict[str, List[Dict[str, Any]]]): The strategies to compare.
            
        Returns:
            Dict[str, Any]: The comparison results.
        """
        try:
            self.logger.info(f"Comparing {len(strategies)} strategies")
            
            # Simulate each strategy
            results = {}
            
            for strategy_name, trades in strategies.items():
                strategy_results = self.simulate_strategy(strategy_name, trades)
                results[strategy_name] = strategy_results
            
            # Compare strategies
            comparison = self._compare_results(results)
            
            # Generate comparison plots
            self._generate_comparison_plots(results)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def _compare_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare simulation results.
        
        Args:
            results (Dict[str, Dict[str, Any]]): The simulation results.
            
        Returns:
            Dict[str, Any]: The comparison results.
        """
        # Extract key metrics
        comparison = {}
        
        for strategy_name, strategy_results in results.items():
            if not strategy_results or 'statistics' not in strategy_results:
                continue
            
            stats = strategy_results['statistics']
            
            # Extract key metrics
            comparison[strategy_name] = {
                'total_return': stats['total_return']['mean'],
                'max_drawdown': stats['max_drawdown']['mean'],
                'sharpe_ratio': stats['sharpe_ratio']['mean'],
                'sortino_ratio': stats['sortino_ratio']['mean'],
                'calmar_ratio': stats['calmar_ratio']['mean'],
                'win_rate': stats['win_rate']['mean'],
                'profit_factor': stats['profit_factor']['mean'],
                'profit_probability': strategy_results['profit_probability']
            }
        
        return comparison
    
    def _generate_comparison_plots(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate comparison plots.
        
        Args:
            results (Dict[str, Dict[str, Any]]): The simulation results.
        """
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'monte_carlo_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract key metrics
        metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'profit_factor']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            for strategy_name, strategy_results in results.items():
                if not strategy_results or 'statistics' not in strategy_results:
                    continue
                
                stats = strategy_results['statistics']
                
                if metric not in stats:
                    continue
                
                # Extract values
                mean = stats[metric]['mean']
                lower_ci = stats[metric]['lower_ci']
                upper_ci = stats[metric]['upper_ci']
                
                # Plot mean and confidence interval
                plt.errorbar(
                    x=[strategy_name],
                    y=[mean],
                    yerr=[[mean - lower_ci], [upper_ci - mean]],
                    fmt='o',
                    capsize=10,
                    label=strategy_name
                )
            
            plt.title(f'Comparison of {metric.replace("_", " ").title()}')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            
            # Add horizontal line at key values
            if metric == 'total_return':
                plt.axhline(y=0, color='r', linestyle='--')
            elif metric == 'sharpe_ratio':
                plt.axhline(y=1, color='r', linestyle='--')
            elif metric == 'profit_factor':
                plt.axhline(y=1, color='r', linestyle='--')
            
            plt.savefig(os.path.join(plots_dir, f'comparison_{metric}.png'))
            plt.close()
        
        # Plot profit probability
        plt.figure(figsize=(12, 6))
        
        for strategy_name, strategy_results in results.items():
            if not strategy_results or 'profit_probability' not in strategy_results:
                continue
            
            # Extract value
            profit_probability = strategy_results['profit_probability']
            
            # Plot bar
            plt.bar(strategy_name, profit_probability)
        
        plt.title('Comparison of Profit Probability')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'comparison_profit_probability.png'))
        plt.close()
