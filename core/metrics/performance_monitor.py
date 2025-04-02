"""
Performance monitoring module for the ELVIS project.
This module provides functionality for monitoring and reporting trading performance.
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

from config import FILE_PATHS

class PerformanceMonitor:
    """
    Performance monitor for trading.
    Tracks and reports trading performance metrics.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the performance monitor.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logger
        
        # Performance parameters
        self.metrics_file = kwargs.get('metrics_file', os.path.join(FILE_PATHS['METRICS_FILE']))
        self.plot_dir = kwargs.get('plot_dir', os.path.join(FILE_PATHS['PLOT_DIR']))
        
        # Initialize metrics
        self.trades = []
        self.daily_returns = {}
        self.metrics = {}
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the performance monitor.
        
        Args:
            trade (Dict[str, Any]): The trade to add.
        """
        try:
            self.logger.info(f"Adding trade: {trade}")
            
            # Add timestamp if not present
            if 'timestamp' not in trade:
                trade['timestamp'] = datetime.now().isoformat()
            
            # Add trade to list
            self.trades.append(trade)
            
            # Update daily returns
            date = datetime.fromisoformat(trade['timestamp']).date().isoformat()
            if date not in self.daily_returns:
                self.daily_returns[date] = 0.0
            
            # Add PnL to daily returns
            if 'pnl' in trade:
                self.daily_returns[date] += trade['pnl']
            
            # Save trades
            self._save_trades()
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")
    
    def _save_trades(self) -> None:
        """
        Save trades to file.
        """
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Save to CSV
            trades_file = os.path.join(os.path.dirname(self.metrics_file), 'trades.csv')
            trades_df.to_csv(trades_file, index=False)
            
            self.logger.info(f"Saved {len(self.trades)} trades to {trades_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving trades: {e}")
    
    def load_trades(self) -> None:
        """
        Load trades from file.
        """
        try:
            # Load from CSV
            trades_file = os.path.join(os.path.dirname(self.metrics_file), 'trades.csv')
            
            if os.path.exists(trades_file):
                trades_df = pd.read_csv(trades_file)
                
                # Convert to list of dictionaries
                self.trades = trades_df.to_dict('records')
                
                # Rebuild daily returns
                self.daily_returns = {}
                for trade in self.trades:
                    if 'timestamp' in trade and 'pnl' in trade:
                        date = datetime.fromisoformat(trade['timestamp']).date().isoformat()
                        if date not in self.daily_returns:
                            self.daily_returns[date] = 0.0
                        self.daily_returns[date] += trade['pnl']
                
                self.logger.info(f"Loaded {len(self.trades)} trades from {trades_file}")
            else:
                self.logger.warning(f"Trades file not found: {trades_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading trades: {e}")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dict[str, float]: The performance metrics.
        """
        try:
            self.logger.info("Calculating performance metrics")
            
            # Check if we have trades
            if not self.trades:
                self.logger.warning("No trades to calculate metrics")
                # Return a dictionary with total_trades = 0
                metrics = {'total_trades': 0}
                self.metrics = metrics
                self._save_metrics()
                return metrics
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Calculate metrics
            metrics = {}
            
            # Total trades
            metrics['total_trades'] = len(trades_df)
            
            # Winning trades
            if 'pnl' in trades_df.columns:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(trades_df) - len(winning_trades)
                
                # Win rate
                metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0
                
                # Total PnL
                metrics['total_pnl'] = trades_df['pnl'].sum()
                
                # Average PnL
                metrics['avg_pnl'] = trades_df['pnl'].mean()
                
                # Average winning trade
                metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
                
                # Average losing trade
                losing_trades = trades_df[trades_df['pnl'] <= 0]
                metrics['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
                
                # Profit factor
                total_wins = winning_trades['pnl'].sum()
                total_losses = abs(losing_trades['pnl'].sum())
                metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Sharpe ratio
                if len(self.daily_returns) > 0:
                    daily_returns = pd.Series(self.daily_returns.values())
                    metrics['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0.0
                
                # Maximum drawdown
                if 'cumulative_pnl' in trades_df.columns:
                    cumulative_pnl = trades_df['cumulative_pnl'].values
                    max_drawdown = 0.0
                    peak = cumulative_pnl[0]
                    
                    for value in cumulative_pnl:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak if peak > 0 else 0.0
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    metrics['max_drawdown'] = max_drawdown
            
            # Store metrics
            self.metrics = metrics
            
            # Save metrics
            self._save_metrics()
            
            self.logger.info(f"Calculated metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _save_metrics(self) -> None:
        """
        Save metrics to file.
        """
        try:
            # Save to JSON
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"Saved metrics to {self.metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def load_metrics(self) -> Dict[str, float]:
        """
        Load metrics from file.
        
        Returns:
            Dict[str, float]: The performance metrics.
        """
        try:
            # Load from JSON
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                
                self.logger.info(f"Loaded metrics from {self.metrics_file}")
            else:
                self.logger.warning(f"Metrics file not found: {self.metrics_file}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
            return {}
    
    def plot_equity_curve(self) -> str:
        """
        Plot the equity curve.
        
        Returns:
            str: The path to the saved plot.
        """
        try:
            self.logger.info("Plotting equity curve")
            
            # Check if we have trades
            if not self.trades:
                self.logger.warning("No trades to plot equity curve")
                return ""
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Check if we have PnL
            if 'pnl' not in trades_df.columns:
                self.logger.warning("No PnL data to plot equity curve")
                return ""
            
            # Calculate cumulative PnL
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df.index, trades_df['cumulative_pnl'])
            plt.title('Equity Curve')
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative PnL')
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(self.plot_dir, 'equity_curve.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved equity curve to {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
            return ""
    
    def plot_daily_returns(self) -> str:
        """
        Plot the daily returns.
        
        Returns:
            str: The path to the saved plot.
        """
        try:
            self.logger.info("Plotting daily returns")
            
            # Check if we have daily returns
            if not self.daily_returns:
                self.logger.warning("No daily returns to plot")
                return ""
            
            # Convert to DataFrame
            daily_returns_df = pd.DataFrame(
                list(self.daily_returns.items()),
                columns=['date', 'return']
            )
            daily_returns_df['date'] = pd.to_datetime(daily_returns_df['date'])
            daily_returns_df = daily_returns_df.sort_values('date')
            
            # Plot daily returns
            plt.figure(figsize=(12, 6))
            plt.bar(daily_returns_df['date'], daily_returns_df['return'])
            plt.title('Daily Returns')
            plt.xlabel('Date')
            plt.ylabel('Return')
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(self.plot_dir, 'daily_returns.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved daily returns to {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error plotting daily returns: {e}")
            return ""
    
    def plot_win_loss_distribution(self) -> str:
        """
        Plot the win/loss distribution.
        
        Returns:
            str: The path to the saved plot.
        """
        try:
            self.logger.info("Plotting win/loss distribution")
            
            # Check if we have trades
            if not self.trades:
                self.logger.warning("No trades to plot win/loss distribution")
                return ""
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Check if we have PnL
            if 'pnl' not in trades_df.columns:
                self.logger.warning("No PnL data to plot win/loss distribution")
                return ""
            
            # Plot win/loss distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(trades_df['pnl'], bins=20, kde=True)
            plt.title('Win/Loss Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(self.plot_dir, 'win_loss_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved win/loss distribution to {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error plotting win/loss distribution: {e}")
            return ""
    
    def generate_report(self) -> str:
        """
        Generate a performance report.
        
        Returns:
            str: The path to the saved report.
        """
        try:
            self.logger.info("Generating performance report")
            
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            # Generate plots
            equity_curve_path = self.plot_equity_curve()
            daily_returns_path = self.plot_daily_returns()
            win_loss_path = self.plot_win_loss_distribution()
            
            # Generate HTML report
            report = f"""
            <html>
            <head>
                <title>ELVIS Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    .plot {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>ELVIS Performance Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            # Add metrics to report
            for metric, value in metrics.items():
                # Format value based on type
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                report += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{formatted_value}</td>
                    </tr>
                """
            
            report += """
                </table>
                
                <h2>Equity Curve</h2>
            """
            
            # Add equity curve to report
            if equity_curve_path:
                report += f"""
                <div class="plot">
                    <img src="{os.path.basename(equity_curve_path)}" alt="Equity Curve" width="800">
                </div>
                """
            
            report += """
                <h2>Daily Returns</h2>
            """
            
            # Add daily returns to report
            if daily_returns_path:
                report += f"""
                <div class="plot">
                    <img src="{os.path.basename(daily_returns_path)}" alt="Daily Returns" width="800">
                </div>
                """
            
            report += """
                <h2>Win/Loss Distribution</h2>
            """
            
            # Add win/loss distribution to report
            if win_loss_path:
                report += f"""
                <div class="plot">
                    <img src="{os.path.basename(win_loss_path)}" alt="Win/Loss Distribution" width="800">
                </div>
                """
            
            report += """
            </body>
            </html>
            """
            
            # Save report
            report_path = os.path.join(self.plot_dir, 'performance_report.html')
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Saved performance report to {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return ""
