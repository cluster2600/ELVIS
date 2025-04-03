"""
Command-line interface for strategy validation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trading.testing.strategy_validator import (
    StrategyValidator,
    MonteCarloConfig,
    WalkForwardConfig,
    StatisticalConfig
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Validate trading strategies using various methods.'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='trading/config/validation_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Path to strategy module'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to historical data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['monte_carlo', 'walk_forward', 'statistical', 'all'],
        default='all',
        help='Validation mode'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    return parser

def display_results(results: dict, console: Console):
    """Display validation results in a rich format."""
    # Monte Carlo Results
    if 'monte_carlo' in results:
        console.print(Panel.fit("Monte Carlo Simulation Results", style="bold blue"))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_column("Confidence Interval")
        
        for metric, value in results['monte_carlo'].items():
            if isinstance(value, dict) and 'confidence_intervals' in value:
                ci = value['confidence_intervals']
                table.add_row(
                    metric,
                    f"{value['mean']:.4f}",
                    f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"
                )
            else:
                table.add_row(metric, str(value), "-")
                
        console.print(table)
        
    # Walk-Forward Results
    if 'walk_forward' in results:
        console.print(Panel.fit("Walk-Forward Analysis Results", style="bold blue"))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Split")
        table.add_column("Train Sharpe")
        table.add_column("Test Sharpe")
        table.add_column("Parameters")
        
        for i, (train, test, params) in enumerate(zip(
            results['walk_forward']['train_metrics'],
            results['walk_forward']['test_metrics'],
            results['walk_forward']['optimized_params']
        )):
            table.add_row(
                str(i+1),
                f"{train['sharpe_ratio']:.4f}",
                f"{test['sharpe_ratio']:.4f}",
                str(params)
            )
            
        console.print(table)
        
    # Statistical Test Results
    if 'statistical' in results:
        console.print(Panel.fit("Statistical Test Results", style="bold blue"))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test")
        table.add_column("Value")
        table.add_column("Significant")
        
        for test, result in results['statistical'].items():
            if isinstance(result, dict):
                table.add_row(
                    test,
                    f"{result['value']:.4f}",
                    "Yes" if result['significant'] else "No"
                )
            else:
                table.add_row(test, f"{result:.4f}", "-")
                
        console.print(table)

def main():
    """Main function."""
    # Setup
    logger = setup_logging()
    console = Console()
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directories
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validator
        validator = StrategyValidator(
            mc_config=MonteCarloConfig(**config['monte_carlo']),
            wf_config=WalkForwardConfig(**config['walk_forward']),
            stats_config=StatisticalConfig(**config['statistical'])
        )
        
        # Load strategy and data
        strategy_path = Path(args.strategy)
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
            
        # Import strategy module
        sys.path.append(str(strategy_path.parent))
        strategy_module = __import__(strategy_path.stem)
        
        # Load data
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        data = pd.read_csv(data_path)
        
        results = {}
        
        # Run selected validation modes
        with Progress() as progress:
            if args.mode in ['monte_carlo', 'all']:
                task = progress.add_task("[cyan]Running Monte Carlo simulations...", total=1)
                results['monte_carlo'] = validator.run_monte_carlo_simulation(
                    strategy_module.strategy,
                    initial_capital=100000,
                    data=data,
                    params=config['walk_forward']['parameter_ranges']
                )
                progress.update(task, completed=1)
                
            if args.mode in ['walk_forward', 'all']:
                task = progress.add_task("[cyan]Running walk-forward analysis...", total=1)
                results['walk_forward'] = validator.run_walk_forward_analysis(
                    strategy_module.strategy,
                    data=data,
                    params=config['walk_forward']['parameter_ranges']
                )
                progress.update(task, completed=1)
                
            if args.mode in ['statistical', 'all']:
                task = progress.add_task("[cyan]Running statistical tests...", total=1)
                results['statistical'] = validator.run_statistical_tests(
                    strategy_module.strategy,
                    data=data,
                    params=config['walk_forward']['parameter_ranges']
                )
                progress.update(task, completed=1)
                
        # Display results
        display_results(results, console)
        
        # Save results
        validator.save_results(results, output_dir / 'validation_results.json')
        
        # Generate plots
        validator.plot_results(results, save_path=output_dir / 'plots')
        
        logger.info("Validation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 