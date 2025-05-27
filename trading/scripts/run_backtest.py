"""
Script to run backtests on trading strategies
"""

import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.backtesting import BacktestEngine, BacktestConfig
from trading.strategies.ensemble_strategy import EnsembleStrategy
from trading.strategies.base_strategy import BaseStrategy
from core.models.ensemble_model import EnsembleModel
from utils.logger_config import setup_logging

# Setup logging
logger = setup_logging(
    app_name="BACKTEST",
    log_level="INFO",
    enable_file_logging=True
)


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple SMA crossover strategy for testing"""
    
    def __init__(self, short_period: int = 20, long_period: int = 50):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        self.last_signal = None
    
    def reset(self):
        """Reset strategy state"""
        self.last_signal = None
    
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """Generate trading signal based on SMA crossover"""
        if len(data) < self.long_period:
            return None
        
        # Calculate SMAs
        short_sma = data['close'].rolling(window=self.short_period).mean().iloc[-1]
        long_sma = data['close'].rolling(window=self.long_period).mean().iloc[-1]
        
        # Previous values
        prev_short_sma = data['close'].rolling(window=self.short_period).mean().iloc[-2]
        prev_long_sma = data['close'].rolling(window=self.long_period).mean().iloc[-2]
        
        # Check for crossover
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            # Bullish crossover
            signal = {
                'type': 'BUY',
                'confidence': 0.7,
                'reason': 'SMA bullish crossover',
                'short_sma': short_sma,
                'long_sma': long_sma
            }
            self.last_signal = 'BUY'
            return signal
        
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            # Bearish crossover
            if not self.config.get('allow_shorting', False):
                return None
            
            signal = {
                'type': 'SELL',
                'confidence': 0.7,
                'reason': 'SMA bearish crossover',
                'short_sma': short_sma,
                'long_sma': long_sma
            }
            self.last_signal = 'SELL'
            return signal
        
        return None


def load_historical_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load historical data for backtesting"""
    # Try to load from CSV first
    data_dir = Path("data/processed")
    csv_file = data_dir / f"{symbol}_{timeframe}_{days}d.csv"
    
    if csv_file.exists():
        logger.info(f"Loading data from {csv_file}")
        data = pd.read_csv(csv_file)
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    
    # Generate sample data for testing
    logger.warning("No historical data found. Generating sample data for testing.")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    
    # Generate price data with realistic patterns
    import numpy as np
    
    base_price = 50000
    volatility = 0.02
    trend = 0.0001
    
    prices = [base_price]
    for i in range(1, len(timestamps)):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(prices))),
        'high': prices * (1 + np.random.uniform(0, 0.005, len(prices))),
        'low': prices * (1 - np.random.uniform(0, 0.005, len(prices))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(prices))
    })
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Run backtests on trading strategies")
    parser.add_argument("--strategy", type=str, default="sma", 
                       choices=["sma", "ensemble"],
                       help="Strategy to backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="5min",
                       help="Timeframe for data")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days to backtest")
    parser.add_argument("--initial-balance", type=float, default=10000,
                       help="Initial balance for backtesting")
    parser.add_argument("--max-trades", type=int, default=3,
                       help="Maximum concurrent trades")
    parser.add_argument("--risk-per-trade", type=float, default=0.02,
                       help="Risk per trade (0.02 = 2%)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    
    args = parser.parse_args()
    
    # Load historical data
    logger.info(f"Loading {args.days} days of {args.timeframe} data for {args.symbol}")
    data = load_historical_data(args.symbol, args.timeframe, args.days)
    
    if data.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(data)} data points from {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Configure backtest
    config = BacktestConfig(
        initial_balance=args.initial_balance,
        max_open_trades=args.max_trades,
        risk_per_trade=args.risk_per_trade,
        maker_fee=0.001,
        taker_fee=0.001,
        slippage=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    # Initialize strategy
    if args.strategy == "sma":
        strategy = SimpleMovingAverageStrategy(short_period=20, long_period=50)
    elif args.strategy == "ensemble":
        # For ensemble, we need models - using mock for now
        logger.warning("Ensemble strategy requires trained models. Using simple strategy instead.")
        strategy = SimpleMovingAverageStrategy()
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(
        data=data,
        strategy=strategy,
        start_date=data['timestamp'].min(),
        end_date=data['timestamp'].max()
    )
    
    # Display results
    stats = results['statistics']
    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Winning Trades: {stats['winning_trades']}")
    logger.info(f"Losing Trades: {stats['losing_trades']}")
    logger.info(f"Win Rate: {stats['win_rate']:.2%}")
    logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    logger.info(f"Average Win: ${stats.get('average_win', 0):.2f}")
    logger.info(f"Average Loss: ${stats.get('average_loss', 0):.2f}")
    logger.info(f"Total Fees: ${stats.get('total_fees', 0):.2f}")
    logger.info(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"Total Return: ${stats['total_return']:.2f} ({stats['total_return_pct']:.2%})")
    logger.info("="*50)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_results(str(output_path))
        logger.info(f"Results saved to {output_path}")
    
    # Plot results if requested
    if args.plot:
        plot_path = output_path.with_suffix('.png') if args.output else None
        engine.plot_results(str(plot_path) if plot_path else None)


if __name__ == "__main__":
    main()
