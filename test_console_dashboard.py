"""
Test script for the enhanced console dashboard.
This script demonstrates the advanced visualization features of the ELVIS console dashboard.
"""

import logging
import time
import random
import os
import psutil
import platform
from datetime import datetime, timedelta
from utils.console_dashboard import ConsoleDashboardManager
from utils.price_fetcher import PriceFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ELVIS.TestDashboard")

def get_system_stats():
    """
    Get system statistics.
    
    Returns:
        dict: System statistics.
    """
    process = psutil.Process(os.getpid())
    
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': process.memory_percent(),
        'uptime': int(time.time() - process.create_time()),
        'api_calls': random.randint(100, 500),
        'last_error': None
    }

def get_market_stats(current_price):
    """
    Get market statistics.
    
    Args:
        current_price (float): Current price.
        
    Returns:
        dict: Market statistics.
    """
    # Generate realistic market stats based on current price
    daily_high = current_price * (1 + random.uniform(0.01, 0.05))
    daily_low = current_price * (1 - random.uniform(0.01, 0.05))
    
    # Determine market sentiment
    sentiment_options = ['Bearish', 'Neutral', 'Bullish']
    sentiment_weights = [0.2, 0.3, 0.5]  # Slightly bullish bias
    market_sentiment = random.choices(sentiment_options, weights=sentiment_weights)[0]
    
    # Determine trend
    trend_options = ['Downtrend', 'Sideways', 'Uptrend']
    trend_weights = [0.2, 0.3, 0.5]  # Slightly uptrend bias
    trend = random.choices(trend_options, weights=trend_weights)[0]
    
    return {
        'daily_high': daily_high,
        'daily_low': daily_low,
        'daily_volume': random.uniform(1000, 5000),
        'market_sentiment': market_sentiment,
        'volatility': random.uniform(1.0, 5.0),
        'trend': trend
    }

def main():
    """
    Main function to test the enhanced console dashboard.
    """
    logger.info("Starting enhanced console dashboard test")
    logger.info("Press '1', '2', or '3' to switch between view modes")
    logger.info("Press 'q' to quit")
    
    # Create dashboard manager
    dashboard_manager = ConsoleDashboardManager(logger)
    
    # Start dashboard
    dashboard_manager.start_dashboard()
    
    try:
        # Initialize price fetcher
        price_fetcher = PriceFetcher(logger, update_interval=2)
        price_fetcher.start()
        
        # Wait for initial price fetch
        time.sleep(3)
        
        # Get initial price
        current_price = price_fetcher.get_current_price()
        if current_price == 0:
            current_price = 75655.0  # Fallback price if API fetch fails
        
        # Initial portfolio value
        portfolio_value = 10000.0
        
        # Initial position
        position_size = 0.0
        entry_price = 0.0
        
        # Set testnet status (default is True, but we'll set it explicitly)
        dashboard_manager.set_testnet(True)  # Set to False for production mode
        
        # Update dashboard with initial values
        dashboard_manager.update_portfolio_value(portfolio_value)
        dashboard_manager.update_position(position_size, entry_price, current_price)
        
        # Initialize price history from fetcher or with random values if empty
        price_history = price_fetcher.get_price_history()
        if price_history:
            for price in price_history:
                dashboard_manager.dashboard.update_price_history(price)
        else:
            for _ in range(50):
                dashboard_manager.dashboard.update_price_history(
                    current_price + random.uniform(-500, 500)
                )
        
        # Update metrics
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'daily_return': 0.0,
            'monthly_return': 0.0,
            'yearly_return': 0.0,
            'volatility': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
        dashboard_manager.update_metrics(metrics)
        
        # Update strategy signals
        signals = {
            'EMA_RSI_Strategy': {
                'buy': False,
                'sell': False
            },
            'Technical_Strategy': {
                'buy': False,
                'sell': False
            },
            'Mean_Reversion': {
                'buy': False,
                'sell': False
            },
            'Trend_Following': {
                'buy': False,
                'sell': False
            },
            'Sentiment_Strategy': {
                'buy': False,
                'sell': False
            }
        }
        dashboard_manager.update_strategy_signals(signals)
        
        # Update market stats
        market_stats = get_market_stats(current_price)
        dashboard_manager.dashboard.update_market_stats(market_stats)
        
        # Update system stats
        system_stats = get_system_stats()
        dashboard_manager.dashboard.update_system_stats(system_stats)
        
        # Simulate trading with continuous updates
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Iteration {iteration}")
            
            # Get real-time price from fetcher
            fetched_price = price_fetcher.get_current_price()
            if fetched_price > 0:
                current_price = fetched_price
            else:
                # Fallback to random movement if API fetch fails
                price_change = current_price * random.uniform(-0.002, 0.002)  # 0.2% max change
                current_price += price_change
            
            # Update price history and dashboard with current price
            dashboard_manager.dashboard.update_price_history(current_price)
            
            # Simulate trading decisions
            if position_size == 0 and random.random() > 0.8:
                # Buy
                position_size = portfolio_value / current_price * 0.1
                entry_price = current_price
                portfolio_value -= position_size * entry_price
                
                # Update signals - randomly activate one strategy
                for strategy in signals:
                    signals[strategy]['buy'] = False
                    signals[strategy]['sell'] = False
                
                # Choose a random strategy to generate the buy signal
                chosen_strategy = random.choice(list(signals.keys()))
                signals[chosen_strategy]['buy'] = True
                
                # Add trade
                dashboard_manager.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'price': entry_price,
                    'quantity': position_size,
                    'pnl': 0.0
                })
                
                logger.info(f"BUY: {position_size:.8f} BTC at ${entry_price:.2f}")
            elif position_size > 0 and random.random() > 0.8:
                # Sell
                pnl = (current_price - entry_price) * position_size
                portfolio_value += position_size * current_price
                
                # Update signals - randomly activate one strategy
                for strategy in signals:
                    signals[strategy]['buy'] = False
                    signals[strategy]['sell'] = False
                
                # Choose a random strategy to generate the sell signal
                chosen_strategy = random.choice(list(signals.keys()))
                signals[chosen_strategy]['sell'] = True
                
                # Add trade
                dashboard_manager.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'price': current_price,
                    'quantity': position_size,
                    'pnl': pnl
                })
                
                logger.info(f"SELL: {position_size:.8f} BTC at ${current_price:.2f} (PnL: ${pnl:.2f})")
                
                # Reset position
                position_size = 0.0
                entry_price = 0.0
            else:
                # Hold - occasionally show some signals even when not trading
                if random.random() > 0.9:
                    for strategy in signals:
                        signals[strategy]['buy'] = False
                        signals[strategy]['sell'] = False
                    
                    # Randomly show some signals
                    for _ in range(random.randint(0, 2)):
                        chosen_strategy = random.choice(list(signals.keys()))
                        signal_type = random.choice(['buy', 'sell'])
                        signals[chosen_strategy][signal_type] = True
            
            # Update dashboard
            dashboard_manager.update_portfolio_value(portfolio_value)
            dashboard_manager.update_position(position_size, entry_price, current_price)
            dashboard_manager.update_strategy_signals(signals)
            
            # Update market stats
            market_stats = get_market_stats(current_price)
            dashboard_manager.dashboard.update_market_stats(market_stats)
            
            # Update system stats
            system_stats = get_system_stats()
            dashboard_manager.dashboard.update_system_stats(system_stats)
            
            # Update metrics with more realistic values
            metrics['total_trades'] = iteration // 2  # Not every iteration results in a trade
            metrics['winning_trades'] = int(metrics['total_trades'] * random.uniform(0.5, 0.7))  # 50-70% win rate
            metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades']
            
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] * 100
                metrics['profit_factor'] = random.uniform(1.2, 2.8)
                metrics['sharpe_ratio'] = random.uniform(0.8, 2.5)
                metrics['sortino_ratio'] = random.uniform(1.0, 3.0)
                metrics['calmar_ratio'] = random.uniform(0.5, 2.0)
                metrics['max_drawdown'] = random.uniform(3.0, 12.0)
                metrics['avg_profit'] = random.uniform(80.0, 250.0)
                metrics['avg_loss'] = random.uniform(30.0, 120.0)
                metrics['daily_return'] = random.uniform(-1.5, 2.5)
                metrics['monthly_return'] = random.uniform(-5.0, 15.0)
                metrics['yearly_return'] = random.uniform(10.0, 80.0)
                metrics['volatility'] = random.uniform(10.0, 30.0)
            
            dashboard_manager.update_metrics(metrics)
            
            # Sleep - shorter interval for more responsive dashboard
            time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop price fetcher
        price_fetcher.stop()
        
        # Stop dashboard
        dashboard_manager.stop_dashboard()
        logger.info("Enhanced console dashboard test stopped")

if __name__ == "__main__":
    main()
