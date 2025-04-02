#!/usr/bin/env python3
"""
Main entry point for the ELVIS project with real-time candlestick dashboard.
"""

ELVIS_ASCII = r"""
 _______  _        __      __  _____   _____ 
|  ____| | |       \ \    / / |_   _| / ____|
| |__    | |        \ \  / /    | |  | (___  
|  __|   | |         \ \/ /     | |   \___ \ 
| |____  | |____      \  /     _| |_  ____) |
|______| |______|      \/     |_____||_____/ 
"""

import argparse
import logging
import sys
import time
import random
import psutil
from datetime import datetime
from utils import setup_logger, print_info, print_error
from utils.console_dashboard import ConsoleDashboardManager
from utils.price_fetcher import PriceFetcher
from config import API_CONFIG, TRADING_CONFIG, LOGGING_CONFIG

def parse_arguments():
    parser = argparse.ArgumentParser(description='ELVIS - Enhanced Leveraged Virtual Investment System')
    parser.add_argument('--mode', type=str, choices=['live', 'backtest', 'paper'], default=TRADING_CONFIG['DEFAULT_MODE'],
                        help=f'Trading mode (default: {TRADING_CONFIG["DEFAULT_MODE"]})')
    parser.add_argument('--symbol', type=str, default=TRADING_CONFIG['SYMBOL'],
                        help=f'Trading symbol (default: {TRADING_CONFIG["SYMBOL"]})')
    parser.add_argument('--timeframe', type=str, default=TRADING_CONFIG['TIMEFRAME'],
                        help=f'Trading timeframe (default: {TRADING_CONFIG["TIMEFRAME"]})')
    parser.add_argument('--leverage', type=int, default=TRADING_CONFIG['LEVERAGE_MIN'],
                        help=f'Initial leverage (default: {TRADING_CONFIG["LEVERAGE_MIN"]})')
    parser.add_argument('--strategy', type=str, choices=['technical', 'mean_reversion', 'trend_following', 'ema_rsi'], 
                        default='technical', help='Trading strategy (default: technical)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=LOGGING_CONFIG['LOG_LEVEL'],
                        help=f'Logging level (default: {LOGGING_CONFIG["LOG_LEVEL"]})')
    return parser.parse_args()

def get_system_stats():
    process = psutil.Process()
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': process.memory_percent(),
        'uptime': int(time.time() - process.create_time()),
        'api_calls': random.randint(100, 500),
        'last_error': None
    }

def get_strategy(strategy_name, logger):
    from trading.strategies import (
        TechnicalStrategy, MeanReversionStrategy, TrendFollowingStrategy, EmaRsiStrategy
    )
    strategies = {
        'technical': TechnicalStrategy,
        'mean_reversion': MeanReversionStrategy,
        'trend_following': TrendFollowingStrategy,
        'ema_rsi': EmaRsiStrategy
    }
    if strategy_name not in strategies:
        raise ValueError(f"Invalid strategy: {strategy_name}")
    return strategies[strategy_name](logger)

def initialize_bot(args, logger):
    strategy = get_strategy(args.strategy, logger)
    if args.mode == 'live':
        from trading.live_bot import LiveBot
        return LiveBot(args.symbol, args.timeframe, args.leverage, strategy=strategy, logger=logger)
    elif args.mode == 'backtest':
        from trading.backtest_bot import BacktestBot
        return BacktestBot(args.symbol, args.timeframe, args.leverage, strategy=strategy, logger=logger)
    elif args.mode == 'paper':
        from trading.paper_bot import PaperBot
        return PaperBot(args.symbol, args.timeframe, args.leverage, strategy=strategy, logger=logger)
    raise ValueError(f"Invalid mode: {args.mode}")

def run_dashboard_simulation(logger, args):
    dashboard_manager = ConsoleDashboardManager(logger)
    price_fetcher = PriceFetcher(logger, symbol=args.symbol, timeframe="1m")
    
    dashboard_manager.start_dashboard()
    price_fetcher.start()
    time.sleep(5)  # Increased delay to ensure WebSocket data starts flowing
    
    portfolio_value = 10000.0
    position_size = 0.0
    entry_price = 0.0
    
    dashboard_manager.set_testnet(True)
    dashboard_manager.update_portfolio_value(portfolio_value)
    
    signals = {
        'EMA_RSI_Strategy': {'buy': False, 'sell': False},
        'Technical_Strategy': {'buy': False, 'sell': False},
        'Mean_Reversion': {'buy': False, 'sell': False},
        'Trend_Following': {'buy': False, 'sell': False}
    }
    metrics = {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0,
        'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'avg_profit': 0.0,
        'avg_loss': 0.0, 'daily_return': 0.0, 'monthly_return': 0.0, 'yearly_return': 0.0,
        'volatility': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0
    }
    
    try:
        iteration = 0
        while True:
            iteration += 1
            current_candle = price_fetcher.get_current_candle()
            current_price = current_candle['close']
            logger.debug(f"Current candle in main: {current_candle}")
            dashboard_manager.update_candle(current_candle)
            
            if current_price <= 0:
                logger.warning("No valid price data received yet.")
                time.sleep(1)
                continue
            
            if position_size == 0 and random.random() > 0.8:
                position_size = portfolio_value / current_price * 0.1
                entry_price = current_price
                portfolio_value -= position_size * entry_price
                for strategy in signals:
                    signals[strategy] = {'buy': False, 'sell': False}
                chosen_strategy = random.choice(list(signals.keys()))
                signals[chosen_strategy]['buy'] = True
                dashboard_manager.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': args.symbol,
                    'side': 'buy',
                    'price': entry_price,
                    'quantity': position_size,
                    'pnl': 0.0
                })
                logger.info(f"BUY: {position_size:.8f} BTC at ${entry_price:.2f}")
            elif position_size > 0 and random.random() > 0.8:
                pnl = (current_price - entry_price) * position_size
                portfolio_value += position_size * current_price
                for strategy in signals:
                    signals[strategy] = {'buy': False, 'sell': False}
                chosen_strategy = random.choice(list(signals.keys()))
                signals[chosen_strategy]['sell'] = True
                dashboard_manager.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': args.symbol,
                    'side': 'sell',
                    'price': current_price,
                    'quantity': position_size,
                    'pnl': pnl
                })
                logger.info(f"SELL: {position_size:.8f} BTC at ${current_price:.2f} (PnL: ${pnl:.2f})")
                position_size = 0.0
                entry_price = 0.0
            
            dashboard_manager.update_portfolio_value(portfolio_value)
            dashboard_manager.update_position(position_size, entry_price, current_price)
            dashboard_manager.update_strategy_signals(signals)
            dashboard_manager.update_system_stats(get_system_stats())
            
            metrics['total_trades'] = iteration // 2
            metrics['winning_trades'] = int(metrics['total_trades'] * random.uniform(0.5, 0.7))
            metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades']
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] * 100
                metrics['profit_factor'] = random.uniform(1.2, 2.8)
                metrics['sharpe_ratio'] = random.uniform(0.8, 2.5)
                metrics['max_drawdown'] = random.uniform(3.0, 12.0)
            dashboard_manager.update_metrics(metrics)
            
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        price_fetcher.stop()
        dashboard_manager.stop_dashboard()

def main():
    print(ELVIS_ASCII)
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG['LOG_TO_FILE'], log_level=log_level)
    
    if not all([API_CONFIG['BINANCE_API_KEY'], API_CONFIG['BINANCE_API_SECRET']]) and args.mode == 'live':
        print_error(logger, "Missing API keys for live mode. Please set them in the .env file.")
        sys.exit(1)
    
    if args.mode == 'live' and not TRADING_CONFIG['PRODUCTION_MODE']:
        print_error(logger, "PRODUCTION_MODE is disabled. Cannot run in live mode.")
        sys.exit(1)
    
    try:
        if args.mode == 'paper':
            print_info(logger, "Running dashboard simulation in paper mode with real-time candlestick data")
            run_dashboard_simulation(logger, args)
        else:
            bot = initialize_bot(args, logger)
            print_info(logger, f"Bot initialized in {args.mode} mode for {args.symbol} on {args.timeframe} with {args.strategy}")
            bot.run()
    except Exception as e:
        print_error(logger, f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()