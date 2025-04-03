#!/usr/bin/env python3
"""
Main entry point for the ELVIS project.
Initializes and runs the appropriate trading bot based on command-line arguments.
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
from utils import setup_logger, print_info, print_error
from utils.console_dashboard import ConsoleDashboardManager
from config import API_CONFIG, TRADING_CONFIG, LOGGING_CONFIG

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='ELVIS - Enhanced Leveraged Virtual Investment System')
    parser.add_argument('--mode', type=str, choices=['live', 'backtest', 'paper'], default=TRADING_CONFIG['DEFAULT_MODE'],
                        help=f'Trading mode (default: {TRADING_CONFIG["DEFAULT_MODE"]})')
    parser.add_argument('--symbol', type=str, default=TRADING_CONFIG['SYMBOL'],
                        help=f'Trading symbol (default: {TRADING_CONFIG["SYMBOL"]})')
    parser.add_argument('--timeframe', type=str, default=TRADING_CONFIG['TIMEFRAME'],
                        help=f'Trading timeframe (default: {TRADING_CONFIG["TIMEFRAME"]})')
    parser.add_argument('--leverage', type=int, default=TRADING_CONFIG['LEVERAGE_MIN'],
                        help=f'Initial leverage (default: {TRADING_CONFIG["LEVERAGE_MIN"]})')
    parser.add_argument('--strategy', type=str,
                        choices=['technical', 'mean_reversion', 'trend_following', 'ema_rsi', 'ensemble'],
                        default='technical', help='Trading strategy (default: technical)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=LOGGING_CONFIG.get('LOG_LEVEL', 'INFO'),
                        help=f'Logging level (default: {LOGGING_CONFIG["LOG_LEVEL"]})')
    parser.add_argument('--dashboard', type=str, choices=['console', 'none'], default='none',
                        help='Dashboard type (default: none)')
    return parser.parse_args()

def get_strategy(strategy_name, logger):
    """Imports and returns the specified strategy class."""
    from trading.strategies import (
        TechnicalStrategy, MeanReversionStrategy, TrendFollowingStrategy, EmaRsiStrategy, EnsembleStrategy
    )
    strategies = {
        'technical': TechnicalStrategy,
        'mean_reversion': MeanReversionStrategy,
        'trend_following': TrendFollowingStrategy,
        'ema_rsi': EmaRsiStrategy,
        'ensemble': EnsembleStrategy
    }
    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        logger.error(f"Invalid strategy: {strategy_name}. Available: {available}")
        raise ValueError(f"Invalid strategy: {strategy_name}")
    logger.info(f"Selected strategy: {strategy_name}")
    return strategies[strategy_name](logger)

def initialize_bot(args, logger, dashboard_manager=None):
    """Initializes the appropriate bot based on the mode."""
    strategy_instance = get_strategy(args.strategy, logger)
    
    if args.mode == 'live':
        from trading.live_bot import LiveBot
        logger.info("Initializing LiveBot...")
        return LiveBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger, dashboard_manager=dashboard_manager)
    elif args.mode == 'backtest':
        from trading.backtest_bot import BacktestBot
        logger.info("Initializing BacktestBot...")
        return BacktestBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger, dashboard_manager=dashboard_manager)
    elif args.mode == 'paper':
        from trading.paper_bot import PaperBot
        logger.info("Initializing PaperBot...")
        return PaperBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger, dashboard_manager=dashboard_manager)
    else:
        logger.error(f"Invalid mode specified: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")

def main():
    """Main execution function."""
    print(ELVIS_ASCII)
    args = parse_arguments()
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG.get('LOG_TO_FILE', True), log_level=log_level)
    
    logger.info("Starting ELVIS...")
    logger.info(f"Arguments: Mode={args.mode}, Symbol={args.symbol}, Timeframe={args.timeframe}, Strategy={args.strategy}, Leverage={args.leverage}, Dashboard={args.dashboard}")

    if args.mode == 'live':
        if not all([API_CONFIG.get('BINANCE_API_KEY'), API_CONFIG.get('BINANCE_API_SECRET')]):
            print_error(logger, "Missing Binance API keys for live mode. Please set BINANCE_API_KEY and BINANCE_API_SECRET in the .env file or environment variables.")
            sys.exit(1)
        if not TRADING_CONFIG.get('PRODUCTION_MODE', False):
            print_error(logger, "PRODUCTION_MODE is disabled in config.py. Cannot run in live mode for safety. Set PRODUCTION_MODE = True to enable live trading.")
            sys.exit(1)
        else:
            print_info(logger, "PRODUCTION_MODE enabled. Running in live trading mode.")
    elif args.mode == 'paper':
        print_info(logger, "Running in paper trading mode.")
    elif args.mode == 'backtest':
        print_info(logger, "Running in backtesting mode.")

    try:
        # Initialize dashboard if requested
        dashboard_manager = None
        if args.dashboard == 'console':
            dashboard_manager = ConsoleDashboardManager(logger)
            print_info(logger, "Console dashboard initialized.")

        bot = initialize_bot(args, logger, dashboard_manager)
        print_info(logger, f"Bot initialized successfully for {args.mode} mode.")
        
        # Run the bot
        bot.run()
        
        logger.info(f"ELVIS {args.mode} run completed.")
        
        # Stop dashboard if it was started
        if dashboard_manager:
            dashboard_manager.stop_dashboard()
            
    except ValueError as ve:
        print_error(logger, f"Configuration error: {ve}")
        sys.exit(1)
    except ImportError as ie:
        print_error(logger, f"Import error: {ie}. Ensure all dependencies are installed and modules exist.")
        sys.exit(1)
    except Exception as e:
        print_error(logger, f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()