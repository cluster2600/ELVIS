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
import time # Keep time for potential future use in bots
from utils import setup_logger, print_info, print_error
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
    # Updated strategy choices based on available implementations in trading/strategies/
    # Note: Sentiment and Grid strategies might need separate handling if they require different inputs/models
    parser.add_argument('--strategy', type=str, 
                        choices=['technical', 'mean_reversion', 'trend_following', 'ema_rsi', 'sentiment', 'grid'], 
                        default='technical', help='Trading strategy (default: technical)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=LOGGING_CONFIG['LOG_LEVEL'],
                        help=f'Logging level (default: {LOGGING_CONFIG["LOG_LEVEL"]})')
    return parser.parse_args()

def get_strategy(strategy_name, logger):
    """Imports and returns the specified strategy class."""
    # Import strategies locally to avoid circular dependencies if strategies import config
    from trading.strategies import (
        TechnicalStrategy, MeanReversionStrategy, TrendFollowingStrategy, EmaRsiStrategy,
        SentimentStrategy, GridStrategy # Assuming these are implemented
    )
    strategies = {
        'technical': TechnicalStrategy,
        'mean_reversion': MeanReversionStrategy,
        'trend_following': TrendFollowingStrategy,
        'ema_rsi': EmaRsiStrategy,
        'sentiment': SentimentStrategy,
        'grid': GridStrategy
    }
    if strategy_name not in strategies:
        # Log available strategies for clarity
        available = ", ".join(strategies.keys())
        logger.error(f"Invalid strategy: {strategy_name}. Available: {available}")
        raise ValueError(f"Invalid strategy: {strategy_name}")
    logger.info(f"Selected strategy: {strategy_name}")
    return strategies[strategy_name](logger) # Instantiate the strategy

def initialize_bot(args, logger):
    """Initializes the appropriate bot based on the mode."""
    strategy_instance = get_strategy(args.strategy, logger)
    
    if args.mode == 'live':
        from trading.live_bot import LiveBot
        logger.info("Initializing LiveBot...")
        return LiveBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger)
    elif args.mode == 'backtest':
        from trading.backtest_bot import BacktestBot
        logger.info("Initializing BacktestBot...")
        return BacktestBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger)
    elif args.mode == 'paper':
        # PaperBot will now handle its own dashboard integration
        from trading.paper_bot import PaperBot
        logger.info("Initializing PaperBot...")
        return PaperBot(args.symbol, args.timeframe, args.leverage, strategy=strategy_instance, logger=logger)
    else:
        # This case should not be reachable due to argparse choices, but included for safety
        logger.error(f"Invalid mode specified: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")

def main():
    """Main execution function."""
    print(ELVIS_ASCII)
    args = parse_arguments()
    
    # Setup logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG.get('LOG_TO_FILE', True), log_level=log_level)
    
    logger.info("Starting ELVIS...")
    logger.info(f"Arguments: Mode={args.mode}, Symbol={args.symbol}, Timeframe={args.timeframe}, Strategy={args.strategy}, Leverage={args.leverage}")

    # Check API keys only if running in live mode
    if args.mode == 'live':
        if not all([API_CONFIG.get('BINANCE_API_KEY'), API_CONFIG.get('BINANCE_API_SECRET')]):
            print_error(logger, "Missing Binance API keys for live mode. Please set BINANCE_API_KEY and BINANCE_API_SECRET in the .env file or environment variables.")
            sys.exit(1)
        
        # Check production mode flag
        if not TRADING_CONFIG.get('PRODUCTION_MODE', False):
            print_error(logger, "PRODUCTION_MODE is disabled in config.py. Cannot run in live mode for safety. Set PRODUCTION_MODE = True to enable live trading.")
            sys.exit(1)
        else:
             print_info(logger, "PRODUCTION_MODE enabled. Running in live trading mode.")

    elif args.mode == 'paper':
         print_info(logger, "Running in paper trading mode.")
         # Note: PaperBot will handle dashboard internally if needed.

    elif args.mode == 'backtest':
         print_info(logger, "Running in backtesting mode.")

    try:
        # Initialize and run the bot for the selected mode
        bot = initialize_bot(args, logger)
        print_info(logger, f"Bot initialized successfully for {args.mode} mode.")
        bot.run() # Each bot class should implement its own run loop
        logger.info(f"ELVIS {args.mode} run completed.")
        
    except ValueError as ve:
        # Handle known errors like invalid strategy or mode
        print_error(logger, f"Configuration error: {ve}")
        sys.exit(1)
    except ImportError as ie:
         print_error(logger, f"Import error: {ie}. Ensure all dependencies are installed and modules exist.")
         sys.exit(1)
    except Exception as e:
        # Catch unexpected errors during initialization or run
        print_error(logger, f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main()
