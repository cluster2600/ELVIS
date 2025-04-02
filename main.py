#!/usr/bin/env python3
"""
Main entry point for the ELVIS project.
This script initializes and runs the trading bot.

ELVIS: Enhanced Leveraged Virtual Investment System
"""

# ASCII Art for ELVIS
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
from utils import setup_logger, print_info, print_error
from config import API_CONFIG, TRADING_CONFIG, LOGGING_CONFIG

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='ELVIS - Enhanced Leveraged Virtual Investment System')
    parser.add_argument('--mode', type=str, choices=['live', 'backtest', 'paper'], default=TRADING_CONFIG['DEFAULT_MODE'],
                        help=f'Trading mode: live, backtest, or paper trading (default: {TRADING_CONFIG["DEFAULT_MODE"]})')
    parser.add_argument('--symbol', type=str, default=TRADING_CONFIG['SYMBOL'],
                        help=f'Trading symbol (default: {TRADING_CONFIG["SYMBOL"]})')
    parser.add_argument('--timeframe', type=str, default=TRADING_CONFIG['TIMEFRAME'],
                        help=f'Trading timeframe (default: {TRADING_CONFIG["TIMEFRAME"]})')
    parser.add_argument('--leverage', type=int, default=TRADING_CONFIG['LEVERAGE_MIN'],
                        help=f'Initial leverage (default: {TRADING_CONFIG["LEVERAGE_MIN"]})')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=LOGGING_CONFIG['LOG_LEVEL'],
                        help=f'Logging level (default: {LOGGING_CONFIG["LOG_LEVEL"]})')
    return parser.parse_args()

def initialize_bot(args):
    """
    Initialize the trading bot based on the specified mode.
    
    Args:
        args (argparse.Namespace): The parsed arguments.
        
    Returns:
        object: The initialized bot.
    """
    # Import the appropriate modules based on the mode
    if args.mode == 'live':
        from trading.live_bot import LiveBot
        return LiveBot(args.symbol, args.timeframe, args.leverage)
    elif args.mode == 'backtest':
        from trading.backtest_bot import BacktestBot
        return BacktestBot(args.symbol, args.timeframe, args.leverage)
    elif args.mode == 'paper':
        from trading.paper_bot import PaperBot
        return PaperBot(args.symbol, args.timeframe, args.leverage)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def main():
    """
    Main function to run the trading bot.
    """
    # Display ASCII art
    print(ELVIS_ASCII)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG['LOG_TO_FILE'], log_level=log_level)
    
    # Check if API keys are set
    if not all([API_CONFIG['BINANCE_API_KEY'], API_CONFIG['BINANCE_API_SECRET']]):
        print_error(logger, "Missing API keys. Please set them in the .env file.")
        sys.exit(1)
    
    # Check if trying to run in live mode while production mode is disabled
    if args.mode == 'live' and not TRADING_CONFIG['PRODUCTION_MODE']:
        print_error(logger, "PRODUCTION_MODE is disabled in config. Cannot run in live mode.")
        print_info(logger, "To enable live trading, set PRODUCTION_MODE to True in config/config.py")
        sys.exit(1)
    
    # Initialize the bot
    try:
        bot = initialize_bot(args)
        print_info(logger, f"Bot initialized in {args.mode} mode for {args.symbol} on {args.timeframe} timeframe")
    except Exception as e:
        print_error(logger, f"Failed to initialize bot: {e}")
        sys.exit(1)
    
    # Run the bot
    try:
        bot.run()
    except KeyboardInterrupt:
        print_info(logger, "Bot stopped by user")
    except Exception as e:
        print_error(logger, f"Bot error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
