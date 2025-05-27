import argparse
import logging
import threading
import os

# Import configurations
from config import TRADING_CONFIG, API_CONFIG

# Import necessary trading modules
from trading.execution.binance_executor import BinanceExecutor
from trading.strategies.ensemble_strategy import EnsembleStrategy
from trading.utils.telegram_notifier import TelegramNotifier
from trading.data.price_fetcher import PriceFetcher
from utils.console_dashboard import ConsoleDashboard
from trading.utils.trade_history_api import app as trade_history_app

# Risk Manager
from trading.risk.advanced_risk_manager import AdvancedRiskManager

# Import new logging configuration
from utils.logger_config import setup_logging, TradingLogger


def start_trade_history_server():
    """
    Start the Trade History Flask server in a separate thread.
    """
    trade_history_app.run(host="0.0.0.0", port=5050)


def main(mode: str, log_level: str):
    """
    Main entry point for the trading bot.

    Args:
        mode (str): Trading mode, either 'paper' or 'live'.
        log_level (str): Logging level string.
    """
    # Setup enhanced logging
    trading_context = {
        "mode": mode,
        "symbol": TRADING_CONFIG.get('SYMBOL', 'BTCUSDT'),
    }
    
    logger = setup_logging(
        app_name="ELVIS",
        log_level=log_level,
        enable_file_logging=True,
        enable_json_logging=os.getenv('ENABLE_JSON_LOGS', 'false').lower() == 'true',
        enable_remote_logging=bool(os.getenv('REMOTE_LOG_HOST')),
        remote_host=os.getenv('REMOTE_LOG_HOST'),
        remote_port=int(os.getenv('REMOTE_LOG_PORT', 514)),
        trading_context=trading_context
    )
    
    # Create specialized trading logger
    trading_logger = TradingLogger(
        "main",
        symbol=TRADING_CONFIG.get('SYMBOL', 'BTCUSDT'),
        strategy="ensemble"
    )

    # Start Trade History Server in background
    threading.Thread(target=start_trade_history_server, daemon=True).start()
    logger.info("Started Trade History Server on 0.0.0.0:5050...")

    # Initialize components
    notifier = TelegramNotifier(logger, TRADING_CONFIG)
    from config import API_CONFIG
    price_fetcher = PriceFetcher(API_CONFIG)
    starting_balance = TRADING_CONFIG.get('STARTING_BALANCE', 1000.0)  # Default to 1000.0 if not set
    risk_manager = AdvancedRiskManager(logger, starting_balance=starting_balance)

    # Initialize Binance executor
    executor = BinanceExecutor(
        logger=logger,
        api_key=API_CONFIG.BINANCE_API_KEY,
        api_secret=API_CONFIG.BINANCE_API_SECRET,
        is_testnet=(mode == 'paper')
    )
    executor.initialize()

    # Initialize Ensemble strategy
    strategy = EnsembleStrategy(
        logger=logger,
        executor=executor,
        price_fetcher=price_fetcher,
        risk_manager=risk_manager,
        notifier=notifier
    )

    # Start Console Dashboard
    dashboard = ConsoleDashboard(logger, strategy, risk_manager)
    dashboard.start()

    # Main trading loop
    try:
        strategy.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Trading Bot")
    parser.add_argument("--mode", type=str, default="paper", help="Trading mode: paper or live")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Start the main bot with log level as string
    main(mode=args.mode, log_level=args.log_level.upper())
