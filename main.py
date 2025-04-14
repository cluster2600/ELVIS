#!/usr/bin/env python3
"""
ELVIS: Enhanced Leveraged Virtual Investment System
Main entry point - Modular, config-driven crypto trading bot for Binance Futures.
Implements ML/RL strategies, advanced risk management, monitoring, and notifications.
"""

import argparse
import logging
import sys
import os
from dotenv import load_dotenv
from config import API_CONFIG, TRADING_CONFIG, LOGGING_CONFIG
from utils.logging_utils import setup_logger, print_info, print_error
from utils.notification_utils import send_notification
from utils.price_fetcher import PriceFetcher
from trading.risk.advanced_risk_manager import AdvancedRiskManager
from core.metrics.performance_monitor import PerformanceMonitor
from trading.strategies import (
    TechnicalStrategy, MeanReversionStrategy, TrendFollowingStrategy,
    EmaRsiStrategy, EnsembleStrategy, SentimentStrategy, GridStrategy
)
import time
from utils.paper_trade_db import (
    init_db, record_trade, add_open_position, close_open_position,
    get_open_positions, calculate_unrealized_pnl
)
from prometheus_client import start_http_server, Gauge, Counter

# Load environment variables
load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description='ELVIS - Enhanced Leveraged Virtual Investment System')
    parser.add_argument('--mode', type=str, choices=['live', 'backtest', 'paper'], default=TRADING_CONFIG['DEFAULT_MODE'],
                        help='Trading mode')
    parser.add_argument('--symbol', type=str, default=TRADING_CONFIG['SYMBOL'], help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='5m', help='Trading timeframe (default: 5m for more frequent trading)')
    parser.add_argument('--strategies', type=str, nargs='+',
                        choices=['technical', 'mean_reversion', 'trend_following', 'ema_rsi', 'ensemble', 'sentiment', 'grid'],
                        default=['ema_rsi', 'sentiment', 'ensemble'], help='List of strategies to combine')
    parser.add_argument('--log-level', type=str, default=LOGGING_CONFIG.get('LOG_LEVEL', 'INFO'), help='Logging level')
    return parser.parse_args()

def select_strategies(names, logger):
    strategies = {
        'technical': TechnicalStrategy,
        'mean_reversion': MeanReversionStrategy,
        'trend_following': TrendFollowingStrategy,
        'ema_rsi': EmaRsiStrategy,
        'ensemble': EnsembleStrategy,
        'sentiment': SentimentStrategy,
        'grid': GridStrategy
    }
    selected = []
    for name in names:
        if name not in strategies:
            logger.error(f"Invalid strategy: {name}")
            raise ValueError(f"Invalid strategy: {name}")
        selected.append(strategies[name](logger=logger))
    return selected

def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG.get('LOG_TO_FILE', True), log_level=log_level)
    print_info(logger, "Starting ELVIS...")
    print_info(logger, f"Arguments: Mode={args.mode}, Symbol={args.symbol}, Timeframe={args.timeframe}, Strategy={args.strategy}")

    # Config-driven initialization
    try:
        # Data pipeline (real or mock)
        price_fetcher = PriceFetcher(logger, symbol=args.symbol, timeframe=args.timeframe)
        # Strategies (ensemble)
        strategy_list = select_strategies(args.strategies, logger)
        # Risk management
        risk_manager = AdvancedRiskManager(
            max_position_size=TRADING_CONFIG['MAX_POSITION_SIZE'],
            max_daily_trades=TRADING_CONFIG['MAX_DAILY_TRADES'],
            max_daily_loss=TRADING_CONFIG['MAX_DAILY_LOSS'],
            max_drawdown=TRADING_CONFIG['MAX_DRAWDOWN'],
            risk_per_trade=TRADING_CONFIG['RISK_PER_TRADE'],
            logger=logger
        )
        # Performance monitoring
        performance_monitor = PerformanceMonitor(logger=logger)
        # Notification system
        def notify(msg, level="info"):
            send_notification(logger, msg, notification_type=level)

        # Paper mode: initialize local trade DB
        if args.mode == "paper":
            init_db()

        # Monitoring: start Prometheus server and define metrics
        start_http_server(8000)
        PNL_GAUGE = Gauge('elvis_unrealized_pnl', 'Unrealized PnL')
        TRADE_COUNT = Counter('elvis_trade_count', 'Number of trades executed')
        OPEN_POSITIONS_GAUGE = Gauge('elvis_open_positions', 'Number of open positions')
        # Main event loop
        print_info(logger, "ELVIS main event loop starting...")
        running = True
        while running:
            # 1. Fetch latest data
            price = price_fetcher.get_current_price()
            candle = price_fetcher.get_current_candle()
            # 2. Run strategy
            import pandas as pd
            candles = price_fetcher.get_candle_history()
            # Robust conversion: always ensure DataFrame
            if not isinstance(candles, pd.DataFrame):
                try:
                    data = pd.DataFrame(candles)
                except Exception as e:
                    logger.error(f"Failed to convert candles to DataFrame: {e}")
                    data = pd.DataFrame()
            else:
                data = candles
            logger.debug(f"Type of data passed to generate_signals: {type(data)}, shape: {getattr(data, 'shape', None)}, data: {data.head() if hasattr(data, 'head') else data}")
            if data is None or data.empty:
                logger.debug("Not enough data to run strategy. Waiting for more candles.")
                time.sleep(5)
                continue
            # Combine signals from all strategies (majority vote for buy/sell)
            signals_list = [s.generate_signals(data) for s in strategy_list]
            buy_votes = sum(1 for sig in signals_list if sig[0])
            sell_votes = sum(1 for sig in signals_list if sig[1])
            buy_signal = buy_votes > len(signals_list) // 2
            sell_signal = sell_votes > len(signals_list) // 2

            # Dynamic position sizing (Kelly criterion, simplified)
            # For demo: assume win_rate=0.6, reward/risk=2, can be estimated from backtest
            win_rate = 0.6
            reward_risk = 2.0
            kelly_fraction = max(0.0, min(1.0, win_rate - (1 - win_rate) / reward_risk))
            capital = 10000  # Placeholder, should be actual portfolio value
            quantity = kelly_fraction * capital / price if price > 0 else 0

            # 3. Apply risk management (placeholder)
            # 4. Execute trades (simulated in paper mode)
            if args.mode == "paper":
                # Check for open position
                open_positions = get_open_positions()
                has_position = any(pos[0] == args.symbol for pos in open_positions)
                # Buy signal
                if buy_signal and not has_position and quantity > 0:
                    add_open_position(args.symbol, price, quantity)
                    record_trade(args.symbol, "buy", price, quantity, pnl=0.0)
                    logger.info(f"Simulated BUY: {args.symbol} at {price} qty={quantity:.4f}")
                # Sell signal
                elif sell_signal and has_position:
                    # Calculate PnL
                    for pos in open_positions:
                        if pos[0] == args.symbol:
                            entry_price = pos[1]
                            qty = pos[2]
                            pnl = (price - entry_price) * qty
                            record_trade(args.symbol, "sell", price, qty, pnl=pnl)
                            close_open_position(args.symbol)
                            logger.info(f"Simulated SELL: {args.symbol} at {price}, PnL: {pnl:.2f}")
                # Log unrealized PnL
                unrealized_pnl = calculate_unrealized_pnl(args.symbol, price)
                logger.info(f"Unrealized PnL for {args.symbol}: {unrealized_pnl:.2f}")
                # Update Prometheus metrics
                PNL_GAUGE.set(unrealized_pnl)
                open_positions = get_open_positions()
                OPEN_POSITIONS_GAUGE.set(len(open_positions))
                # Count trades
                if buy_signal or sell_signal:
                    TRADE_COUNT.inc()
            # 5. Log and monitor
            performance_monitor.add_trade({
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
                'symbol': args.symbol,
                'side': 'buy' if buy_signal else 'sell' if sell_signal else 'hold',
                'price': price,
                'quantity': quantity if buy_signal or sell_signal else 0,
                'pnl': unrealized_pnl if buy_signal or sell_signal else 0
            })
            # 6. Send notifications/alerts if needed
            # 7. Sleep or wait for next tick
            time.sleep(TRADING_CONFIG.get('SLEEP_INTERVAL', 60))
            # For demo, break after one loop
            running = False

        print_info(logger, "ELVIS run completed.")
    except Exception as e:
        print_error(logger, f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
