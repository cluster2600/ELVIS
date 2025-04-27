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
import pandas as pd
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
from prometheus_client import start_http_server, Gauge, Counter, Info
import psutil
import talib
import numpy as np
import subprocess  # Import subprocess to launch trade_history_api.py

# Load environment variables
load_dotenv()

# Define all Prometheus metrics at the module level
PNL_GAUGE = Gauge('elvis_unrealized_pnl', 'Unrealized PnL')
TRADE_COUNT = Counter('elvis_trade_count_total', 'Number of trades executed', ['side'])
OPEN_POSITIONS_GAUGE = Gauge('elvis_open_positions', 'Number of open positions')
CURRENT_PRICE_GAUGE = Gauge('elvis_current_price', 'Current price of the asset')
PORTFOLIO_VALUE_GAUGE = Gauge('elvis_portfolio_value', 'Total portfolio value')
EMA_SHORT_GAUGE = Gauge('elvis_ema_short', 'Short-term EMA')
EMA_LONG_GAUGE = Gauge('elvis_ema_long', 'Long-term EMA')
RSI_GAUGE = Gauge('elvis_rsi', 'Relative Strength Index')
MACD_GAUGE = Gauge('elvis_macd', 'MACD line')
MACD_SIGNAL_GAUGE = Gauge('elvis_macd_signal', 'MACD signal line')
BB_UPPER_GAUGE = Gauge('elvis_bb_upper', 'Bollinger Bands Upper')
BB_LOWER_GAUGE = Gauge('elvis_bb_lower', 'Bollinger Bands Lower')
SMA_GAUGE = Gauge('elvis_sma', 'Simple Moving Average')
FUNDING_RATE_GAUGE = Gauge('elvis_funding_rate', 'Funding rate')
CPU_USAGE_GAUGE = Gauge('elvis_cpu_usage', 'CPU usage percentage')
MEMORY_USAGE_GAUGE = Gauge('elvis_memory_usage', 'Memory usage percentage')
ORDER_BOOK_BIDS_GAUGE = Gauge('elvis_order_book_bids', 'Number of bids in order book')
ORDER_BOOK_ASKS_GAUGE = Gauge('elvis_order_book_asks', 'Number of asks in order book')
ORDER_BOOK_BID_VOLUME_GAUGE = Gauge('elvis_order_book_bid_volume', 'Total volume of bids')
ORDER_BOOK_ASK_VOLUME_GAUGE = Gauge('elvis_order_book_ask_volume', 'Total volume of asks')
ORDER_BOOK_SPREAD_GAUGE = Gauge('elvis_order_book_spread', 'Spread between best bid and ask')
PENDING_ORDERS_GAUGE = Gauge('elvis_pending_orders', 'Total number of pending orders')
PENDING_ORDERS_BUY_GAUGE = Gauge('elvis_pending_orders_buy', 'Number of pending buy orders')
PENDING_ORDERS_SELL_GAUGE = Gauge('elvis_pending_orders_sell', 'Number of pending sell orders')

# Trade history metric (for dashboard table)
TRADE_HISTORY_INFO = Info('elvis_last_trade', 'Last trade details')

# Start Prometheus metrics server at the top level
PROMETHEUS_PORT = 8000
start_http_server(PROMETHEUS_PORT)
# TEST: Set all dashboard metrics to test values immediately after starting server
PORTFOLIO_VALUE_GAUGE.set(1000.0)
# Fetch actual price for BTC/USDT and set initial value 10,000 less
try:
    import requests
    response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
    actual_price = float(response.json()['price'])
    CURRENT_PRICE_GAUGE.set(actual_price - 10000.0)  # Ensure decimal is properly formatted
except Exception as e:
    CURRENT_PRICE_GAUGE.set(95000.0)
    print(f"Warning: Could not fetch actual BTC price, using default. Error: {e}")
EMA_LONG_GAUGE.set(94900.0)
RSI_GAUGE.set(55.0)
MACD_GAUGE.set(120.0)
TRADE_COUNT.labels(side="buy").inc(5)
TRADE_COUNT.labels(side="sell").inc(3)

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

if __name__ == "__main__":
    # Warm-up call for psutil.cpu_percent to avoid initial 0 value
    psutil.cpu_percent(interval=1)
    try:
        args = parse_arguments()
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG.get('LOG_TO_FILE', True), log_level=log_level)
        logger.info("Starting ELVIS...")
        logger.info(f"Arguments: Mode={args.mode}, Symbol={args.symbol}, Timeframe={args.timeframe}, Strategies={args.strategies}")

        try:
            from trading.execution.binance_executor import BinanceExecutor
            executor = None
            client_instance = None
            if args.mode in ["paper", "live"]:
                executor = BinanceExecutor(logger=logger)
                executor.initialize()
                client_instance = executor.client

            price_fetcher = PriceFetcher(logger, client=client_instance, symbol=args.symbol, timeframe=args.timeframe, history_limit=TRADING_CONFIG.get('DATA_LIMIT', 200))
            price_fetcher.start()

            strategy_list = select_strategies(args.strategies, logger)
            risk_manager = AdvancedRiskManager(
                max_position_size=TRADING_CONFIG['MAX_POSITION_SIZE'],
                max_daily_trades=TRADING_CONFIG['MAX_DAILY_TRADES'],
                max_daily_loss=TRADING_CONFIG['MAX_DAILY_LOSS'],
                max_drawdown=TRADING_CONFIG['MAX_DRAWDOWN'],
                risk_per_trade=TRADING_CONFIG['RISK_PER_TRADE'],
                logger=logger
            )
            performance_monitor = PerformanceMonitor(logger=logger)
            def notify(msg, level="info"):
                send_notification(logger, msg, notification_type=level)

            if args.mode == "paper":
                init_db()
                # Initialize portfolio value for paper mode
                paper_portfolio_value = 1000.0  # Set to 1000 USD for paper mode
                PORTFOLIO_VALUE_GAUGE.set(paper_portfolio_value)  # Set initial value

            # Launch trade_history_api.py in a subprocess
            import subprocess
            trade_api_process = subprocess.Popen(['python3', 'utils/trade_history_api.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Launched trade_history_api.py as a subprocess")
            
            # Launch trading_dashboard.py
            dashboard_process = subprocess.Popen(['python3', 'utils/trading_dashboard.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Launched trading_dashboard.py as a subprocess")

            running = True
            while running:
                try:
                    price = price_fetcher.get_current_price()
                    if price:
                        CURRENT_PRICE_GAUGE.set(price)
                    candle = price_fetcher.get_current_candle()
                    candles = price_fetcher.get_candle_history()
                    if not isinstance(candles, pd.DataFrame):
                        # Explicitly set columns for Binance kline data
                        data = pd.DataFrame(candles, columns=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                        ])
                        # Convert numeric columns to float
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in data.columns:
                                data[col] = pd.to_numeric(data[col], errors='coerce')
                    else:
                        data = candles
                    if data is not None and not data.empty and 'close' in data.columns and len(data) >= 50:
                        try:
                            data['rsi'] = talib.RSI(data['close'], timeperiod=14)
                            RSI_GAUGE.set(data['rsi'].iloc[-1])
                            macd, macdsignal, _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                            data['macd'] = macd
                            data['macdsignal'] = macdsignal
                            MACD_GAUGE.set(data['macd'].iloc[-1])
                            MACD_SIGNAL_GAUGE.set(data['macdsignal'].iloc[-1])
                            if 'close' in data.columns:
                                data['ema_short'] = data['close'].ewm(span=9, adjust=False).mean()
                                data['ema_long'] = data['close'].ewm(span=21, adjust=False).mean()
                                EMA_SHORT_GAUGE.set(data['ema_short'].iloc[-1])
                                EMA_LONG_GAUGE.set(data['ema_long'].iloc[-1])
                                data['sma'] = data['close'].rolling(window=20).mean()
                                SMA_GAUGE.set(data['sma'].iloc[-1])
                                std = data['close'].rolling(window=20).std()
                                data['bb_upper'] = data['sma'] + (std * 2)
                                data['bb_lower'] = data['sma'] - (std * 2)
                                BB_UPPER_GAUGE.set(data['bb_upper'].iloc[-1])
                                BB_LOWER_GAUGE.set(data['bb_lower'].iloc[-1])
                            # Set Funding Rate, Order Book, and other metrics to test values in paper mode
                            if args.mode == "paper":
                                FUNDING_RATE_GAUGE.set(0.0001)
                                ORDER_BOOK_BIDS_GAUGE.set(10)
                                ORDER_BOOK_ASKS_GAUGE.set(12)
                                ORDER_BOOK_BID_VOLUME_GAUGE.set(5.0)
                                ORDER_BOOK_ASK_VOLUME_GAUGE.set(6.0)
                                ORDER_BOOK_SPREAD_GAUGE.set(1.5)
                                PENDING_ORDERS_GAUGE.set(2)
                                PENDING_ORDERS_BUY_GAUGE.set(1)
                                PENDING_ORDERS_SELL_GAUGE.set(1)
                        except Exception as e:
                            logger.error(f"Error generating signals: {e} | Data columns: {data.columns}")
                    elif data is not None and not data.empty:
                        logger.warning(f"DataFrame missing 'close' column. Columns: {data.columns}")
                    ORDER_BOOK_BID_VOLUME_GAUGE.set(1.0)
                    ORDER_BOOK_ASK_VOLUME_GAUGE.set(1.0)
                    ORDER_BOOK_SPREAD_GAUGE.set(0.5)
                    PENDING_ORDERS_GAUGE.set(0)
                    PENDING_ORDERS_BUY_GAUGE.set(0)
                    PENDING_ORDERS_SELL_GAUGE.set(0)
                    OPEN_POSITIONS_GAUGE.set(0)
                    # Update system usage metrics
                    CPU_USAGE_GAUGE.set(psutil.cpu_percent(interval=1))
                    MEMORY_USAGE_GAUGE.set(psutil.virtual_memory().percent)

                    buy_votes = 0
                    sell_votes = 0
                    for s in strategy_list:
                        buy, sell = s.generate_signals(data)
                        if buy:
                            buy_votes += 1
                        if sell:
                            sell_votes += 1
                    buy_signal = buy_votes > len(strategy_list) / 2
                    sell_signal = sell_votes > len(strategy_list) / 2
                    if buy_signal and sell_signal:
                        logger.warning("Simultaneous buy and sell signals detected, prioritizing sell.")
                        buy_signal = False

                    # Simulate trades in paper mode
                    if args.mode == "paper":
                        logger.info(f"[PAPER MODE] Setting portfolio value gauge to {paper_portfolio_value}")
                        PORTFOLIO_VALUE_GAUGE.set(paper_portfolio_value)
                        # Simulate a trade if a signal is generated
                        import random
                        # Simulate more frequent trades by randomizing signals
                        if random.random() < 0.5:
                            buy_signal = True
                        if random.random() < 0.5:
                            sell_signal = True

                        if buy_signal:
                            logger.info("[PAPER MODE] Simulating BUY trade")
                            record_trade('buy', price, 0.01, paper_portfolio_value)
                            add_open_position('buy', price, 0.01, TRADING_CONFIG.get('LEVERAGE_MAX', 1))
                            TRADE_COUNT.labels(side="buy").inc()
                            # Update trade history info metric
                            TRADE_HISTORY_INFO.info({
                                "side": "buy",
                                "buy_price": str(price),
                                "leverage": str(TRADING_CONFIG.get('LEVERAGE_MAX', 1)),
                                "sell_price": "",
                                "pnl": ""
                            })
                        if sell_signal:
                            logger.info("[PAPER MODE] Simulating SELL trade")
                            record_trade('sell', price, 0.01, paper_portfolio_value)
                            close_open_position('buy')
                            TRADE_COUNT.labels(side="sell").inc()
                            # Update trade history info metric
                            TRADE_HISTORY_INFO.info({
                                "side": "sell",
                                "buy_price": "",
                                "leverage": str(TRADING_CONFIG.get('LEVERAGE_MAX', 1)),
                                "sell_price": str(price),
                                "pnl": str(random.uniform(-10.0, 10.0))  # Ensure decimal literals are floats
                            })
                        # Update portfolio value and open positions gauge
                        open_positions = get_open_positions()
                        OPEN_POSITIONS_GAUGE.set(len(open_positions))
                        # Simulate portfolio value change
                        paper_portfolio_value += random.uniform(-10.0, 10.0)  # Ensure decimal literals are floats
                        PORTFOLIO_VALUE_GAUGE.set(paper_portfolio_value)
                    elif executor:
                        balance_info = executor.get_balance()
                        if balance_info and isinstance(balance_info, list):
                            portfolio_value = next((float(b['balance']) for b in balance_info if b.get('asset') == 'USDT'), 0.0)
                            logger.info(f"[LIVE MODE] Setting portfolio value gauge to {portfolio_value}")
                            PORTFOLIO_VALUE_GAUGE.set(portfolio_value)
                        funding_info = executor.get_funding_rate(args.symbol)
                        if funding_info:
                            funding_rate = float(funding_info.get('fundingRate', 0.0))
                            FUNDING_RATE_GAUGE.set(funding_rate)
                        order_book_data = executor.get_order_book(args.symbol, limit=10)
                        if order_book_data:
                            order_book = {'bids': order_book_data['bids'], 'asks': order_book_data['asks']}
                            ORDER_BOOK_BIDS_GAUGE.set(len(order_book['bids']))
                            ORDER_BOOK_ASKS_GAUGE.set(len(order_book['asks']))
                except Exception as e:
                    logger.error(f"Error in event loop: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
    except Exception as e:
        logging.basicConfig(level=logging.DEBUG)
        logging.error(f"Top-level error: {e}")
