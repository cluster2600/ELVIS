#!/usr/bin/env python3
"""
Day Trading Bot for Binance Futures aiming for 1000 USD daily profit with 75x-125x Leverage.
Uses BinanceProcessor for enhanced data processing and technical indicators.
"""

import os
import sys
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, Optional, List

import ccxt
import numpy as np
import pandas as pd
import requests
import logging
from logging.handlers import RotatingFileHandler
import colorlog
import dotenv
from telebot import TeleBot
from binance.client import Client
from config_api import *
import datetime as dt
from processor_Yahoo import Yahoofinance
from processor_Binance import BinanceProcessor
from fracdiff.sklearn import FracdiffStat

# Global Configuration
TEST_MODE = False
USE_TESTNET = False
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
LEVERAGE_MIN = 75
LEVERAGE_MAX = 125
STOP_LOSS_PCT = 0.01    # 1% SL
TAKE_PROFIT_PCT = 0.03  # 3% TP
DAILY_PROFIT_TARGET_USD = 1000.0
DAILY_LOSS_LIMIT_USD = -500.0
COOLDOWN = 3600.0  # 1 hour
SLEEP_INTERVAL = 300.0  # 5 minutes
MAX_TRADES_PER_DAY = 2
MIN_CAPITAL_USD = 1000.0
TECH_INDICATORS = ['rsi', 'macd', 'dx', 'obv']  # Subset of TA-Lib indicators
DATA_LIMIT = 200  # Number of recent candles to fetch

# Logger Setup
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DayTradingBot")
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "bold_white", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red"}
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if LOG_TO_FILE:
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        file_handler = RotatingFileHandler(f"DayTradingBot_{current_datetime}.log", maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"))
        logger.addHandler(file_handler)
    return logger

log = setup_logger()

def print_error(message: str) -> None:
    log.error(message)

def print_info(message: str) -> None:
    log.info(message)

# Load environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    print_error("Missing required environment variables.")
    sys.exit(1)

# Telegram Bot Setup
bot = TeleBot(TELEGRAM_TOKEN)

def telegram_notify(message: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&parse_mode=Markdown&text={urllib.parse.quote(message)}"
        requests.get(url, timeout=10)
        print_info(f"Telegram notification sent: {message}")
    except Exception as e:
        print_error(f"Failed to send Telegram notification: {e}")

# Binance Exchange Setup
exchange_futures = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'urls': {'api': {'public': 'https://fapi.binance.com/fapi/v1', 'private': 'https://fapi.binance.com/fapi/v1'}},
})

# Initialize BinanceProcessor
processor = BinanceProcessor()

# Global State
initial_balance: Optional[float] = None
current_position: Optional[float] = None
entry_price: Optional[float] = None
last_close_time: float = 0
daily_profit_usd: float = 0.0
trade_count_today: int = 0
last_day: str = datetime.now().strftime("%Y-%m-%d")
current_leverage: int = LEVERAGE_MIN

def initialize_funds() -> None:
    global initial_balance, current_position
    try:
        print_info("Fetching initial balance from Binance futures...")
        fut_bal = exchange_futures.fetch_balance({'type': 'future'})
        initial_balance = float(fut_bal.get('USDT', {}).get("total", 0))
        current_position = float(fut_bal.get('BTC', {}).get("free", 0))
        if initial_balance < MIN_CAPITAL_USD:
            print_error(f"Initial balance {initial_balance:.2f} USD below minimum {MIN_CAPITAL_USD} USD.")
            sys.exit(1)
        print_info(f"Initial USDT: {initial_balance:.2f}, BTC: {current_position:.8f}")
    except Exception as e:
        print_error(f"Error initializing funds: {e}")
        sys.exit(1)

def display_profit() -> None:
    try:
        print_info("Calculating profit/loss...")
        bal = exchange_futures.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("total", 0))
        profit_usd = current_usdt - initial_balance
        profit_str = f"Profit: {profit_usd:.2f} USD" if profit_usd >= 0 else f"Loss: {profit_usd:.2f} USD"
        print_info(f"[OVERALL] {profit_str}")
        global daily_profit_usd
        daily_profit_usd = profit_usd
    except Exception as e:
        print_error(f"Error displaying profit: {e}")

def get_processed_data() -> pd.DataFrame:
    """Fetch and process recent data using BinanceProcessor."""
    try:
        print_info(f"Fetching and processing recent {DATA_LIMIT} candles for {SYMBOL}, timeframe: {TIMEFRAME}")
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - pd.Timedelta(hours=DATA_LIMIT)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Run processor for recent data
        data, _, _, _ = processor.run(
            ticker_list=[SYMBOL],
            start_date=start_date,
            end_date=end_date,
            time_interval=TIMEFRAME,
            technical_indicator_list=TECH_INDICATORS,
            if_vix=False  # No VIX for futures
        )
        return data
    except Exception as e:
        print_error(f"Error processing data with BinanceProcessor: {e}")
        return pd.DataFrame()

def calculate_signal(indicators: pd.Series) -> bool:
    """Generate buy signal based on processed indicators."""
    try:
        rsi = indicators['rsi']
        macd = indicators['macd']
        dx = indicators['dx']
        obv = indicators['obv']
        close = indicators['close']
        
        # Trend strength (DX > 25), momentum (MACD > 0), RSI not overbought (< 70), OBV increasing
        buy_signal = (dx > 25 and macd > 0 and rsi < 70 and obv > obv.shift(1).iloc[-1])
        print_info(f"Signal Check: RSI={rsi:.2f}, MACD={macd:.2f}, DX={dx:.2f}, OBV={obv:.2f}, Buy={buy_signal}")
        return buy_signal
    except Exception as e:
        print_error(f"Error calculating signal: {e}")
        return False

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float, leverage: int) -> Optional[float]:
    try:
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        lot_step = next((float(f.get('stepSize')) for f in filters if f.get('filterType') == 'LOT_SIZE'), 1e-8)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        free_usdt = float(bal.get('USDT', {}).get("free", 0))
        min_usdt_trade = 100.0
        min_btc_qty = min_usdt_trade / current_price

        margin_required = (calculated_qty * current_price) / leverage
        if side.upper() == 'BUY':
            if free_usdt < MIN_CAPITAL_USD or margin_required > free_usdt * 0.5:
                print_info(f"Insufficient funds: Free USDT={free_usdt:.2f}, Required={margin_required:.2f}")
                return None
            final_qty = max(calculated_qty, min_btc_qty)
            adjusted_qty = math.ceil(final_qty / lot_step) * lot_step
            if adjusted_qty * current_price / leverage > free_usdt * 0.5:
                adjusted_qty = math.floor((free_usdt * 0.5 * leverage / current_price) / lot_step) * lot_step
        elif side.upper() == 'SELL' and current_position > 0:
            adjusted_qty = math.floor(min(calculated_qty, current_position) / lot_step) * lot_step
            if adjusted_qty <= 0 or adjusted_qty * current_price < min_usdt_trade:
                return None
        else:
            return None
        print_info(f"Adjusted {side} qty: {adjusted_qty:.8f}, value: {adjusted_qty * current_price:.2f} USD, Leverage: {leverage}x")
        return adjusted_qty
    except Exception as e:
        print_error(f"Error adjusting quantity: {e}")
        return None

def execute_order(symbol: str, side: str, qty: float, price: float, leverage: int) -> Optional[Dict[str, Any]]:
    global current_position, entry_price
    try:
        exchange_futures.set_leverage(leverage, symbol)
        if side.upper() == 'BUY':
            order = exchange_futures.create_limit_buy_order(symbol, qty, price, params={"reduceOnly": False})
            current_position = qty
            entry_price = price
            msg = f"[TRADE] BUY {qty:.8f} BTC at {price:.2f} USD (Leverage: {leverage}x)"
        else:  # SELL
            order = exchange_futures.create_limit_sell_order(symbol, qty, price, params={"reduceOnly": True})
            current_position = 0
            entry_price = None
            msg = f"[TRADE] SELL {qty:.8f} BTC at {price:.2f} USD (Leverage: {leverage}x)"
        print_info(msg)
        telegram_notify(msg)
        return order
    except Exception as e:
        print_error(f"Error executing {side} order: {e}")
        telegram_notify(f"ALERT: Error executing {side} order: {e}")
        return None

def trading_logic(symbol: str) -> None:
    global last_close_time, current_position, entry_price, trade_count_today, daily_profit_usd, last_day, current_leverage

    current_day = datetime.now().strftime("%Y-%m-%d")
    if current_day != last_day:
        trade_count_today = 0
        daily_profit_usd = 0.0
        last_day = current_day
        print_info(f"New trading day: {current_day}")

    if trade_count_today >= MAX_TRADES_PER_DAY or daily_profit_usd >= DAILY_PROFIT_TARGET_USD or daily_profit_usd <= DAILY_LOSS_LIMIT_USD:
        print_info(f"Daily limit reached: Trades={trade_count_today}/{MAX_TRADES_PER_DAY}, Profit={daily_profit_usd:.2f}/{DAILY_PROFIT_TARGET_USD} USD")
        return

    data = get_processed_data()
    if data.empty or len(data) < 50:
        print_info(f"Insufficient processed data: {len(data)} rows")
        return

    current_price = exchange_futures.fetch_ticker(symbol).get('last')
    if not current_price:
        return

    current_position = exchange_futures.fetch_positions([symbol])[0]['contracts'] if exchange_futures.fetch_positions([symbol]) else 0

    if current_position > 0 and entry_price:
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= TAKE_PROFIT_PCT or profit_pct <= -STOP_LOSS_PCT:
            qty = adjust_quantity(symbol, "SELL", current_position, current_price, current_leverage)
            if qty:
                execute_order(symbol, "SELL", qty, current_price, current_leverage)
                trade_count_today += 1
                last_close_time = time.time()
                display_profit()

    elapsed_since_close = time.time() - last_close_time
    if current_position == 0 and elapsed_since_close > COOLDOWN:
        latest_indicators = data.iloc[-1]
        buy_signal = calculate_signal(latest_indicators)
        trend_strength = latest_indicators['dx'] / 100  # Normalize DX for leverage scaling
        current_leverage = int(LEVERAGE_MIN + (LEVERAGE_MAX - LEVERAGE_MIN) * trend_strength)
        current_leverage = min(max(current_leverage, LEVERAGE_MIN), LEVERAGE_MAX)

        if buy_signal:
            bal = exchange_futures.fetch_balance({'type': 'future'})
            free_usdt = float(bal.get('USDT', {}).get("free", 0))
            qty = (free_usdt * 0.5 * current_leverage) / current_price  # 50% of capital
            final_qty = adjust_quantity(symbol, "BUY", qty, current_price, current_leverage)
            if final_qty:
                order = execute_order(symbol, "BUY", final_qty, current_price, current_leverage)
                if order:
                    trade_count_today += 1
                    sl_price = entry_price * (1 - STOP_LOSS_PCT)
                    tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    exchange_futures.create_order(symbol, 'stop', 'sell', final_qty, params={'stopPrice': sl_price, 'reduceOnly': True})
                    exchange_futures.create_order(symbol, 'limit', 'sell', final_qty, tp_price, params={'reduceOnly': True})
                    print_info(f"SL set at {sl_price:.2f}, TP set at {tp_price:.2f}")

def main() -> None:
    print_info("Starting Day Trading Bot with BinanceProcessor...")
    initialize_funds()
    symbol = f"{SYMBOL}:USDT"
    try:
        exchange_futures.set_leverage(LEVERAGE_MIN, symbol)
        print_info(f"Initial leverage set to {LEVERAGE_MIN}x")
    except Exception as e:
        print_error(f"Failed to set leverage: {e}")
        sys.exit(1)

    while True:
        try:
            trading_logic(symbol)
            display_profit()
            time.sleep(SLEEP_INTERVAL)
        except Exception as e:
            print_error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()
