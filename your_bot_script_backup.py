#!/usr/bin/env python3
"""
Day Trading Bot for Binance Futures aiming for 1000 USD daily profit with 75x-125x Leverage.
Enhanced with order book depth, news sentiment, backtesting, and error handling.
"""

import os
import sys
import time
import json
import math
import urllib.parse
from datetime import datetime
from typing import Any, Dict, Optional, List
import signal

import ccxt
import numpy as np
import pandas as pd
import requests
import logging
from logging.handlers import RotatingFileHandler
import colorlog
import ta
from telebot import TeleBot
from processor_Binance import BinanceProcessor
import ensemble_models
from newsapi import NewsApiClient  # Requires 'pip install newsapi-python'
import coremltools as ct  # For MLModel
import nltk

# Download NLTK data (run once, comment out after first run if desired)
nltk.download('vader_lexicon')

# Force print statements to flush immediately
import functools
print = functools.partial(print, flush=True)

# Manual .env loading with extra debugging
env_path = '/Users/maxime/BTC_BOT/BTC_BOT/.env'
print(f"Manually loading .env from: {env_path}")
if os.path.exists(env_path):
    print(f"File exists, permissions: {oct(os.stat(env_path).st_mode)[-3:]}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"Loaded {key}={value[:4]}...")
    print("Manual load complete")
else:
    print(f"Error: .env file not found at {env_path}")
    sys.exit(1)

# Verify loaded variables
print("Post-load environment check:")
for key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID', 'NEWSAPI_KEY']:
    value = os.getenv(key)
    print(f"{key}: {value[:4] if value else 'None'}...")

# Global Configuration
TEST_MODE = False
USE_TESTNET = False
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
LEVERAGE_MIN = 75
LEVERAGE_MAX = 125
STOP_LOSS_PCT = 0.01    # 1% SL
TAKE_PROFIT_PCT = 0.03  # 3% TP
CONFIDENCE_THRESHOLD = 0.9  # Adjustable via backtest
TAKER_FEE = 0.0006
MAKER_FEE = 0.0002
MIN_USDT = 1000.0
MIN_BTC = 0.00105
TRADE_PERCENTAGE = 0.5
DAILY_PROFIT_TARGET_USD = 1000.0
DAILY_LOSS_LIMIT_USD = -500.0
COOLDOWN = 3600.0
SLEEP_INTERVAL = 300.0
MAX_TRADES_PER_DAY = 2
DATA_LIMIT = 200
TECH_INDICATORS = ['rsi', 'macd', 'dx', 'obv']
YDF_MODEL_PATH = "model_rf.ydf"
NN_MODEL_PATH = "/Users/maxime/pump_project/NNModel.mlpackage"
MLX_SERVER_URL = "http://localhost:1234/v1/completions"
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')  # Loaded from .env

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

# Validate environment variables with detailed output
required_vars = {
    'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
    'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
    'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET')
}
print("Validating environment variables:")
for key, value in required_vars.items():
    print(f"{key}: {'Set' if value else 'Missing'}")
if not all(required_vars.values()):
    missing = [k for k, v in required_vars.items() if not v]
    print_error(f"Missing required environment variables: {missing}")
    print("Check values:", required_vars)
    sys.exit(1)

# Assign environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

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
    'headers': {'User-Agent': 'Mozilla/5.0', 'X-MBX-APIKEY': BINANCE_API_KEY}
})

# Initialize Processor and Models
processor = BinanceProcessor()
models_loaded = True
try:
    ydf_model = ensemble_models.load_ydf_model(YDF_MODEL_PATH)
    nn_model = ct.models.MLModel(NN_MODEL_PATH)
    ensemble_models.load_mlx_model()
except Exception as e:
    print_error(f"Failed to initialize models: {e}. Falling back to indicator-based logic.")
    models_loaded = False

newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

# Global State
initial_futures_balance: Optional[float] = None
current_position: Optional[float] = None
entry_price: Optional[float] = None
previous_position: Optional[float] = None
last_close_time: float = 0
cumulative_funding_fees: float = 0.0
current_leverage: int = LEVERAGE_MIN
daily_profit_usd: float = 0.0
trade_count_today: int = 0
last_day: str = datetime.now().strftime("%Y-%m-%d")

def initialize_funds() -> None:
    global initial_futures_balance, current_position
    try:
        print_info("Fetching initial balance from Binance futures...")
        fut_bal = exchange_futures.fetch_balance({'type': 'future'})
        initial_futures_balance = float(fut_bal.get('USDT', {}).get("total", 0))
        current_position = float(fut_bal.get('BTC', {}).get("free", 0))
        if initial_futures_balance < MIN_USDT:
            print_error(f"Initial Futures USDT balance {initial_futures_balance:.2f} below minimum {MIN_USDT} USDT.")
            sys.exit(1)
        print_info(f"Initial Futures USDT: {initial_futures_balance:.2f}, BTC: {current_position:.8f}")
    except Exception as e:
        print_error(f"Error initializing funds: {e}")
        sys.exit(1)

def display_profit() -> None:
    try:
        print_info("Calculating profit/loss...")
        bal = exchange_futures.fetch_balance({'type': 'future'})
        current_usdt = float(bal.get('USDT', {}).get("total", 0))
        profit_usd = current_usdt - initial_futures_balance - cumulative_funding_fees
        profit_str = f"Profit: {profit_usd:.2f} USD" if profit_usd >= 0 else f"Loss: {profit_usd:.2f} USD"
        print_info(f"[OVERALL] {profit_str} (USDT: {profit_usd:.2f}, Funding Fees: {cumulative_funding_fees:.2f} USDT)")
        global daily_profit_usd
        daily_profit_usd = profit_usd
    except Exception as e:
        print_error(f"Error displaying profit: {e}")

def get_processed_data() -> pd.DataFrame:
    try:
        print_info(f"Fetching and processing recent {DATA_LIMIT} candles for {SYMBOL}")
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - pd.Timedelta(hours=DATA_LIMIT)).strftime("%Y-%m-%d %H:%M:%S")
        
        data, _, _, _ = processor.run(
            ticker_list=[SYMBOL],
            start_date=start_date,
            end_date=end_date,
            time_interval=TIMEFRAME,
            technical_indicator_list=TECH_INDICATORS,
            if_vix=False
        )
        return data
    except Exception as e:
        print_error(f"Error processing data: {e}")
        return pd.DataFrame()

def get_order_book_depth(symbol: str, price: float) -> float:
    try:
        order_book = exchange_futures.fetch_order_book(symbol)
        bids = order_book['bids']
        asks = order_book['asks']
        depth_range = 0.01  # 1% range
        bid_depth = sum([vol for p, vol in bids if p >= price * (1 - depth_range)])
        ask_depth = sum([vol for p, vol in asks if p <= price * (1 + depth_range)])
        return bid_depth + ask_depth
    except Exception as e:
        print_error(f"Error fetching order book depth: {e}")
        return 0.0

def get_news_sentiment() -> float:
    if not newsapi:
        print_info("NewsAPI key missing, sentiment set to 0.0")
        return 0.0
    try:
        articles = newsapi.get_everything(q="Bitcoin", language="en", sort_by="relevancy", page_size=10)
        sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(article['title'])['compound'] for article in articles['articles']]
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
        print_error(f"Error fetching news sentiment: {e}")
        return 0.0

def prepare_features(data: pd.DataFrame, current_price: float) -> Dict[str, float]:
    latest = data.iloc[-1]
    features = {
        "price": float(latest['close']),
        "volume": float(latest['volume']),
        "rsi": float(latest['rsi']),
        "macd": float(latest['macd']),
        "adx": float(latest['dx']),
        "Order_Amount": 0.0,
        "sma": float(data['close'].rolling(window=20).mean().iloc[-1]),
        "Filled": 0.0,
        "Total": 0.0,
        "future_price": 0.0,
        "atr": float(data['close'].pct_change().rolling(window=14).std().iloc[-1] * latest['close']),
        "vol_adjusted_price": float(latest['close']),
        "volume_ma": float(data['volume'].rolling(window=20).mean().iloc[-1]),
        "signal_line": 0.0,
        "lower_bb": 0.0,
        "sma_bb": 0.0,
        "upper_bb": 0.0,
        "news_sentiment": get_news_sentiment(),
        "social_feature": 0.0,  # Placeholder for X scraping
        "order_book_depth": get_order_book_depth(f"{SYMBOL}:USDT", current_price)
    }
    return ensemble_models.ensure_required_features(features)

def indicator_fallback(data: pd.DataFrame) -> str:
    try:
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
        dx = data.iloc[-1]['dx']  # Fixed 'latest' undefined
        if sma_20 > sma_50 and dx > 25:
            return "BUY"
        elif sma_20 < sma_50 and dx > 25:
            return "SELL"
        return "HOLD"
    except Exception as e:
        print_error(f"Indicator fallback failed: {e}")
        return "HOLD"

def adjust_quantity(symbol: str, side: str, calculated_qty: float, current_price: float) -> Optional[float]:
    try:
        filters = exchange_futures.markets[symbol]['info'].get('filters', [])
        lot_step = next((float(f.get('stepSize')) for f in filters if f.get('filterType') == 'LOT_SIZE'), 1e-8)
        bal = exchange_futures.fetch_balance({'type': 'future'})
        free_usdt = float(bal.get('USDT', {}).get("free", 0))
        min_usdt_trade = 100.0
        min_btc_qty = min_usdt_trade / current_price

        margin_required = (calculated_qty * current_price) / current_leverage
        if side.upper() == 'BUY':
            if free_usdt < MIN_USDT or margin_required > free_usdt * TRADE_PERCENTAGE:
                print_info(f"Insufficient funds: Free USDT={free_usdt:.2f}, Required={margin_required:.2f}")
                return None
            final_qty = max(calculated_qty, min_btc_qty)
            adjusted_qty = math.ceil(final_qty / lot_step) * lot_step
            if adjusted_qty * current_price < min_usdt_trade:
                adjusted_qty = math.ceil(min_btc_qty / lot_step) * lot_step
            if adjusted_qty * current_price / current_leverage > free_usdt * TRADE_PERCENTAGE:
                return None
        elif side.upper() == 'SELL' and current_position > 0:
            adjusted_qty = math.floor(min(calculated_qty, current_position) / lot_step) * lot_step
            if adjusted_qty <= 0 or adjusted_qty * current_price < min_usdt_trade:
                return None
        else:
            return None
        print_info(f"Adjusted {side} qty: {adjusted_qty:.8f}, value: {adjusted_qty * current_price:.2f} USD")
        return adjusted_qty
    except Exception as e:
        print_error(f"Error adjusting quantity: {e}")
        return None

def execute_order(symbol: str, side: str, qty: float, price: float) -> Optional[Dict[str, Any]]:
    global current_position, entry_price
    try:
        exchange_futures.set_leverage(current_leverage, symbol)
        if side.upper() == 'BUY':
            order = exchange_futures.create_limit_buy_order(symbol, qty, price, params={"reduceOnly": False})
            current_position = qty
            entry_price = price
            msg = f"[TRADE] BUY {qty:.8f} BTC at {price:.2f} USD (Leverage: {current_leverage}x)"
        else:  # SELL
            order = exchange_futures.create_limit_sell_order(symbol, qty, price, params={"reduceOnly": True})
            current_position = 0
            entry_price = None
            msg = f"[TRADE] SELL {qty:.8f} BTC at {price:.2f} USD (Leverage: {current_leverage}x)"
        print_info(msg)
        telegram_notify(msg)
        return order
    except Exception as e:
        print_error(f"Error executing {side} order: {e}")
        telegram_notify(f"ALERT: Error executing {side} order: {e}")
        return None

def backtest_strategy(start_date: str = "2024-01-01", end_date: str = "2025-02-28") -> float:
    try:
        print_info(f"Backtesting strategy from {start_date} to {end_date}")
        data, _, _, _ = processor.run(
            ticker_list=[SYMBOL],
            start_date=start_date,
            end_date=end_date,
            time_interval=TIMEFRAME,
            technical_indicator_list=TECH_INDICATORS,
            if_vix=False
        )
        if len(data) < 50:
            print_error("Insufficient data for backtest")
            return CONFIDENCE_THRESHOLD

        capital = 1000.0
        position = 0.0
        entry_price = 0.0
        trades = 0
        for i in range(50, len(data)):
            subset = data.iloc[:i+1]
            price = subset['close'].iloc[-1]
            features = prepare_features(subset, price)
            decision = indicator_fallback(subset) if not models_loaded else ensemble_models.get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)[0]
            confidence = 1.0 if not models_loaded else ensemble_models.get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)[1]

            if decision == "BUY" and confidence >= CONFIDENCE_THRESHOLD and position == 0:
                position = capital * TRADE_PERCENTAGE * LEVERAGE_MIN / price
                entry_price = price
                trades += 1
            elif decision == "SELL" and position > 0:
                profit = (price - entry_price) * position - (price * position * TAKER_FEE * 2)
                capital += profit
                position = 0
                trades += 1
            elif position > 0:
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= TAKE_PROFIT_PCT or profit_pct <= -STOP_LOSS_PCT:
                    profit = profit_pct * position * entry_price - (price * position * TAKER_FEE * 2)
                    capital += profit
                    position = 0
                    trades += 1

        print_info(f"Backtest Result: Final Capital={capital:.2f} USD, Trades={trades}")
        return 0.85 if capital < 1000 else 0.9  # Tune confidence based on profit
    except Exception as e:
        print_error(f"Backtest failed: {e}")
        return CONFIDENCE_THRESHOLD

def trading_logic(symbol: str) -> None:
    global previous_position, last_close_time, current_position, entry_price, trade_count_today, daily_profit_usd, last_day, current_leverage, CONFIDENCE_THRESHOLD

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
            qty = adjust_quantity(symbol, "SELL", current_position, current_price)
            if qty:
                execute_order(symbol, "SELL", qty, current_price)
                trade_count_today += 1
                last_close_time = time.time()
                display_profit()

    elapsed_since_close = time.time() - last_close_time
    if elapsed_since_close > COOLDOWN:
        features = prepare_features(data, current_price)
        decision, confidence = (indicator_fallback(data), 1.0) if not models_loaded else ensemble_models.get_ensemble_decision(features, ydf_model, nn_model, mlx_url=MLX_SERVER_URL)
        trend_strength = features['adx'] / 100
        current_leverage = int(LEVERAGE_MIN + (LEVERAGE_MAX - LEVERAGE_MIN) * trend_strength)
        current_leverage = min(max(current_leverage, LEVERAGE_MIN), LEVERAGE_MAX)

        print_info(f"Decision: {decision}, Confidence: {confidence:.2f}, Leverage: {current_leverage}x")
        if decision == "BUY" and confidence >= CONFIDENCE_THRESHOLD and current_position == 0:
            bal = exchange_futures.fetch_balance({'type': 'future'})
            free_usdt = float(bal.get('USDT', {}).get("free", 0))
            qty = TRADE_PERCENTAGE * (free_usdt / current_price)
            final_qty = adjust_quantity(symbol, "BUY", qty, current_price)
            if final_qty:
                order = execute_order(symbol, "BUY", final_qty, current_price)
                if order:
                    trade_count_today += 1
                    sl_price = entry_price * (1 - STOP_LOSS_PCT)
                    tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    exchange_futures.create_order(symbol, 'stop', 'sell', final_qty, params={'stopPrice': sl_price, 'reduceOnly': True})
                    exchange_futures.create_order(symbol, 'limit', 'sell', final_qty, tp_price, params={'reduceOnly': True})
                    print_info(f"SL set at {sl_price:.2f}, TP set at {tp_price:.2f}")
        elif decision == "SELL" and confidence >= CONFIDENCE_THRESHOLD and current_position > 0:
            qty = adjust_quantity(symbol, "SELL", current_position, current_price)
            if qty:
                execute_order(symbol, "SELL", qty, current_price)
                trade_count_today += 1
                last_close_time = time.time()
                display_profit()

    previous_position = current_position

def main() -> None:
    print_info("Starting Day Trading Bot...")
    initialize_funds()
    symbol = f"{SYMBOL}:USDT"
    try:
        exchange_futures.set_leverage(LEVERAGE_MIN, symbol)
        current_leverage = LEVERAGE_MIN
        print_info(f"Leverage set to {LEVERAGE_MIN}x")
    except Exception as e:
        print_error(f"Failed to set {LEVERAGE_MIN}x leverage: {e}. Using default.")
        sys.exit(1)

    # Run backtest and adjust confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = backtest_strategy()

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