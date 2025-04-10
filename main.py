#!/usr/bin/env python3
"""
Main entry point for the ELVIS project with Prometheus integration.
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
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import pandas as pd
import numpy as np
import talib
import psutil
from collections import deque
from prometheus_client import start_http_server, Gauge, Counter
from utils import setup_logger, print_info, print_error
from config import API_CONFIG, TRADING_CONFIG, LOGGING_CONFIG
from utils.trading_dashboard import TradingDashboard

# Load environment variables from .env file
load_dotenv()

# Database configuration from .env
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Prometheus metrics
PRICE_GAUGE = Gauge('elvis_current_price', 'Current BTC/USDT price')
PORTFOLIO_VALUE_GAUGE = Gauge('elvis_portfolio_value', 'Portfolio value in USDT')
TRADE_COUNT = Counter('elvis_trade_count', 'Number of trades executed', ['side'])
CPU_USAGE_GAUGE = Gauge('elvis_cpu_usage', 'CPU usage percentage')
MEMORY_USAGE_GAUGE = Gauge('elvis_memory_usage', 'Memory usage percentage')
OPEN_POSITIONS_GAUGE = Gauge('elvis_open_positions', 'Number of open positions')
# New metrics for indicators
EMA_SHORT_GAUGE = Gauge('elvis_ema_short', 'Short-term EMA (9)')
EMA_LONG_GAUGE = Gauge('elvis_ema_long', 'Long-term EMA (21)')
RSI_GAUGE = Gauge('elvis_rsi', 'Relative Strength Index')
MACD_GAUGE = Gauge('elvis_macd', 'MACD value')
MACD_SIGNAL_GAUGE = Gauge('elvis_macd_signal', 'MACD signal line')
SMA_GAUGE = Gauge('elvis_sma', 'Simple Moving Average (20)')
BB_UPPER_GAUGE = Gauge('elvis_bb_upper', 'Bollinger Band Upper')
BB_LOWER_GAUGE = Gauge('elvis_bb_lower', 'Bollinger Band Lower')
# New metrics for order book and pending orders
ORDER_BOOK_BIDS_GAUGE = Gauge('elvis_order_book_bids', 'Number of bids in order book')
ORDER_BOOK_ASKS_GAUGE = Gauge('elvis_order_book_asks', 'Number of asks in order book')
ORDER_BOOK_BID_VOLUME_GAUGE = Gauge('elvis_order_book_bid_volume', 'Total volume of bids in order book')
ORDER_BOOK_ASK_VOLUME_GAUGE = Gauge('elvis_order_book_ask_volume', 'Total volume of asks in order book')
ORDER_BOOK_SPREAD_GAUGE = Gauge('elvis_order_book_spread', 'Spread between best bid and ask')
PENDING_ORDERS_GAUGE = Gauge('elvis_pending_orders', 'Number of pending orders')
PENDING_ORDERS_BUY_GAUGE = Gauge('elvis_pending_orders_buy', 'Number of pending buy orders')
PENDING_ORDERS_SELL_GAUGE = Gauge('elvis_pending_orders_sell', 'Number of pending sell orders')
# Sentiment metrics
FUNDING_RATE_GAUGE = Gauge('elvis_funding_rate', 'Current funding rate percentage')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

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
                        default='ema_rsi', help='Trading strategy (default: ema_rsi)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=LOGGING_CONFIG.get('LOG_LEVEL', 'INFO'),
                        help=f'Logging level (default: {LOGGING_CONFIG["LOG_LEVEL"]})')
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
        logger.info("Initializing TradingDashboard for paper mode with Prometheus...")
        return TradingDashboardWithDB(args.symbol, args.timeframe, args.leverage, strategy_instance, logger)
    else:
        logger.error(f"Invalid mode specified: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")

class TradingDashboardWithDB(TradingDashboard):
    """Enhanced TradingDashboard with database and Prometheus integration."""
    def __init__(self, symbol, timeframe, leverage, strategy, logger):
        super().__init__(logger=logger)
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.strategy = strategy
        self.logger = logger
        self.conn = get_db_connection()
        self.config['leverage'] = leverage
        self.running = True
        self.executor.set_leverage(self.symbol, self.leverage)
        self.trade_size = 0.002
        self.portfolio_value = TRADING_CONFIG.get('MIN_CAPITAL_USD', 10000)
        self.config['portfolio_value'] = self.portfolio_value
        self.pending_orders = []
        self.alerts = {'price_high': 80000, 'price_low': 70000}
        self.trade_history = deque(maxlen=100)

    def log_trade(self, timestamp, symbol, side, price, quantity, pnl=0.0):
        """Logs a trade to the database and updates Prometheus metrics."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (timestamp, symbol, side, price, quantity, pnl))
                self.conn.commit()
                trade = {
                    'symbol': symbol,
                    'type': side.upper(),
                    'size': quantity,
                    'price': price,
                    'pnl': pnl,
                    'time': int(timestamp.timestamp() * 1000)
                }
                self.performance_monitor.add_trade({
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'quantity': quantity,
                    'pnl': pnl
                })
                self.trade_history.append(trade)
                TRADE_COUNT.labels(side=side).inc()
                self.logger.info(f"Trade logged: {symbol} {side} at {price}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to log trade: {e}")

    def update_open_position(self, symbol, size, entry_price, current_price, leverage, pnl=0.0):
        """Updates or inserts an open position in the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id FROM open_positions WHERE symbol = %s", (symbol,))
                result = cur.fetchone()
                position = {
                    'symbol': symbol,
                    'size': size,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'leverage': leverage,
                    'pnl': pnl
                }
                if result:
                    cur.execute("""
                        UPDATE open_positions
                        SET size = %s, entry_price = %s, current_price = %s, leverage = %s, pnl = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = %s
                    """, (size, entry_price, current_price, leverage, pnl, symbol))
                else:
                    cur.execute("""
                        INSERT INTO open_positions (symbol, size, entry_price, current_price, leverage, pnl)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (symbol, size, entry_price, current_price, leverage, pnl))
                self.conn.commit()
                self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != symbol] + [position]
                OPEN_POSITIONS_GAUGE.set(len(self.config['open_positions']))
                self.logger.info(f"Open position updated: {symbol} at {current_price}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to update open position: {e}")

    def _process_signals(self, signals, current_price, position):
        """Process strategy signals and return buy/sell actions."""
        buy_signal = False
        sell_signal = False
        if isinstance(signals, tuple):
            buy_signal, sell_signal = signals
        elif isinstance(signals, dict):
            signal = signals.get('signal', 'HOLD')
            buy_signal = signal == 'BUY'
            sell_signal = signal == 'SELL'
        min_notional = 100
        quantity = max(self.trade_size, min_notional / current_price)
        return buy_signal and not position, sell_signal and position, quantity

    def _manual_buy(self, order_type='market', limit_price=None):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping manual buy: invalid price")
            return
        min_notional = 100
        quantity = max(self.trade_size, min_notional / current_price)
        timestamp = datetime.utcnow()
        if order_type == 'market':
            entry_price = current_price * 1.001
            self.log_trade(timestamp, self.symbol, 'buy', entry_price, quantity)
            self.update_open_position(self.symbol, quantity, entry_price, current_price, self.leverage)
            self.portfolio_value -= quantity * entry_price / self.leverage
            self.config['portfolio_value'] = self.portfolio_value
            self.logger.info("Manual BUY (market) simulated in paper mode")
        elif order_type in ['limit', 'stop']:
            self.pending_orders.append({
                'type': order_type,
                'side': 'buy',
                'price': limit_price,
                'quantity': quantity,
                'timestamp': timestamp
            })
            self.logger.info(f"Manual {order_type.upper()} BUY order placed at {limit_price}")

    def _manual_sell(self, order_type='market', limit_price=None):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping manual sell: invalid price")
            return
        position = next((p for p in self.config['open_positions'] if p['symbol'] == self.symbol), None)
        if not position:
            self.logger.debug("No position to sell")
            return
        timestamp = datetime.utcnow()
        if order_type == 'market':
            exit_price = current_price * 0.999
            pnl = (exit_price - position['entry_price']) * position['size']
            self.log_trade(timestamp, self.symbol, 'sell', exit_price, position['size'], pnl)
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM open_positions WHERE symbol = %s", (self.symbol,))
                self.conn.commit()
            self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != self.symbol]
            self.portfolio_value += (position['size'] * exit_price / self.leverage) + pnl
            self.config['portfolio_value'] = self.portfolio_value
            self.logger.info("Manual SELL (market) simulated in paper mode")
        elif order_type in ['limit', 'stop']:
            self.pending_orders.append({
                'type': order_type,
                'side': 'sell',
                'price': limit_price,
                'quantity': position['size'],
                'timestamp': timestamp
            })
            self.logger.info(f"Manual {order_type.upper()} SELL order placed at {limit_price}")

    def _adjust_leverage(self, delta: int):
        new_leverage = max(1, min(125, self.leverage + delta))
        self.leverage = new_leverage
        self.config['leverage'] = new_leverage
        self.executor.set_leverage(self.symbol, new_leverage)
        self.logger.info(f"Leverage adjusted to {new_leverage}x in paper mode")

    def _close_all_positions(self):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping close all: invalid price")
            return
        for position in self.config['open_positions']:
            timestamp = datetime.utcnow()
            exit_price = current_price * 0.999
            pnl = (exit_price - position['entry_price']) * position['size']
            self.log_trade(timestamp, position['symbol'], 'sell', exit_price, position['size'], pnl)
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM open_positions WHERE symbol = %s", (position['symbol'],))
                self.conn.commit()
            self.portfolio_value += (position['size'] * exit_price / self.leverage) + pnl
        self.config['open_positions'] = []
        self.config['portfolio_value'] = self.portfolio_value
        self.logger.info("All positions closed in paper mode")

    def _set_tp_sl(self, tp_price=None, sl_price=None):
        position = next((p for p in self.config['open_positions'] if p['symbol'] == self.symbol), None)
        if not position:
            self.logger.debug("No position to set TP/SL")
            return
        if tp_price:
            self.pending_orders.append({
                'type': 'limit',
                'side': 'sell',
                'price': tp_price,
                'quantity': position['size'],
                'timestamp': datetime.utcnow(),
                'purpose': 'take_profit'
            })
            self.logger.info(f"Take Profit set at {tp_price}")
        if sl_price:
            self.pending_orders.append({
                'type': 'stop',
                'side': 'sell',
                'price': sl_price,
                'quantity': position['size'],
                'timestamp': datetime.utcnow(),
                'purpose': 'stop_loss'
            })
            self.logger.info(f"Stop Loss set at {sl_price}")

    def _check_pending_orders(self, current_price):
        executed = []
        for order in self.pending_orders[:]:
            if order['type'] == 'limit' and order['side'] == 'buy' and current_price <= order['price']:
                self._manual_buy('market')
                executed.append(order)
            elif order['type'] == 'limit' and order['side'] == 'sell' and current_price >= order['price']:
                self._manual_sell('market')
                executed.append(order)
            elif order['type'] == 'stop' and order['side'] == 'buy' and current_price >= order['price']:
                self._manual_buy('market')
                executed.append(order)
            elif order['type'] == 'stop' and order['side'] == 'sell' and current_price <= order['price']:
                self._manual_sell('market')
                executed.append(order)
        self.pending_orders = [o for o in self.pending_orders if o not in executed]

    def _check_alerts(self, current_price):
        if current_price >= self.alerts['price_high']:
            self.logger.warning(f"Price alert: BTC above {self.alerts['price_high']}")
        elif current_price <= self.alerts['price_low']:
            self.logger.warning(f"Price alert: BTC below {self.alerts['price_low']}")

    def _update_all_data(self):
        self.logger.info("Updating all data from Binance")
        self.config['PRODUCTION_MODE'] = not self.is_testnet
        
        try:
            account = self.executor.client.account()
            self.config['portfolio_value'] = float(account['totalWalletBalance'])
            self.config['available_margin'] = float(account['availableBalance'])
            PORTFOLIO_VALUE_GAUGE.set(self.config['portfolio_value'])
        except Exception as e:
            self.logger.error(f"Failed to fetch account info: {e}")
        
        symbol = 'BTCUSDT'
        try:
            ticker = self.executor.client.mark_price(symbol=symbol)
            self.config['current_price'] = float(ticker['markPrice'])
            self.config['price_history'].append(self.config['current_price'])
            PRICE_GAUGE.set(self.config['current_price'])
            
            spot_ticker = self.spot_client.get_symbol_ticker(symbol=symbol)
            self.config['spot_price'] = float(spot_ticker['price'])
            
            order_book = self.executor.client.depth(symbol=symbol, limit=5)
            self.config['order_book'] = {'bids': order_book['bids'], 'asks': order_book['asks']}
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            ORDER_BOOK_BIDS_GAUGE.set(len(bids))
            ORDER_BOOK_ASKS_GAUGE.set(len(asks))
            
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            ORDER_BOOK_BID_VOLUME_GAUGE.set(bid_volume)
            ORDER_BOOK_ASK_VOLUME_GAUGE.set(ask_volume)
            
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = best_ask - best_bid
                ORDER_BOOK_SPREAD_GAUGE.set(spread)
            
            funding = self.executor.client.funding_rate(symbol=symbol, limit=1)
            self.config['funding_rate'] = float(funding[0]['fundingRate']) * 100
            FUNDING_RATE_GAUGE.set(self.config['funding_rate'])
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}", exc_info=True)
        
        self._update_technical_indicators()
        try:
            account = self.executor.client.account()
            self.logger.debug(f"Account positions: {account['positions']}")
            real_positions = []
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    size = float(pos['positionAmt'])
                    entry_price = None
                    for field in ['avgPrice', 'entryPrice', 'price']:
                        if field in pos:
                            entry_price = float(pos[field])
                            break
                    
                    if entry_price is None:
                        self.logger.warning(f"Could not find entry price for position: {pos}")
                        continue
                        
                    leverage = float(pos['leverage'])
                    pnl = float(pos['unrealizedProfit'])
                    real_positions.append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': self.config['current_price'],
                        'leverage': leverage,
                        'pnl': pnl,
                        'time': int(time.time() * 1000)
                    })
            
            self.config['open_positions'] = self.config['local_positions'].copy()
            for real_pos in real_positions:
                matching_local = next((p for p in self.config['open_positions'] if p['symbol'] == real_pos['symbol'] and p['entry_price'] == real_pos['entry_price']), None)
                if matching_local:
                    self.config['open_positions'] = [p for p in self.config['open_positions'] if p != matching_local]
                self.config['open_positions'].append(real_pos)
            
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
            
            OPEN_POSITIONS_GAUGE.set(len(self.config['open_positions']))
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}", exc_info=True)
            self.config['open_positions'] = self.config['local_positions'].copy()
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
        
        try:
            trades = self.executor.client.get_account_trades(symbol=symbol, limit=10)
            self.config['recent_trades'] = deque([
                {
                    'symbol': t['symbol'],
                    'type': t['side'].lower(),
                    'size': float(t['qty']),
                    'price': float(t['price']),
                    'pnl': float(t['realizedPnl']),
                    'time': t['time']
                }
                for t in trades
            ], maxlen=10)
            for trade in trades:
                self.performance_monitor.add_trade({
                    'timestamp': datetime.fromtimestamp(trade['time'] / 1000).isoformat(),
                    'symbol': trade['symbol'],
                    'side': trade['side'].lower(),
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'pnl': float(trade['realizedPnl'])
                })
            self.logger.debug(f"Fetched recent trades: {len(self.config['recent_trades'])} trades")
        except Exception as e:
            self.logger.error(f"Failed to fetch recent trades: {e}", exc_info=True)
        
        pending_buy_orders = [o for o in self.pending_orders if o['side'] == 'buy']
        pending_sell_orders = [o for o in self.pending_orders if o['side'] == 'sell']
        PENDING_ORDERS_GAUGE.set(len(self.pending_orders))
        PENDING_ORDERS_BUY_GAUGE.set(len(pending_buy_orders))
        PENDING_ORDERS_SELL_GAUGE.set(len(pending_sell_orders))
        
        self.config['cpu_usage'] = psutil.cpu_percent()
        self.config['memory_usage'] = psutil.virtual_memory().percent
        CPU_USAGE_GAUGE.set(self.config['cpu_usage'])
        MEMORY_USAGE_GAUGE.set(self.config['memory_usage'])
        # Update indicator metrics
        indicators = self.config.get('indicators', {})
        EMA_SHORT_GAUGE.set(indicators.get('ema_short', 0.0))
        EMA_LONG_GAUGE.set(indicators.get('ema_long', 0.0))
        RSI_GAUGE.set(indicators.get('rsi', 0.0))
        MACD_GAUGE.set(indicators.get('macd', {}).get('macd', 0.0))
        MACD_SIGNAL_GAUGE.set(indicators.get('macd', {}).get('signal', 0.0))
        SMA_GAUGE.set(indicators.get('sma', 0.0))
        BB_UPPER_GAUGE.set(indicators.get('bb_upper', 0.0))
        BB_LOWER_GAUGE.set(indicators.get('bb_lower', 0.0))

    def run(self):
        """Runs paper trading with Prometheus metrics exposed."""
        self.logger.info("Starting paper trading with Prometheus metrics...")
        start_http_server(8000)
        start_time = time.time()
        while self.running:
            try:
                self._update_all_data()
                current_price = self.config['current_price']
                if current_price <= 0:
                    self.logger.debug("Skipping trade execution: invalid price")
                    time.sleep(5)
                    continue

                price_data = list(self.config['price_history'])
                if len(price_data) < 50:
                    self.logger.debug(f"Insufficient price data: {len(price_data)} prices")
                    time.sleep(5)
                    continue

                df = pd.DataFrame({'close': price_data})
                signals = self.strategy.generate_signals(df)
                position = next((p for p in self.config['open_positions'] if p['symbol'] == self.symbol), None)

                buy_signal, sell_signal, quantity = self._process_signals(signals, current_price, position)

                if buy_signal:
                    timestamp = datetime.utcnow()
                    self.logger.info(f"Signal: BUY at {current_price}")
                    entry_price = current_price * 1.001
                    self.log_trade(timestamp, self.symbol, 'buy', entry_price, quantity)
                    self.update_open_position(self.symbol, quantity, entry_price, current_price, self.leverage)
                    self.portfolio_value -= quantity * entry_price / self.leverage
                    self.config['portfolio_value'] = self.portfolio_value

                elif sell_signal and position:
                    timestamp = datetime.utcnow()
                    self.logger.info(f"Signal: SELL at {current_price}")
                    exit_price = current_price * 0.999
                    pnl = (exit_price - position['entry_price']) * position['size']
                    self.log_trade(timestamp, self.symbol, 'sell', exit_price, position['size'], pnl)
                    with self.conn.cursor() as cur:
                        cur.execute("DELETE FROM open_positions WHERE symbol = %s", (self.symbol,))
                        self.conn.commit()
                    self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != self.symbol]
                    self.portfolio_value += (position['size'] * exit_price / self.leverage) + pnl
                    self.config['portfolio_value'] = self.portfolio_value

                self._check_pending_orders(current_price)
                self._check_alerts(current_price)

                for pos in self.config['open_positions']:
                    pos['current_price'] = current_price
                    pos['pnl'] = (current_price - pos['entry_price']) * pos['size']
                    self.update_open_position(pos['symbol'], pos['size'], pos['entry_price'], current_price, pos['leverage'], pos['pnl'])

                if len(price_data) >= 20:
                    df['sma'] = talib.SMA(df['close'], timeperiod=20)
                    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20)
                    df['volume'] = 100
                    self.config['indicators'].update({
                        'sma': df['sma'].iloc[-1],
                        'bb_upper': df['bb_upper'].iloc[-1],
                        'bb_lower': df['bb_lower'].iloc[-1],
                        'volume': df['volume'].iloc[-1]
                    })

                self.config['sentiment'] = {
                    'funding_rate': self.config.get('funding_rate', 0.0),
                    'open_interest': 0.0,
                    'long_short_ratio': 0.0,
                    'fear_greed_index': 50
                }

                self.config['uptime'] = int(time.time() - start_time)
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in paper trading loop: {e}", exc_info=True)
                time.sleep(5)

        self.conn.close()
        self.logger.info("Paper trading stopped.")

def main():
    """Main execution function."""
    print(ELVIS_ASCII)
    args = parse_arguments()
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("ELVIS", log_to_file=LOGGING_CONFIG.get('LOG_TO_FILE', True), log_level=log_level)
    
    logger.info("Starting ELVIS...")
    logger.info(f"Arguments: Mode={args.mode}, Symbol={args.symbol}, Timeframe={args.timeframe}, Strategy={args.strategy}, Leverage={args.leverage}")

    if args.mode == 'live' and not TRADING_CONFIG.get('PRODUCTION_MODE', False):
        print_error(logger, "PRODUCTION_MODE is disabled in config.py. Cannot run in live mode for safety. Set PRODUCTION_MODE = True to enable live trading.")
        sys.exit(1)
    elif args.mode == 'paper':
        print_info(logger, "Running in paper trading mode with Prometheus integration.")
    elif args.mode == 'backtest':
        print_info(logger, "Running in backtesting mode.")

    try:
        bot = initialize_bot(args, logger)
        print_info(logger, f"Bot initialized successfully for {args.mode} mode.")
        bot.run()
        logger.info(f"ELVIS {args.mode} run completed.")
    except ValueError as ve:
        print_error(logger, f"Configuration error: {ve}")
        sys.exit(1)
    except ImportError as ie:
        print_error(logger, f"Import error: {ie}. Ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print_error(logger, f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()