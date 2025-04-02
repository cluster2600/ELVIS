"""
Paper trading bot for the ELVIS project.
This module provides a paper trading implementation for testing strategies without real money.
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from utils.price_fetcher import PriceFetcher
from trading.strategies.base_strategy import BaseStrategy
from trading.risk.risk_manager import RiskManager
from core.metrics.performance_monitor import PerformanceMonitor
from utils.console_dashboard import ConsoleDashboardManager
from config import TRADING_CONFIG

class PaperBot:
    def __init__(self, symbol: str, timeframe: str, leverage: int, strategy=None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ELVIS.PaperBot")
        self.stop_event = threading.Event()
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = TRADING_CONFIG['MIN_CAPITAL_USD']
        self.running = False
        
        self.logger.info("Initializing paper trading bot...")
        self.data_history = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        self.history_lock = threading.Lock()
        self.price_fetcher = PriceFetcher(self.logger, symbol=self.symbol, timeframe='1m')
        self.price_fetcher.start()
        
        if not isinstance(strategy, BaseStrategy):
            self.logger.error(f"Invalid strategy object provided: {type(strategy)}. Must inherit from BaseStrategy.")
            raise TypeError("Strategy must inherit from BaseStrategy")
        self.strategy = strategy
        
        self.risk_manager = RiskManager(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.dashboard_manager = ConsoleDashboardManager(self.logger)
        
        self.logger.info(f"Paper trading bot initialized for {symbol} on {timeframe} timeframe with {leverage}x leverage using {self.strategy.__class__.__name__}")

    def _process_new_candle(self, candle: Dict[str, Any]):
        if not candle or candle.get('close') is None or candle.get('close') <= 0:
            self.logger.debug(f"Received invalid or empty candle: {candle}")
            return

        current_price = candle['close']
        try:
            candle_time = pd.to_datetime(candle['T'], unit='ms')
        except (KeyError, ValueError):
            self.logger.error(f"Invalid timestamp in candle: {candle}. Using current time.")
            candle_time = pd.Timestamp.now()

        new_data = pd.DataFrame([{
            'date': candle_time,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        
        with self.history_lock:
            self.data_history = pd.concat([self.data_history, new_data], ignore_index=True)
            self.data_history = self.data_history.iloc[-200:]
            strategy_data = self.data_history.copy()

        if not strategy_data.empty and len(strategy_data) >= 50:
            try:
                buy_signal, sell_signal = self.strategy.generate_signals(strategy_data)
                if buy_signal and self.position == 0:
                    self.execute_buy(current_price, strategy_data)
                elif sell_signal and self.position > 0:
                    self.execute_sell(current_price)
            except Exception as e:
                self.logger.error(f"Error during signal generation or trade execution: {e}", exc_info=True)

        if self.position > 0:
            self.check_exit_conditions(current_price)

        self._update_console_dashboard(strategy_data, candle)

    def run(self):
        self.logger.info("Starting paper trading bot run loop...")
        self.running = True
        
        dashboard_thread = threading.Thread(target=self.dashboard_manager.start_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(1)

        try:
            while self.running and not self.stop_event.is_set():
                candle = self.price_fetcher.get_current_candle()
                if candle:
                    self.logger.debug(f"Processing candle: {candle}")
                    self._process_new_candle(candle)
                else:
                    time.sleep(0.1)
                if not self.dashboard_manager.is_running():
                    self.logger.info("Dashboard closed, stopping paper bot.")
                    self.running = False
        except KeyboardInterrupt:
            self.logger.info("Paper trading bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            self.logger.error(f"Unhandled error in paper trading bot run loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.logger.info("Stopping paper trading bot...")
        self.running = False
        self.stop_event.set()
        if self.price_fetcher:
            self.price_fetcher.stop()
        if self.dashboard_manager:
            self.dashboard_manager.stop_dashboard()

    def execute_buy(self, price: float, data: pd.DataFrame):
        if not self.risk_manager.check_trade_limits():
            self.logger.warning("Trade limits reached. Skipping buy signal.")
            return
        
        volatility = 0.02  # Default
        if not data.empty and len(data) > 14:
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                recent_data = data.tail(14)
                if recent_data['close'].mean() != 0:
                    volatility = (recent_data['high'].mean() - recent_data['low'].mean()) / recent_data['close'].mean()

        position_size = self.risk_manager.calculate_position_size(self.portfolio_value, price, volatility)
        entry_price = price * 1.001
        self.position = position_size
        self.entry_price = entry_price
        self.portfolio_value -= position_size * entry_price
        
        stop_loss = self.entry_price * (1 - TRADING_CONFIG.get('STOP_LOSS_PCT', 0.01))
        take_profit = self.entry_price * (1 + TRADING_CONFIG.get('TAKE_PROFIT_PCT', 0.03))
        if len(data) >= 20:
            try:
                sl = self.strategy.calculate_stop_loss(data, entry_price)
                tp = self.strategy.calculate_take_profit(data, entry_price)
                if sl is not None: stop_loss = sl
                if tp is not None: take_profit = tp
            except Exception as e:
                self.logger.error(f"Error calculating SL/TP: {e}", exc_info=True)

        self.logger.info(f"BUY: {position_size} {self.symbol} at {entry_price} (Portfolio: ${self.portfolio_value:.2f})")
        self.performance_monitor.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': 'buy',
            'price': entry_price,
            'quantity': position_size,
            'pnl': 0.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })

    def execute_sell(self, price: float):
        exit_price = price * 0.999
        pnl = (exit_price - self.entry_price) * self.position
        self.portfolio_value += self.position * exit_price
        self.logger.info(f"SELL: {self.position} {self.symbol} at {exit_price} (PnL: ${pnl:.2f} | Portfolio: ${self.portfolio_value:.2f})")
        self.performance_monitor.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': 'sell',
            'price': exit_price,
            'quantity': self.position,
            'pnl': pnl
        })
        self.risk_manager.update_trade_stats(pnl)
        self.position = 0.0
        self.entry_price = 0.0

    def check_exit_conditions(self, current_price: float):
        if not self.performance_monitor.trades:
            return
        last_trade = self.performance_monitor.trades[-1]
        if 'stop_loss' in last_trade and last_trade['stop_loss'] is not None and current_price <= last_trade['stop_loss']:
            self.logger.info(f"Stop loss triggered at {current_price} (SL: {last_trade['stop_loss']})")
            self.execute_sell(current_price)
        elif 'take_profit' in last_trade and last_trade['take_profit'] is not None and current_price >= last_trade['take_profit']:
            self.logger.info(f"Take profit triggered at {current_price} (TP: {last_trade['take_profit']})")
            self.execute_sell(current_price)

    def _update_console_dashboard(self, history_data: pd.DataFrame, latest_candle: Dict[str, Any]):
        if not self.dashboard_manager or not self.dashboard_manager.is_running():
            return
        try:
            current_price = latest_candle['close']
            total_value = self.portfolio_value + (self.position * current_price if self.position > 0 else 0)
            self.dashboard_manager.update_portfolio_value(total_value)
            self.dashboard_manager.update_position(
                self.position,
                self.entry_price if self.position > 0 else 0.0,
                current_price
            )
            metrics = {
                'portfolio_value': round(total_value, 2),
                'position_size': round(self.position, 8),
                'entry_price': round(self.entry_price, 2) if self.position > 0 else 0.0,
                'current_price': round(current_price, 2),
                'unrealized_pnl': round((current_price - self.entry_price) * self.position if self.position > 0 else 0.0, 2),
                'unrealized_pnl_pct': round(((current_price / self.entry_price) - 1) * 100 if self.position > 0 and self.entry_price != 0 else 0.0, 2)
            }
            metrics.update(self.performance_monitor.get_metrics())
            self.dashboard_manager.update_metrics(metrics)
            strategy_signals = {self.strategy.__class__.__name__: {'buy': '?', 'sell': '?'}}
            self.dashboard_manager.update_strategy_signals(strategy_signals)
            self.dashboard_manager.update_candle(latest_candle)
            if not history_data.empty:
                chart_data = history_data.tail(60)
                prices = [{'time': row['date'].strftime('%H:%M'), 'price': row['close']} for _, row in chart_data.iterrows()]
                market_data = {'prices': prices}
                self.dashboard_manager.update_market_data(market_data)
        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}", exc_info=True)