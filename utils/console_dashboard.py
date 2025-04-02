"""
Console dashboard utilities for the ELVIS project.
This module provides functionality for creating and updating a real-time console dashboard
with ASCII art visualization, including real candlestick charts.
"""

import os
import time
import curses
import threading
import logging
import math
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

ELVIS_LOGO = r"""
 _______  _        __      __  _____   _____ 
|  ____| | |       \ \    / / |_   _| / ____|
| |__    | |        \ \  / /    | |  | (___  
|  __|   | |         \ \/ /     | |   \___ \ 
| |____  | |____      \  /     _| |_  ____) |
|______| |______|      \/     |_____||_____/ 
"""

class ConsoleDashboard:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.running = False
        self.thread = None
        self.stdscr = None
        
        self.portfolio_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.trades = []
        self.metrics = {}
        self.strategy_signals = {}
        self.market_data = {}
        
        self.current_candle = {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'closed': False}
        self.candle_history = []
        self.max_candle_history = 50
        
        self.is_testnet = True
        self.price_history = []
        self.max_price_history = 50
        
        self.market_stats = {
            'daily_high': 0.0, 'daily_low': 0.0, 'daily_volume': 0.0,
            'market_sentiment': 'Neutral', 'volatility': 0.0, 'trend': 'Sideways'
        }
        self.system_stats = {
            'cpu_usage': 0.0, 'memory_usage': 0.0, 'uptime': 0, 'api_calls': 0, 'last_error': None
        }
        
        self.lock = threading.Lock()
        self.view_mode = 'standard'

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Console dashboard thread started")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stdscr:
            try:
                curses.endwin()
            except Exception as e:
                self.logger.error(f"Error during curses cleanup: {e}")
        self.logger.info("Console dashboard stopped")

    def update_candle(self, candle: Dict[str, float]):
        with self.lock:
            self.current_candle = candle
            self.current_price = candle['close']
            if candle['closed']:
                self.candle_history.append(candle.copy())
                self.logger.debug(f"Added closed candle to history: O:{candle['open']:.2f} H:{candle['high']:.2f} L:{candle['low']:.2f} C:{candle['close']:.2f}")
                if len(self.candle_history) > self.max_candle_history:
                    self.candle_history = self.candle_history[-self.max_candle_history:]
            self.price_history.append(candle['close'])
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]
            self.logger.debug(f"Updated current candle: {self.current_candle}")

    def update_portfolio_value(self, value: float):
        with self.lock:
            self.portfolio_value = value

    def update_position(self, size: float, entry_price: float, current_price: float):
        with self.lock:
            self.position_size = size
            self.entry_price = entry_price
            self.current_price = current_price
            if size > 0 and entry_price > 0:
                self.unrealized_pnl = (current_price - entry_price) * size
                self.unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
            else:
                self.unrealized_pnl = 0.0
                self.unrealized_pnl_pct = 0.0

    def add_trade(self, trade: Dict[str, Any]):
        with self.lock:
            self.trades.append(trade)
            if len(self.trades) > 10:
                self.trades = self.trades[-10:]

    def update_metrics(self, metrics: Dict[str, Any]):
        with self.lock:
            self.metrics = metrics

    def update_strategy_signals(self, signals: Dict[str, Any]):
        with self.lock:
            self.strategy_signals = signals

    def update_market_data(self, market_data: Dict[str, Any]):
        with self.lock:
            self.market_data = market_data

    def set_testnet(self, is_testnet: bool):
        with self.lock:
            self.is_testnet = is_testnet

    def update_market_stats(self, stats: Dict[str, Any]):
        with self.lock:
            self.market_stats.update(stats)

    def update_system_stats(self, stats: Dict[str, Any]):
        with self.lock:
            self.system_stats.update(stats)

    def update_price_history(self, price: float):
        with self.lock:
            self.price_history.append(price)
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]
            self.current_price = price

    def _run_dashboard(self):
        try:
            self.logger.info("Attempting to initialize curses")
            self.stdscr = curses.initscr()
            if self.stdscr is None:
                raise RuntimeError("curses.initscr() returned None")
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            curses.init_pair(6, curses.COLOR_BLUE, -1)
            curses.init_pair(7, curses.COLOR_WHITE, -1)
            curses.curs_set(0)
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.timeout(100)
            self.logger.info("Curses initialized successfully")

            while self.running:
                max_y, max_x = self.stdscr.getmaxyx()
                if max_y < 10 or max_x < 50:
                    self.stdscr.clear()
                    self.stdscr.addstr(0, 0, "Terminal too small. Resize to at least 50x10.", curses.color_pair(2))
                    self.stdscr.refresh()
                    time.sleep(1)
                    continue
                
                self.stdscr.clear()
                if self.view_mode == 'detailed':
                    self._draw_detailed_view(max_y, max_x)
                elif self.view_mode == 'chart':
                    self._draw_chart_view(max_y, max_x)
                else:
                    self._draw_standard_view(max_y, max_x)
                self._draw_footer(max_y - 1, max_x)
                self.stdscr.refresh()
                
                key = self.stdscr.getch()
                if key == ord('q'):
                    self.logger.info("Quit key pressed")
                    break
                elif key == ord('1'):
                    self.view_mode = 'standard'
                elif key == ord('2'):
                    self.view_mode = 'detailed'
                elif key == ord('3'):
                    self.view_mode = 'chart'
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Critical error in console dashboard: {e}", exc_info=True)
            print(f"Error: Failed to launch dashboard - {e}. Check logs for details.")
        finally:
            if self.stdscr:
                try:
                    curses.nocbreak()
                    self.stdscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                except Exception as e:
                    self.logger.error(f"Error during curses cleanup: {e}")

    def _draw_standard_view(self, max_y: int, max_x: int):
        self._draw_logo_header(max_x)
        self._draw_portfolio_info(9, max_x)
        self._draw_metrics(15, max_x)
        self._draw_trades(22, max_x)
        self._draw_signals(22, max_x // 2 + 2, max_x)

    def _draw_detailed_view(self, max_y: int, max_x: int):
        self._draw_logo_header(max_x)
        self._draw_portfolio_info(9, max_x)
        self._draw_market_stats(15, max_x)
        self._draw_system_stats(15, max_x // 2 + 2, max_x)
        self._draw_candle_info(22, max_x)
        self._draw_trades(28, max_x)

    def _draw_chart_view(self, max_y: int, max_x: int):
        self._draw_header(max_x)
        self._draw_large_candle_chart(2, max_y - 2, max_x)

    def _draw_large_candle_chart(self, start_y: int, max_y: int, max_x: int):
        with self.lock:
            chart_height = max_y - start_y - 1
            chart_width = max_x - 12
            candle_width = 3
            num_candles = min(len(self.candle_history), chart_width // candle_width)

            if self.candle_history:
                min_low = min(c['low'] for c in self.candle_history[-num_candles:])
                max_high = max(c['high'] for c in self.candle_history[-num_candles:])
                price_range = max(max_high - min_low, 1.0)

                for i in range(0, chart_height, 4):
                    price_level = max_high - (i / (chart_height - 1)) * price_range
                    try:
                        self.stdscr.addstr(start_y + i, 2, f"${price_level:.0f}", curses.color_pair(7))
                        self.stdscr.addstr(start_y + i, 10, "┈" * (chart_width - 10), curses.color_pair(6))
                    except curses.error:
                        pass

                for i in range(num_candles):
                    candle = self.candle_history[-(num_candles - i)]
                    x_pos = 10 + i * candle_width
                    
                    high_y = int((max_high - candle['high']) / price_range * (chart_height - 1))
                    low_y = int((max_high - candle['low']) / price_range * (chart_height - 1))
                    open_y = int((max_high - candle['open']) / price_range * (chart_height - 1))
                    close_y = int((max_high - candle['close']) / price_range * (chart_height - 1))
                    
                    high_y = max(0, min(high_y, chart_height - 1))
                    low_y = max(0, min(low_y, chart_height - 1))
                    open_y = max(0, min(open_y, chart_height - 1))
                    close_y = max(0, min(close_y, chart_height - 1))
                    
                    color = curses.color_pair(1) if candle['close'] >= candle['open'] else curses.color_pair(2)
                    
                    for y in range(high_y, low_y + 1):
                        try:
                            self.stdscr.addstr(start_y + y, x_pos + 1, "│", color)
                        except curses.error:
                            pass
                    
                    body_top = min(open_y, close_y)
                    body_bottom = max(open_y, close_y)
                    for y in range(body_top, body_bottom + 1):
                        try:
                            self.stdscr.addstr(start_y + y, x_pos, "█", color)
                            self.stdscr.addstr(start_y + y, x_pos + 1, "█", color)
                            self.stdscr.addstr(start_y + y, x_pos + 2, "█", color)
                        except curses.error:
                            pass

                if self.current_candle['close'] > 0:
                    current_y = int((max_high - self.current_candle['close']) / price_range * (chart_height - 1))
                    current_y = max(0, min(current_y, chart_height - 1))
                    try:
                        self.stdscr.addstr(start_y + current_y, 10, "─" * (chart_width - 10), curses.color_pair(4) | curses.A_BOLD)
                        self.stdscr.addstr(start_y + current_y, max_x - 10, f"${self.current_candle['close']:.2f}", curses.color_pair(4) | curses.A_BOLD)
                    except curses.error:
                        pass
            else:
                try:
                    self.stdscr.addstr(start_y + chart_height // 2, max_x // 2 - 10, "No candle data available", curses.color_pair(7))
                    self.logger.warning("No candle data in history. Check WebSocket connection.")
                except curses.error:
                    pass

    def _draw_candle_info(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, 0, "│ ")
                self.stdscr.addstr("Current Candle", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - 16)
                self.stdscr.addstr(" " * padding + "│")
                
                candle = self.current_candle
                self.stdscr.addstr(start_y + 1, 2, f"Open: ${candle['open']:.2f}")
                self.stdscr.addstr(start_y + 2, 2, f"High: ${candle['high']:.2f}", curses.color_pair(1))
                self.stdscr.addstr(start_y + 3, 2, f"Low: ${candle['low']:.2f}", curses.color_pair(2))
                self.stdscr.addstr(start_y + 4, 2, f"Close: ${candle['close']:.2f}")
                self.stdscr.addstr(start_y + 5, 2, f"Volume: {candle['volume']:.4f}")
                
                for i in range(1, 6):
                    self.stdscr.addstr(start_y + i, 0, "│")
                    self.stdscr.addstr(start_y + i, max_x - 1, "│")
                self.stdscr.addstr(start_y + 6, 0, "├")
                self.stdscr.addstr(start_y + 6, 1, "─" * (max_x - 2))
                self.stdscr.addstr(start_y + 6, max_x - 1, "┤")
            except Exception as e:
                self.logger.error(f"Error drawing candle info: {e}")

    def _draw_portfolio_info(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, 0, "│ ")
                self.stdscr.addstr("Portfolio Information", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - 23)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing portfolio header: {e}")
            
            self.stdscr.addstr(start_y + 1, 2, "Portfolio Value: ", curses.A_BOLD)
            value_color = curses.color_pair(1) if self.portfolio_value > 10000 else curses.A_NORMAL
            self.stdscr.addstr(f"${self.portfolio_value:.2f}", value_color)
            
            if self.position_size > 0:
                self.stdscr.addstr(start_y + 2, 2, "Position Size: ", curses.A_BOLD)
                self.stdscr.addstr(f"{self.position_size:.8f} BTC")
                self.stdscr.addstr(start_y + 3, 2, "Entry Price: ", curses.A_BOLD)
                self.stdscr.addstr(f"${self.entry_price:.2f}")
                self.stdscr.addstr(start_y + 4, 2, "Current Price: ", curses.A_BOLD)
                price_color = curses.color_pair(1) if self.current_candle['close'] > self.entry_price else curses.color_pair(2)
                self.stdscr.addstr(f"${self.current_candle['close']:.2f}", price_color)
                
                pnl_color = curses.color_pair(1) if self.unrealized_pnl >= 0 else curses.color_pair(2)
                self.stdscr.addstr(start_y + 2, max_x // 2 - 10, "Unrealized PnL: ", curses.A_BOLD)
                self.stdscr.addstr(f"${self.unrealized_pnl:.2f}", pnl_color)
                self.stdscr.addstr(start_y + 3, max_x // 2 - 10, "Unrealized PnL %: ", curses.A_BOLD)
                self.stdscr.addstr(f"{self.unrealized_pnl_pct:.2f}%", pnl_color)
                leverage = random.randint(1, 10)
                self.stdscr.addstr(start_y + 4, max_x // 2 - 10, "Leverage: ", curses.A_BOLD)
                self.stdscr.addstr(f"{leverage}x", curses.color_pair(4))
            else:
                self.stdscr.addstr(start_y + 2, 2, "No open position", curses.color_pair(4))
            
            for i in range(1, 5):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 5, 0, "├")
            self.stdscr.addstr(start_y + 5, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 5, max_x - 1, "┤")

    def _draw_logo_header(self, max_x: int):
        try:
            self.stdscr.addstr(0, 0, "┌")
            self.stdscr.addstr(0, 1, "─" * (max_x - 2))
            self.stdscr.addstr(0, max_x - 1, "┐")
            logo_lines = ELVIS_LOGO.strip().split('\n')
            for i, line in enumerate(logo_lines):
                if i + 1 < 8:
                    line_pos = max(0, min((max_x - len(line)) // 2, max_x - len(line) - 1))
                    self.stdscr.addstr(i + 1, line_pos, line, curses.color_pair(5) | curses.A_BOLD)
            time_str = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if len(time_str) + 2 < max_x:
                self.stdscr.addstr(7, 2, time_str)
            env_str = "TESTNET MODE" if self.is_testnet else "PRODUCTION MODE"
            env_color = curses.color_pair(4) if self.is_testnet else curses.color_pair(2) | curses.A_BOLD
            if len(env_str) + 2 < max_x:
                self.stdscr.addstr(7, max(0, min(max_x - len(env_str) - 2, max_x - len(env_str) - 1)), env_str, env_color)
            self.stdscr.addstr(8, 0, "├")
            self.stdscr.addstr(8, 1, "─" * (max_x - 2))
            self.stdscr.addstr(8, max_x - 1, "┤")
            for i in range(1, 8):
                self.stdscr.addstr(i, 0, "│")
                self.stdscr.addstr(i, max_x - 1, "│")
        except Exception as e:
            self.logger.error(f"Error drawing logo header: {e}")

    def _draw_header(self, max_x: int):
        try:
            self.stdscr.addstr(0, 0, "┌")
            self.stdscr.addstr(0, 1, "─" * (max_x - 2))
            self.stdscr.addstr(0, max_x - 1, "┐")
            title = "ELVIS Console Dashboard - Chart View"
            title_pos = max(0, min((max_x - len(title)) // 2, max_x - len(title) - 1))
            self.stdscr.addstr(0, title_pos, title, curses.A_BOLD | curses.color_pair(3))
            time_str = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if len(time_str) + 2 < max_x:
                self.stdscr.addstr(0, max(0, min(max_x - len(time_str) - 2, max_x - len(time_str) - 1)), time_str)
            self.stdscr.addstr(1, 0, "├")
            self.stdscr.addstr(1, 1, "─" * (max_x - 2))
            self.stdscr.addstr(1, max_x - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing header: {e}")

    def _draw_metrics(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, 0, "│ ")
                self.stdscr.addstr("Performance Metrics", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - 21)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing metrics header: {e}")
            if self.metrics:
                col = 0
                row = 0
                col_width = max_x // 3
                for i, (key, value) in enumerate(self.metrics.items()):
                    key_str = key.replace('_', ' ').title() + ":"
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    color = curses.A_NORMAL
                    if isinstance(value, float) and (key.endswith('_pct') or key.endswith('_rate') or key == 'win_rate'):
                        color = curses.color_pair(1) if value >= 0 else curses.color_pair(2)
                    self.stdscr.addstr(start_y + 1 + row, 2 + col * col_width, key_str, curses.A_BOLD)
                    self.stdscr.addstr(start_y + 1 + row, 2 + col * col_width + len(key_str) + 1, value_str, color)
                    col += 1
                    if col >= 3:
                        col = 0
                        row += 1
            else:
                self.stdscr.addstr(start_y + 1, 2, "No metrics available")
            for i in range(1, 5):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 5, 0, "├")
            self.stdscr.addstr(start_y + 5, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 5, max_x - 1, "┤")

    def _draw_market_stats(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, 0, "│ ")
                self.stdscr.addstr("Market Statistics", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x // 2 - 19)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing market stats header: {e}")
            row = 0
            for key, value in self.market_stats.items():
                key_str = key.replace('_', ' ').title() + ":"
                color = curses.A_NORMAL
                if key == 'market_sentiment':
                    color = {'Bullish': 1, 'Bearish': 2, 'Neutral': 4}.get(value, 4)
                elif key == 'trend':
                    color = {'Uptrend': 1, 'Downtrend': 2, 'Sideways': 4}.get(value, 4)
                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                self.stdscr.addstr(start_y + 1 + row, 2, key_str, curses.A_BOLD)
                self.stdscr.addstr(start_y + 1 + row, 2 + len(key_str) + 1, value_str, curses.color_pair(color))
                row += 1
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x // 2, "│")

    def _draw_system_stats(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, max_x // 2 + 2, "│ ")
                self.stdscr.addstr("System Statistics", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - (max_x // 2 + 2) - 20)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing system stats header: {e}")
            row = 0
            for key, value in self.system_stats.items():
                key_str = key.replace('_', ' ').title() + ":"
                color = curses.A_NORMAL
                if key in ['cpu_usage', 'memory_usage']:
                    if isinstance(value, float):
                        color = 2 if value > 80 else 4 if value > 50 else 1
                        value_str = f"{value:.1f}%"
                elif key == 'uptime':
                    hours, remainder = divmod(value, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    value_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                elif key == 'last_error':
                    value_str = str(value) if value else "None"
                    color = 2 if value else 1
                else:
                    value_str = str(value)
                self.stdscr.addstr(start_y + 1 + row, max_x // 2 + 4, key_str, curses.A_BOLD)
                self.stdscr.addstr(start_y + 1 + row, max_x // 2 + 4 + len(key_str) + 1, value_str, curses.color_pair(color))
                row += 1
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, max_x // 2 + 2, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 6, 0, "├")
            self.stdscr.addstr(start_y + 6, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 6, max_x - 1, "┤")

    def _draw_trades(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, 0, "│ ")
                self.stdscr.addstr("Recent Trades", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - 15)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing trades header: {e}")
            if self.trades:
                self.stdscr.addstr(start_y + 1, 2, "Time", curses.A_BOLD)
                self.stdscr.addstr(start_y + 1, 15, "Symbol", curses.A_BOLD)
                self.stdscr.addstr(start_y + 1, 25, "Side", curses.A_BOLD)
                self.stdscr.addstr(start_y + 1, 32, "Price", curses.A_BOLD)
                self.stdscr.addstr(start_y + 1, 45, "Quantity", curses.A_BOLD)
                self.stdscr.addstr(start_y + 1, 60, "PnL", curses.A_BOLD)
                for i, trade in enumerate(reversed(self.trades[:5])):
                    time_str = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S') if 'timestamp' in trade else "N/A"
                    side_str = trade.get('side', 'N/A')
                    side_color = curses.color_pair(1) if side_str.lower() == 'buy' else curses.color_pair(2)
                    price_str = f"${trade.get('price', 0.0):.2f}"
                    quantity_str = f"{trade.get('quantity', 0.0):.8f}"
                    pnl = trade.get('pnl', 0.0)
                    pnl_str = f"${pnl:.2f}"
                    pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                    self.stdscr.addstr(start_y + 2 + i, 2, time_str)
                    self.stdscr.addstr(start_y + 2 + i, 15, trade.get('symbol', 'N/A'))
                    self.stdscr.addstr(start_y + 2 + i, 25, side_str, side_color)
                    self.stdscr.addstr(start_y + 2 + i, 32, price_str)
                    self.stdscr.addstr(start_y + 2 + i, 45, quantity_str)
                    self.stdscr.addstr(start_y + 2 + i, 60, pnl_str, pnl_color)
            else:
                self.stdscr.addstr(start_y + 1, 2, "No trades available")

    def _draw_signals(self, start_y: int, max_x: int):
        with self.lock:
            try:
                self.stdscr.addstr(start_y, max_x // 2 + 2, "│ ")
                self.stdscr.addstr("Strategy Signals", curses.A_BOLD | curses.color_pair(3))
                padding = max(0, max_x - (max_x // 2 + 2) - 18)
                self.stdscr.addstr(" " * padding + "│")
            except Exception as e:
                self.logger.error(f"Error drawing signals header: {e}")
            if self.strategy_signals:
                for i, (strategy, signal) in enumerate(self.strategy_signals.items()):
                    strategy_str = strategy.replace('_', ' ').title() + ":"
                    if signal.get('buy', False):
                        signal_str, signal_color = "BUY", curses.color_pair(1)
                    elif signal.get('sell', False):
                        signal_str, signal_color = "SELL", curses.color_pair(2)
                    else:
                        signal_str, signal_color = "HOLD", curses.A_NORMAL
                    self.stdscr.addstr(start_y + 1 + i, max_x // 2 + 4, strategy_str, curses.A_BOLD)
                    self.stdscr.addstr(start_y + 1 + i, max_x // 2 + 4 + len(strategy_str) + 1, signal_str, signal_color)
                    self.stdscr.addstr(start_y + 1 + i, max_x // 2 + 2, "|")
                    self.stdscr.addstr(start_y + 1 + i, max_x - 1, "|")
            else:
                self.stdscr.addstr(start_y + 1, max_x // 2 + 4, "No signals available")
                self.stdscr.addstr(start_y + 1, max_x // 2 + 2, "|")
                self.stdscr.addstr(start_y + 1, max_x - 1, "|")

    def _draw_footer(self, y: int, max_x: int):
        try:
            if y >= 0:
                self.stdscr.addstr(y, 0, "└")
                self.stdscr.addstr(y, 1, "─" * (max_x - 2))
                self.stdscr.addstr(y, max_x - 1, "┘")
                commands = "1: Standard | 2: Detailed | 3: Candlestick Chart | q: Quit"
                cmd_pos = max(0, min((max_x - len(commands)) // 2, max_x - len(commands) - 1))
                self.stdscr.addstr(y, cmd_pos, commands, curses.A_BOLD | curses.color_pair(4))
            else:
                self.logger.warning("Footer y-position out of bounds. Terminal too small?")
        except Exception as e:
            self.logger.error(f"Error drawing footer: {e}")

class ConsoleDashboardManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dashboard = None

    def start_dashboard(self):
        self.logger.info("Starting console dashboard")
        self.dashboard = ConsoleDashboard(self.logger)
        self.dashboard.start()

    def stop_dashboard(self):
        if self.dashboard:
            self.logger.info("Stopping console dashboard")
            self.dashboard.stop()

    def update_candle(self, candle: Dict[str, float]):
        if self.dashboard:
            self.dashboard.update_candle(candle)

    def add_trade(self, trade: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.add_trade(trade)

    def update_portfolio_value(self, value: float):
        if self.dashboard:
            self.dashboard.update_portfolio_value(value)

    def update_position(self, size: float, entry_price: float, current_price: float):
        if self.dashboard:
            self.dashboard.update_position(size, entry_price, current_price)

    def update_metrics(self, metrics: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.update_metrics(metrics)

    def update_strategy_signals(self, signals: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.update_strategy_signals(signals)

    def update_market_data(self, market_data: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.update_market_data(market_data)

    def set_testnet(self, is_testnet: bool):
        if self.dashboard:
            self.dashboard.set_testnet(is_testnet)

    def update_market_stats(self, stats: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.update_market_stats(stats)

    def update_system_stats(self, stats: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.update_system_stats(stats)