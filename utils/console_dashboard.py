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
        self.stdscr = None
        
        self.portfolio_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.realised_pnl = 0.0  # Added to track realised PnL
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
        """Start the dashboard in the main thread."""
        if self.running:
            return
        self.running = True
        self.logger.info("Starting dashboard in main thread")
        self._run_dashboard()

    def stop(self):
        """Stop the dashboard and clean up curses."""
        if not self.running:
            return
        self.running = False
        if self.stdscr:
            try:
                curses.endwin()
            except Exception as e:
                self.logger.error(f"Error during curses cleanup: {e}")
        self.logger.info("Console dashboard stopped")

    def update_candle(self, candle: Dict[str, float]):
        """Update the current candle and candle history."""
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
            # If no closed candles after 10 seconds, generate mock data
            if not self.candle_history and time.time() - self.start_time > 10:
                self._generate_mock_candle_data()

    def update_portfolio_value(self, value: float):
        """Update the portfolio value."""
        with self.lock:
            self.portfolio_value = value

    def update_position(self, size: float, entry_price: float, current_price: float):
        """Update the current position and calculate unrealized PnL."""
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
        """Add a trade to the trade history and update realised PnL."""
        with self.lock:
            self.trades.append(trade)
            if len(self.trades) > 10:
                self.trades = self.trades[-10:]
            # Update realised PnL when a trade is closed (sell)
            if trade.get('side', '').upper() == 'SELL' and trade.get('pnl', 0.0) != 0.0:
                self.realised_pnl += trade['pnl']
                self.logger.debug(f"Updated realised PnL: ${self.realised_pnl:.2f}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        with self.lock:
            self.metrics = metrics

    def update_strategy_signals(self, signals: Dict[str, Any]):
        """Update strategy signals."""
        with self.lock:
            self.strategy_signals = signals

    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data."""
        with self.lock:
            self.market_data = market_data

    def set_testnet(self, is_testnet: bool):
        """Set whether we're in testnet mode."""
        with self.lock:
            self.is_testnet = is_testnet

    def update_market_stats(self, stats: Dict[str, Any]):
        """Update market statistics."""
        with self.lock:
            self.market_stats.update(stats)

    def update_system_stats(self, stats: Dict[str, Any]):
        """Update system statistics."""
        with self.lock:
            self.system_stats.update(stats)

    def update_price_history(self, price: float):
        """Update price history with a new price point."""
        with self.lock:
            self.price_history.append(price)
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]
            self.current_price = price

    def _run_dashboard(self):
        """Main dashboard loop."""
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
            self.start_time = time.time()  # Track start time for mock data trigger

            while self.running:
                max_y, max_x = self.stdscr.getmaxyx()
                self.logger.debug(f"Terminal size: {max_y}x{max_x}")
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
                
                # Draw command info below the logo
                self._draw_command_info(8, max_x)
                
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
            raise
        finally:
            if self.stdscr:
                try:
                    curses.nocbreak()
                    self.stdscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                except Exception as e:
                    self.logger.error(f"Error during curses cleanup: {e}")

    def _draw_logo_header(self, max_x: int):
        """Draw the ELVIS logo header."""
        try:
            logo_lines = ELVIS_LOGO.strip().split('\n')
            for i, line in enumerate(logo_lines):
                self.stdscr.addstr(i, (max_x - len(line)) // 2, line, curses.color_pair(5) | curses.A_BOLD)
            
            # Draw environment indicator
            env_text = "TESTNET" if self.is_testnet else "PRODUCTION"
            env_color = curses.color_pair(4) if self.is_testnet else curses.color_pair(2)
            self.stdscr.addstr(1, max_x - len(env_text) - 2, env_text, env_color | curses.A_BOLD)
            
            # Draw border
            self.stdscr.addstr(len(logo_lines), 0, "┌")
            self.stdscr.addstr(len(logo_lines), 1, "─" * (max_x - 2))
            self.stdscr.addstr(len(logo_lines), max_x - 1, "┐")
        except Exception as e:
            self.logger.error(f"Error drawing logo header: {e}")

    def _draw_header(self, max_x: int):
        """Draw a simple header for chart view."""
        try:
            header_text = "ELVIS Trading Bot - BTC/USD Chart"
            self.stdscr.addstr(0, (max_x - len(header_text)) // 2, header_text, curses.color_pair(3) | curses.A_BOLD)
            
            # Draw environment indicator
            env_text = "TESTNET" if self.is_testnet else "PRODUCTION"
            env_color = curses.color_pair(4) if self.is_testnet else curses.color_pair(2)
            self.stdscr.addstr(0, max_x - len(env_text) - 2, env_text, env_color | curses.A_BOLD)
            
            # Draw border
            self.stdscr.addstr(1, 0, "┌")
            self.stdscr.addstr(1, 1, "─" * (max_x - 2))
            self.stdscr.addstr(1, max_x - 1, "┐")
        except Exception as e:
            self.logger.error(f"Error drawing header: {e}")

    def _draw_portfolio_info(self, start_y: int, max_x: int):
        """Draw portfolio information including realised PnL."""
        try:
            self.stdscr.addstr(start_y, 0, "│ ")
            self.stdscr.addstr("Portfolio", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x - 11)
            self.stdscr.addstr(" " * padding + "│")
            
            # Draw portfolio value
            self.stdscr.addstr(start_y + 1, 2, f"Portfolio Value: ${self.portfolio_value:.2f}")
            
            # Draw position info
            if self.position_size > 0:
                self.stdscr.addstr(start_y + 2, 2, f"Position: {self.position_size:.8f} BTC @ ${self.entry_price:.2f}")
                
                # Draw Unrealised PnL
                pnl_color = curses.color_pair(1) if self.unrealized_pnl >= 0 else curses.color_pair(2)
                pnl_sign = "+" if self.unrealized_pnl >= 0 else ""
                self.stdscr.addstr(start_y + 3, 2, f"Unrealized PnL: {pnl_sign}${self.unrealized_pnl:.2f} ({pnl_sign}{self.unrealized_pnl_pct:.2f}%)", pnl_color)
            else:
                self.stdscr.addstr(start_y + 2, 2, "Position: No open position", curses.color_pair(4))
                self.stdscr.addstr(start_y + 3, 2, "Unrealized PnL: $0.00 (0.00%)")
            
            # Draw current price and realised PnL
            self.stdscr.addstr(start_y + 4, 2, f"Current Price: ${self.current_price:.2f}")
            realised_color = curses.color_pair(1) if self.realised_pnl >= 0 else curses.color_pair(2)
            realised_sign = "+" if self.realised_pnl >= 0 else ""
            self.stdscr.addstr(start_y + 5, 2, f"Realised PnL: {realised_sign}${self.realised_pnl:.2f}", realised_color)
            
            # Draw borders (extended to accommodate extra line)
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 6, 0, "├")
            self.stdscr.addstr(start_y + 6, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 6, max_x - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing portfolio info: {e}")

    def _draw_command_info(self, start_y: int, max_x: int):
        """Draw command information below the logo."""
        try:
            commands = "Commands: [1] Standard View  [2] Detailed View  [3] Chart View  [q] Quit"
            self.stdscr.addstr(start_y, (max_x - len(commands)) // 2, commands, curses.color_pair(7))
        except Exception as e:
            self.logger.error(f"Error drawing command info: {e}")

    def _draw_standard_view(self, max_y: int, max_x: int):
        """Draw the standard view."""
        self._draw_logo_header(max_x)
        self._draw_portfolio_info(9, max_x)
        self._draw_metrics_simple(16, max_x)  # Adjusted start_y due to extra line in portfolio
        self._draw_trades(23, max_x)  # Adjusted start_y
        self._draw_signals(23, max_x // 2 + 2, max_x)  # Adjusted start_y

    def _draw_detailed_view(self, max_y: int, max_x: int):
        """Draw the detailed view."""
        try:
            self._draw_logo_header(max_x)
            self._draw_portfolio_info(9, max_x)
            
            # Only draw these sections if we have enough vertical space
            if max_y > 21:  # Adjusted for extra line
                self._draw_market_stats(16, max_x)  # Adjusted start_y
                self._draw_system_stats(16, max_x // 2 + 2, max_x)  # Adjusted start_y
                
                if max_y > 29:  # Adjusted
                    self._draw_candle_info(23, max_x)  # Adjusted start_y
                    
                    if max_y > 35:  # Adjusted
                        self._draw_trades(30, max_x)  # Adjusted start_y
            else:
                # Simplified view for smaller terminals
                self.stdscr.addstr(16, 2, "Terminal too small for detailed view", curses.color_pair(4))
        except Exception as e:
            self.logger.error(f"Error in detailed view: {e}")
            # Fallback to a simple message if detailed view fails
            try:
                self.stdscr.addstr(10, 2, "Error displaying detailed view", curses.color_pair(2))
                self.stdscr.addstr(11, 2, f"Error: {str(e)[:50]}", curses.color_pair(2))
            except:
                pass  # If even the error display fails, just continue

    def _draw_chart_view(self, max_y: int, max_x: int):
        """Draw the chart view."""
        try:
            self._draw_header(max_x)
            self._draw_large_candle_chart(2, max_y - 2, max_x)
        except Exception as e:
            self.logger.error(f"Error in chart view: {e}")
            try:
                self.stdscr.addstr(max_y // 2, 2, "Error displaying chart view", curses.color_pair(2))
                self.stdscr.addstr(max_y // 2 + 1, 2, f"Error: {str(e)[:50]}", curses.color_pair(2))
            except:
                pass  # Silent fail if error message can't be displayed

    def _draw_metrics_simple(self, start_y: int, max_x: int):
        """Draw performance metrics in a simple format."""
        try:
            self.stdscr.addstr(start_y, 0, "│ ")
            self.stdscr.addstr("Performance Metrics", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x - 21)
            self.stdscr.addstr(" " * padding + "│")
            
            # Draw some basic metrics
            if self.metrics:
                row = 0
                for i, (key, value) in enumerate(list(self.metrics.items())[:8]):
                    if i % 2 == 0:
                        x_pos = 2
                    else:
                        x_pos = max_x // 2
                        row += 1
                    
                    key_str = key.replace('_', ' ').title() + ":"
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    self.stdscr.addstr(start_y + 1 + row, x_pos, f"{key_str} {value_str}")
            else:
                self.stdscr.addstr(start_y + 1, 2, "No metrics available", curses.color_pair(4))
            
            # Draw borders
            for i in range(1, 5):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 5, 0, "├")
            self.stdscr.addstr(start_y + 5, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 5, max_x - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing metrics: {e}")

    def _draw_trades(self, start_y: int, max_x: int):
        """Draw recent trades."""
        try:
            self.stdscr.addstr(start_y, 0, "│ ")
            self.stdscr.addstr("Recent Trades", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x // 2 - 14)
            self.stdscr.addstr(" " * padding + "│")
            
            if self.trades:
                for i, trade in enumerate(self.trades[-5:]):
                    if i >= 5:  # Limit to 5 trades
                        break
                    
                    trade_type = trade.get('side', 'UNKNOWN')
                    price = trade.get('price', 0.0)
                    size = trade.get('quantity', 0.0)
                    timestamp = trade.get('timestamp', datetime.now().isoformat())
                    
                    try:
                        # Parse timestamp if it's a string
                        timestamp_str = datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
                    except (ValueError, TypeError):
                        timestamp_str = str(timestamp)
                    
                    color = curses.color_pair(1) if trade_type.upper() == 'BUY' else curses.color_pair(2)
                    self.stdscr.addstr(start_y + i + 1, 2, f"{trade_type}: {size:.8f} @ ${price:.2f} - {timestamp_str}", color)
            else:
                self.stdscr.addstr(start_y + 1, 2, "No trades yet", curses.color_pair(4))
            
            # Draw borders
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x // 2 - 1, "│")
            self.stdscr.addstr(start_y + 6, 0, "└")
            self.stdscr.addstr(start_y + 6, 1, "─" * (max_x // 2 - 2))
            self.stdscr.addstr(start_y + 6, max_x // 2 - 1, "┘")
        except Exception as e:
            self.logger.error(f"Error drawing trades: {e}")

    def _draw_signals(self, start_y: int, start_x: int, max_x: int):
        """Draw strategy signals."""
        try:
            self.stdscr.addstr(start_y, start_x, "│ ")
            self.stdscr.addstr("Strategy Signals", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x - start_x - 18)
            self.stdscr.addstr(" " * padding + "│")
            
            if self.strategy_signals:
                for i, (strategy, signal) in enumerate(self.strategy_signals.items()):
                    if i >= 5:  # Limit to 5 signals
                        break
                    
                    strategy_str = strategy.replace('_', ' ').title()
                    if signal.get('buy', False):
                        signal_text = "BUY"
                        color = curses.color_pair(1)
                    elif signal.get('sell', False):
                        signal_text = "SELL"
                        color = curses.color_pair(2)
                    else:
                        signal_text = "HOLD"
                        color = curses.color_pair(4)
                    
                    self.stdscr.addstr(start_y + i + 1, start_x + 2, f"{strategy_str}: {signal_text}", color)
            else:
                self.stdscr.addstr(start_y + 1, start_x + 2, "No signals available", curses.color_pair(4))
            
            # Draw borders
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, start_x, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 6, start_x, "└")
            self.stdscr.addstr(start_y + 6, start_x + 1, "─" * (max_x - start_x - 2))
            self.stdscr.addstr(start_y + 6, max_x - 1, "┘")
        except Exception as e:
            self.logger.error(f"Error drawing signals: {e}")

    def _draw_market_stats(self, start_y: int, max_x: int):
        """Draw market statistics."""
        try:
            self.stdscr.addstr(start_y, 0, "│ ")
            self.stdscr.addstr("Market Statistics", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x // 2 - 19)
            self.stdscr.addstr(" " * padding + "│")
            
            self.stdscr.addstr(start_y + 1, 2, f"Daily High: ${self.market_stats['daily_high']:.2f}", curses.color_pair(1))
            self.stdscr.addstr(start_y + 2, 2, f"Daily Low: ${self.market_stats['daily_low']:.2f}", curses.color_pair(2))
            self.stdscr.addstr(start_y + 3, 2, f"Daily Volume: {self.market_stats['daily_volume']:.2f}")
            self.stdscr.addstr(start_y + 4, 2, f"Trend: {self.market_stats['trend']}")
            
            # Draw borders
            for i in range(1, 5):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x // 2 - 1, "│")
            self.stdscr.addstr(start_y + 5, 0, "├")
            self.stdscr.addstr(start_y + 5, 1, "─" * (max_x // 2 - 2))
            self.stdscr.addstr(start_y + 5, max_x // 2 - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing market stats: {e}")

    def _draw_system_stats(self, start_y: int, start_x: int, max_x: int):
        """Draw system statistics."""
        try:
            self.stdscr.addstr(start_y, start_x, "│ ")
            self.stdscr.addstr("System Statistics", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x - start_x - 19)
            self.stdscr.addstr(" " * padding + "│")
            
            self.stdscr.addstr(start_y + 1, start_x + 2, f"CPU Usage: {self.system_stats['cpu_usage']:.1f}%")
            self.stdscr.addstr(start_y + 2, start_x + 2, f"Memory Usage: {self.system_stats['memory_usage']:.1f}%")
            self.stdscr.addstr(start_y + 3, start_x + 2, f"API Calls: {self.system_stats['api_calls']}")
            self.stdscr.addstr(start_y + 4, start_x + 2, f"Uptime: {self.system_stats['uptime']} sec")
            
            # Draw borders
            for i in range(1, 5):
                self.stdscr.addstr(start_y + i, start_x, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 5, start_x, "├")
            self.stdscr.addstr(start_y + 5, start_x + 1, "─" * (max_x - start_x - 2))
            self.stdscr.addstr(start_y + 5, max_x - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing system stats: {e}")

    def _generate_mock_candle_data(self):
        """Generate mock candle data for visualization when real data is not available."""
        if self.current_price <= 0:
            self.current_price = 75655.0  # Default BTC price if none available
        
        base_price = self.current_price
        self.logger.info(f"Generating mock candle data with base price: ${base_price:.2f}")
        
        # Generate 50 candles with realistic price movements
        for i in range(50):
            # Create price movement with some randomness but trending slightly upward
            price_change_pct = random.uniform(-0.015, 0.02)  # -1.5% to +2% change
            close_price = base_price * (1 + price_change_pct)
            
            # Create realistic OHLC values
            high_price = close_price * (1 + random.uniform(0.001, 0.01))
            low_price = close_price * (1 - random.uniform(0.001, 0.01))
            open_price = close_price * (1 + random.uniform(-0.008, 0.008))
            
            # Ensure high is highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Create candle
            candle = {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(10, 100),
                'closed': True
            }
            
            self.candle_history.append(candle)
            base_price = close_price  # Use close as base for next candle
        
        # Update current candle
        self.current_candle = {
            'open': self.candle_history[-1]['close'],
            'high': self.candle_history[-1]['close'] * (1 + random.uniform(0.001, 0.005)),
            'low': self.candle_history[-1]['close'] * (1 - random.uniform(0.001, 0.005)),
            'close': self.candle_history[-1]['close'] * (1 + random.uniform(-0.003, 0.003)),
            'volume': random.uniform(5, 50),
            'closed': False
        }
        
        # Ensure high is highest and low is lowest in current candle
        self.current_candle['high'] = max(self.current_candle['high'], self.current_candle['open'], self.current_candle['close'])
        self.current_candle['low'] = min(self.current_candle['low'], self.current_candle['open'], self.current_candle['close'])
        
        self.logger.info(f"Generated {len(self.candle_history)} mock candles")

    def _draw_candle_info(self, start_y: int, max_x: int):
        """Draw current candle information."""
        try:
            self.stdscr.addstr(start_y, 0, "│ ")
            self.stdscr.addstr("Current Candle", curses.A_BOLD | curses.color_pair(3))
            padding = max(0, max_x - 16)
            self.stdscr.addstr(" " * padding + "│")
            
            with self.lock:
                candle = self.current_candle
                self.stdscr.addstr(start_y + 1, 2, f"Open: ${candle['open']:.2f}")
                self.stdscr.addstr(start_y + 2, 2, f"High: ${candle['high']:.2f}", curses.color_pair(1))
                self.stdscr.addstr(start_y + 3, 2, f"Low: ${candle['low']:.2f}", curses.color_pair(2))
                self.stdscr.addstr(start_y + 4, 2, f"Close: ${candle['close']:.2f}")
                self.stdscr.addstr(start_y + 5, 2, f"Volume: {candle['volume']:.4f}")
            
            # Draw borders
            for i in range(1, 6):
                self.stdscr.addstr(start_y + i, 0, "│")
                self.stdscr.addstr(start_y + i, max_x - 1, "│")
            self.stdscr.addstr(start_y + 6, 0, "├")
            self.stdscr.addstr(start_y + 6, 1, "─" * (max_x - 2))
            self.stdscr.addstr(start_y + 6, max_x - 1, "┤")
        except Exception as e:
            self.logger.error(f"Error drawing candle info: {e}")

    def _draw_large_candle_chart(self, start_y: int, max_y: int, max_x: int):
        """Draw a large candlestick chart."""
        try:
            with self.lock:
                chart_height = max_y - start_y - 1
                chart_width = max_x - 12  # Leave space for price labels
                candle_width = 3
                num_candles = min(len(self.candle_history), chart_width // candle_width)

                if not self.candle_history:
                    self.stdscr.addstr(start_y + chart_height // 2, max_x // 2 - 10, "No candle data available", curses.color_pair(7))
                    return

                min_low = min(c['low'] for c in self.candle_history[-num_candles:])
                max_high = max(c['high'] for c in self.candle_history[-num_candles:])
                price_range = max(max_high - min_low, 1.0)

                # Draw price levels
                for i in range(0, chart_height, 4):
                    price_level = max_high - (i / (chart_height - 1)) * price_range
                    try:
                        self.stdscr.addstr(start_y + i, 2, f"${price_level:.0f}", curses.color_pair(7))
                        self.stdscr.addstr(start_y + i, 10, "┈" * (chart_width - 10), curses.color_pair(6))
                    except curses.error:
                        pass

                # Draw candlesticks
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
                    
                    # Draw wick
                    for y in range(high_y, low_y + 1):
                        try:
                            self.stdscr.addstr(start_y + y, x_pos + 1, "│", color)
                        except curses.error:
                            pass
                    
                    # Draw body
                    body_top = min(open_y, close_y)
                    body_bottom = max(open_y, close_y)
                    for y in range(body_top, body_bottom + 1):
                        try:
                            self.stdscr.addstr(start_y + y, x_pos, "█", color)
                            self.stdscr.addstr(start_y + y, x_pos + 1, "█", color)
                            self.stdscr.addstr(start_y + y, x_pos + 2, "█", color)
                        except curses.error:
                            pass

                # Draw current price line
                if self.current_candle['close'] > 0:
                    current_y = int((max_high - self.current_candle['close']) / price_range * (chart_height - 1))
                    current_y = max(0, min(current_y, chart_height - 1))
                    try:
                        self.stdscr.addstr(start_y + current_y, 10, "─" * (chart_width - 10), curses.color_pair(4) | curses.A_BOLD)
                        self.stdscr.addstr(start_y + current_y, max_x - 10, f"${self.current_candle['close']:.2f}", curses.color_pair(4) | curses.A_BOLD)
                    except curses.error:
                        pass
        except Exception as e:
            self.logger.error(f"Error drawing candlestick chart: {e}")
            try:
                self.stdscr.addstr(start_y + chart_height // 2, max_x // 2 - 10, "Error drawing chart", curses.color_pair(2))
            except:
                pass

class ConsoleDashboardManager:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dashboard = None

    def start_dashboard(self):
        """Start the console dashboard."""
        self.logger.info("Starting console dashboard")
        self.dashboard = ConsoleDashboard(self.logger)
        self.dashboard.start()

    def stop_dashboard(self):
        """Stop the console dashboard."""
        if self.dashboard:
            self.logger.info("Stopping console dashboard")
            self.dashboard.stop()

    def update_candle(self, candle: Dict[str, float]):
        """Update candle data in the dashboard."""
        if self.dashboard:
            self.dashboard.update_candle(candle)

    def add_trade(self, trade: Dict[str, Any]):
        """Add a trade to the dashboard."""
        if self.dashboard:
            self.dashboard.add_trade(trade)

    def update_portfolio_value(self, value: float):
        """Update the portfolio value in the dashboard."""
        if self.dashboard:
            self.dashboard.update_portfolio_value(value)

    def update_position(self, size: float, entry_price: float, current_price: float):
        """Update position details in the dashboard."""
        if self.dashboard:
            self.dashboard.update_position(size, entry_price, current_price)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics in the dashboard."""
        if self.dashboard:
            self.dashboard.update_metrics(metrics)

    def update_strategy_signals(self, signals: Dict[str, Any]):
        """Update strategy signals in the dashboard."""
        if self.dashboard:
            self.dashboard.update_strategy_signals(signals)

    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data in the dashboard."""
        if self.dashboard:
            self.dashboard.update_market_data(market_data)

    def set_testnet(self, is_testnet: bool):
        """Set testnet mode in the dashboard."""
        if self.dashboard:
            self.dashboard.set_testnet(is_testnet)

    def update_market_stats(self, stats: Dict[str, Any]):
        """Update market stats in the dashboard."""
        if self.dashboard:
            self.dashboard.update_market_stats(stats)

    def update_system_stats(self, stats: Dict[str, Any]):
        """Update system stats in the dashboard."""
        if self.dashboard:
            self.dashboard.update_system_stats(stats)