"""
Console dashboard for the ELVIS project.
This module provides a terminal-based dashboard for monitoring trading activity.
"""

import curses
import logging
import threading
import time
import math
import os
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

class ConsoleDashboard:
    """
    Terminal-based dashboard for monitoring trading activity using curses.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the console dashboard.
        """
        self.logger = logger
        self.stdscr = None
        self.running = False
        self.data_lock = threading.Lock()
        
        # Dashboard data
        self.portfolio_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.metrics = {}
        self.trades = []
        self.strategy_signals = {}
        self.market_data = {}
        self.system_stats = {}
        self.candle_data = {}
        self.price_history = deque(maxlen=100)
        self.testnet = True
        self.view_mode = 1  # 1: Standard, 2: Detailed, 3: Chart
        self.model_name = "Unknown"
        self.open_positions = []
        
        # ASCII art for ELVIS logo
        self.logo = [
            " _______  _        __      __  _____   _____ ",
            "|  ____| | |       \\ \\    / / |_   _| / ____|",
            "| |__    | |        \\ \\  / /    | |  | (___  ",
            "|  __|   | |         \\/ /      | |   \\___ \\ ",
            "| |____  | |____      \\  /     _| |_  ____) |",
            "|______| |______|      \\/     |_____||_____/ "
        ]
    
    def start(self) -> None:
        """Start the dashboard."""
        if self.running:
            return
        
        self.running = True
        
        # Disable console logging while dashboard is running
        self._disable_console_logging()
        
        # Start the dashboard in a separate thread
        thread = threading.Thread(target=self._run_dashboard)
        thread.daemon = True
        thread.start()
    
    def _disable_console_logging(self) -> None:
        """Disable console logging while dashboard is running."""
        # Store all loggers
        self.loggers = {}
        
        # Also handle the root logger
        root_logger = logging.getLogger()
        root_console_handlers = []
        root_other_handlers = []
        for handler in root_logger.handlers:
            if (isinstance(handler, logging.StreamHandler) and 
                hasattr(handler, 'stream') and 
                handler.stream is not None and 
                hasattr(handler.stream, 'name') and 
                handler.stream.name == '<stdout>'):
                root_console_handlers.append(handler)
            else:
                root_other_handlers.append(handler)
        
        if root_console_handlers:
            self.loggers['root'] = root_console_handlers
            root_logger.handlers = root_other_handlers
        
        # Handle all other loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            # Store and remove console handlers
            console_handlers = []
            other_handlers = []
            for handler in logger.handlers:
                if (isinstance(handler, logging.StreamHandler) and 
                    hasattr(handler, 'stream') and 
                    handler.stream is not None and 
                    hasattr(handler.stream, 'name') and 
                    handler.stream.name == '<stdout>'):
                    console_handlers.append(handler)
                else:
                    other_handlers.append(handler)
            
            if console_handlers:
                self.loggers[name] = console_handlers
                logger.handlers = other_handlers
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self.running = False
        # Restore console logging
        self._restore_console_logging()
    
    def _restore_console_logging(self) -> None:
        """Restore console logging after dashboard is stopped."""
        if hasattr(self, 'loggers'):
            for name, handlers in self.loggers.items():
                if name == 'root':
                    logger = logging.getLogger()  # Root logger
                else:
                    logger = logging.getLogger(name)
                
                for handler in handlers:
                    # Check if handler is already in logger
                    if handler not in logger.handlers:
                        logger.addHandler(handler)
            
            # Clear the loggers dictionary
            self.loggers = {}
    
    def is_running(self) -> bool:
        """Check if the dashboard is running."""
        return self.running
    
    def update_portfolio_value(self, value: float) -> None:
        """Update the portfolio value."""
        with self.data_lock:
            self.portfolio_value = value
    
    def update_position(self, size: float, entry_price: float, current_price: float) -> None:
        """Update the position."""
        with self.data_lock:
            self.position_size = size
            self.entry_price = entry_price
            self.current_price = current_price
            self.price_history.append(current_price)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update the metrics."""
        with self.data_lock:
            self.metrics = metrics
    
    def update_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Update the trades."""
        with self.data_lock:
            self.trades = trades
            
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a single trade to the history."""
        with self.data_lock:
            # Only keep the most recent 50 trades
            if len(self.trades) >= 50:
                self.trades.pop(0)
            self.trades.append(trade)
    
    def update_strategy_signals(self, signals: Dict[str, Dict[str, bool]]) -> None:
        """Update the strategy signals."""
        with self.data_lock:
            self.strategy_signals = signals
    
    def update_market_data(self, data: Dict[str, Any]) -> None:
        """Update the market data."""
        with self.data_lock:
            self.market_data = data
    
    def update_system_stats(self, stats: Dict[str, Any]) -> None:
        """Update the system statistics."""
        with self.data_lock:
            self.system_stats = stats
    
    def update_candle(self, candle: Dict[str, Any]) -> None:
        """Update the candle data."""
        with self.data_lock:
            self.candle_data = candle
    
    def update_price_history(self, price: float) -> None:
        """Update the price history."""
        with self.data_lock:
            self.price_history.append(price)
    
    def set_testnet(self, testnet: bool) -> None:
        """Set the testnet flag."""
        with self.data_lock:
            self.testnet = testnet
    
    def set_model_name(self, model_name: str) -> None:
        """Set the ML model name."""
        with self.data_lock:
            self.model_name = model_name
    
    def update_open_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update the open positions."""
        with self.data_lock:
            self.open_positions = positions
    
    def _run_dashboard(self) -> None:
        """Run the dashboard."""
        try:
            # Initialize curses
            self.stdscr = curses.initscr()
            
            # Set up curses
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)  # Red for negative values
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Yellow for warnings
            curses.init_pair(3, curses.COLOR_GREEN, -1)  # Green for positive values
            curses.init_pair(4, curses.COLOR_MAGENTA, -1)  # Pink/Magenta for ELVIS logo
            curses.curs_set(0)  # Hide cursor
            self.stdscr.keypad(True)  # Enable keypad
            self.stdscr.timeout(100)  # Set timeout for getch
            
            # Main loop
            while self.running:
                try:
                    # Clear screen
                    self.stdscr.clear()
                    
                    # Get data
                    with self.data_lock:
                        data = {
                            'portfolio_value': self.portfolio_value,
                            'position_size': self.position_size,
                            'entry_price': self.entry_price,
                            'current_price': self.current_price,
                            'metrics': self.metrics,
                            'trades': self.trades,
                            'strategy_signals': self.strategy_signals,
                            'market_data': self.market_data,
                            'system_stats': self.system_stats,
                            'candle_data': self.candle_data,
                            'price_history': list(self.price_history),
                            'testnet': self.testnet,
                            'model_name': self.model_name,
                            'open_positions': self.open_positions,
                            'symbol': 'BTC/USDT'  # Add symbol for open positions
                        }
                    
                    # Draw a simple dashboard
                    self._draw_simple_dashboard(data)
                    
                    # Refresh screen
                    self.stdscr.refresh()
                    
                    # Handle input
                    key = self.stdscr.getch()
                    if key == ord('q'):
                        self.running = False
                    
                    # Sleep to reduce CPU usage
                    time.sleep(0.1)
                
                except curses.error:
                    # Handle curses errors (e.g., terminal resize)
                    pass
                except Exception as e:
                    self.logger.error(f"Error in dashboard loop: {e}")
                    time.sleep(1)  # Sleep to avoid tight error loop
        
        except Exception as e:
            self.logger.error(f"Error initializing dashboard: {e}")
        finally:
            # Clean up curses
            if self.stdscr:
                self.stdscr.keypad(False)
                curses.echo()
                curses.nocbreak()
                curses.endwin()
    
    def _draw_simple_dashboard(self, data: Dict[str, Any]) -> None:
        """Draw a simple dashboard."""
        try:
            # Get screen dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Draw frame around the entire dashboard
            self._draw_frame(0, 0, max_y-1, max_x-1)
            
            # Draw the ELVIS logo in pink
            for i, line in enumerate(self.logo):
                if i < max_y-2:  # Leave space for frame
                    self.stdscr.addstr(i+1, (max_x - len(line)) // 2, line, curses.color_pair(4))
            
            # Draw network information (testnet/production)
            network_text = "TESTNET MODE" if data.get('testnet', True) else "PRODUCTION MODE"
            network_attr = curses.A_BOLD
            if curses.has_colors():
                network_attr |= curses.color_pair(2) if data.get('testnet', True) else curses.color_pair(1)
            
            if len(self.logo) + 1 < max_y-1:
                self.stdscr.addstr(len(self.logo) + 1, (max_x - len(network_text)) // 2, network_text, network_attr)
            
            # Draw current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if max_x > len(current_time) + 4:
                self.stdscr.addstr(1, max_x - len(current_time) - 2, current_time)
            
            # Draw strategy and model information
            strategy_name = "Unknown"
            for name in data.get('strategy_signals', {}):
                strategy_name = name.replace('_', ' ').title()
                break
            
            strategy_text = f"Strategy: {strategy_name}"
            if 2 < max_y-1:
                self.stdscr.addstr(2, 2, strategy_text, curses.A_BOLD)
            
            # Draw ML model information
            model_name = data.get('model_name', "Unknown")
            model_text = f"ML Model: {model_name}"
            if 3 < max_y-1:
                self.stdscr.addstr(3, 2, model_text, curses.A_BOLD)
            
            # Draw portfolio information section
            section_y = len(self.logo) + 3
            if section_y < max_y-6:
                self._draw_section_frame(section_y, 2, 6, max_x//2 - 3, "Portfolio Information")
                
                # Draw portfolio information
                y = section_y + 1
                if y < max_y-1:
                    self.stdscr.addstr(y, 4, f"Portfolio Value: ${data.get('portfolio_value', 0.0):,.2f}")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, 4, f"Position Size: {data.get('position_size', 0.0):.8f} BTC")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, 4, f"Entry Price: ${data.get('entry_price', 0.0):,.2f}")
                
                # Draw current price with emphasis
                y += 1
                if y < max_y-1:
                    price_text = f"Current BTC Price: ${data.get('current_price', 0.0):,.2f}"
                    self.stdscr.addstr(y, 4, price_text, curses.A_BOLD)
                
                # Draw realized PnL
                y += 1
                if y < max_y-1:
                    # Calculate realized PnL from trades
                    realized_pnl = sum(trade.get('pnl', 0.0) for trade in data.get('trades', []) if 'pnl' in trade)
                    pnl_text = f"Realized PnL: ${realized_pnl:,.2f}"
                    pnl_attr = curses.A_BOLD
                    if curses.has_colors():
                        pnl_attr |= curses.color_pair(3) if realized_pnl >= 0 else curses.color_pair(1)
                    self.stdscr.addstr(y, 4, pnl_text, pnl_attr)
            
            # Draw open positions section
            positions_y = section_y + 7
            if positions_y < max_y-6:
                open_positions = data.get('open_positions', [])
                if not open_positions and data.get('position_size', 0.0) > 0:
                    # Create a position entry from current position data
                    open_positions = [{
                        'symbol': data.get('symbol', 'BTC/USDT'),
                        'size': data.get('position_size', 0.0),
                        'entry_price': data.get('entry_price', 0.0),
                        'current_price': data.get('current_price', 0.0),
                        'pnl': (data.get('current_price', 0.0) - data.get('entry_price', 0.0)) * data.get('position_size', 0.0),
                        'pnl_pct': ((data.get('current_price', 0.0) / data.get('entry_price', 0.0)) - 1) * 100 if data.get('entry_price', 0.0) > 0 else 0.0
                    }]
                
                if open_positions:
                    # Calculate height based on number of positions (header + 1 line per position + border)
                    positions_height = min(len(open_positions) + 2, 6)
                    self._draw_section_frame(positions_y, 2, positions_height, max_x - 4, "Open Positions")
                    
                    # Draw header
                    y = positions_y + 1
                    if y < max_y-1:
                        self.stdscr.addstr(y, 4, "Symbol")
                        self.stdscr.addstr(y, 15, "Size")
                        self.stdscr.addstr(y, 30, "Entry Price")
                        self.stdscr.addstr(y, 45, "Current Price")
                        self.stdscr.addstr(y, 60, "Leverage")
                        self.stdscr.addstr(y, 70, "PnL")
                        self.stdscr.addstr(y, 85, "PnL %")
                    
                    # Draw positions
                    for i, position in enumerate(open_positions):
                        if i >= positions_height - 2:  # Skip if we run out of space
                            break
                        
                        y = positions_y + 2 + i
                        if y < max_y-1:
                            self.stdscr.addstr(y, 4, position.get('symbol', ''))
                            self.stdscr.addstr(y, 15, f"{position.get('size', 0.0):.8f}")
                            self.stdscr.addstr(y, 30, f"${position.get('entry_price', 0.0):,.2f}")
                            self.stdscr.addstr(y, 45, f"${position.get('current_price', 0.0):,.2f}")
                            
                            # Leverage with bold
                            leverage = position.get('leverage', 1)
                            leverage_text = f"{leverage}x"
                            self.stdscr.addstr(y, 60, leverage_text, curses.A_BOLD)
                            
                            # PnL with color
                            pnl = position.get('pnl', 0.0)
                            pnl_text = f"${pnl:,.2f}"
                            pnl_attr = curses.A_NORMAL
                            if curses.has_colors():
                                pnl_attr = curses.color_pair(3) if pnl >= 0 else curses.color_pair(1)
                            self.stdscr.addstr(y, 70, pnl_text, pnl_attr)
                            
                            # PnL % with color
                            pnl_pct = position.get('pnl_pct', 0.0)
                            pnl_pct_text = f"{pnl_pct:+.2f}%"
                            pnl_pct_attr = curses.A_NORMAL
                            if curses.has_colors():
                                pnl_pct_attr = curses.color_pair(3) if pnl_pct >= 0 else curses.color_pair(1)
                            self.stdscr.addstr(y, 85, pnl_pct_text, pnl_pct_attr)
            
            # Draw metrics section
            metrics_y = section_y
            if metrics_y < max_y-6:
                self._draw_section_frame(metrics_y, max_x//2 + 1, 6, max_x - max_x//2 - 3, "Performance Metrics")
                
                metrics = data.get('metrics', {})
                
                y = metrics_y + 1
                if y < max_y-1:
                    self.stdscr.addstr(y, max_x//2 + 3, f"Total Trades: {metrics.get('total_trades', 0)}")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, max_x//2 + 3, f"Win Rate: {metrics.get('win_rate', 0.0):.1f}%")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, max_x//2 + 3, f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, max_x//2 + 3, f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, max_x//2 + 3, f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2f}%")
            
            # Draw system information section
            system_y = section_y + 7
            if system_y < max_y-4:
                self._draw_section_frame(system_y, 2, 4, max_x - 4, "System Information")
                
                # Get system stats
                system_stats = data.get('system_stats', {})
                if not system_stats:
                    # Generate some basic stats if none provided
                    system_stats = {
                        'cpu_usage': psutil.cpu_percent(),
                        'memory_usage': psutil.virtual_memory().percent,
                        'uptime': time.time() - psutil.boot_time()
                    }
                
                # Format uptime
                uptime = system_stats.get('uptime', 0)
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Draw system stats in two columns
                y = system_y + 1
                if y < max_y-1:
                    self.stdscr.addstr(y, 4, f"CPU Usage: {system_stats.get('cpu_usage', 0.0):.1f}%")
                    self.stdscr.addstr(y, max_x//2 + 3, f"Uptime: {uptime_str}")
                
                y += 1
                if y < max_y-1:
                    self.stdscr.addstr(y, 4, f"Memory Usage: {system_stats.get('memory_usage', 0.0):.1f}%")
                    self.stdscr.addstr(y, max_x//2 + 3, f"API Calls: {system_stats.get('api_calls', 0)}")
            
            # Draw command info
            if max_y - 2 >= 0:
                self.stdscr.addstr(max_y - 2, (max_x - 18) // 2, "Press 'q' to quit")
        
        except Exception as e:
            self.logger.error(f"Error drawing simple dashboard: {e}")
    
    def _draw_frame(self, y1: int, x1: int, y2: int, x2: int) -> None:
        """Draw a frame around the specified area."""
        try:
            # Draw corners
            self.stdscr.addch(y1, x1, curses.ACS_ULCORNER)
            self.stdscr.addch(y1, x2, curses.ACS_URCORNER)
            self.stdscr.addch(y2, x1, curses.ACS_LLCORNER)
            self.stdscr.addch(y2, x2, curses.ACS_LRCORNER)
            
            # Draw horizontal lines
            for x in range(x1 + 1, x2):
                self.stdscr.addch(y1, x, curses.ACS_HLINE)
                self.stdscr.addch(y2, x, curses.ACS_HLINE)
            
            # Draw vertical lines
            for y in range(y1 + 1, y2):
                self.stdscr.addch(y, x1, curses.ACS_VLINE)
                self.stdscr.addch(y, x2, curses.ACS_VLINE)
        except curses.error:
            # Handle curses errors (e.g., drawing at the bottom-right corner)
            pass
    
    def _draw_section_frame(self, y: int, x: int, height: int, width: int, title: str) -> None:
        """Draw a frame for a section with a title."""
        try:
            # Draw the frame
            self._draw_frame(y, x, y + height, x + width)
            
            # Draw the title
            if len(title) > width - 4:
                title = title[:width - 7] + "..."
            
            self.stdscr.addstr(y, x + 2, f" {title} ")
        except curses.error:
            # Handle curses errors
            pass
        except Exception as e:
            self.logger.error(f"Error drawing section frame: {e}")


class ConsoleDashboardManager:
    """
    Manager for the console dashboard.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the console dashboard manager.
        """
        self.logger = logger
        self.dashboard = ConsoleDashboard(logger)
        self.trades = []
    
    def start_dashboard(self) -> None:
        """
        Start the dashboard.
        """
        self.dashboard.start()
    
    def stop_dashboard(self) -> None:
        """
        Stop the dashboard.
        """
        self.dashboard.stop()
    
    def is_running(self) -> bool:
        """
        Check if the dashboard is running.
        """
        return self.dashboard.is_running()
    
    def update_portfolio_value(self, value: float) -> None:
        """
        Update the portfolio value.
        """
        self.dashboard.update_portfolio_value(value)
    
    def update_position(self, size: float, entry_price: float, current_price: float) -> None:
        """
        Update the position.
        """
        self.dashboard.update_position(size, entry_price, current_price)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update the metrics.
        """
        self.dashboard.update_metrics(metrics)
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade.
        """
        self.trades.append(trade)
        self.dashboard.update_trades(self.trades)
    
    def update_strategy_signals(self, signals: Dict[str, Dict[str, bool]]) -> None:
        """
        Update the strategy signals.
        """
        self.dashboard.update_strategy_signals(signals)
    
    def update_market_data(self, data: Dict[str, Any]) -> None:
        """
        Update the market data.
        """
        self.dashboard.update_market_data(data)
    
    def update_system_stats(self, stats: Dict[str, Any]) -> None:
        """
        Update the system statistics.
        """
        self.dashboard.update_system_stats(stats)
    
    def update_candle(self, candle: Dict[str, Any]) -> None:
        """
        Update the candle data.
        """
        self.dashboard.update_candle(candle)
    
    def set_testnet(self, testnet: bool) -> None:
        """
        Set the testnet flag.
        """
        self.dashboard.set_testnet(testnet)
    
    def set_model_name(self, model_name: str) -> None:
        """
        Set the ML model name.
        """
        self.dashboard.set_model_name(model_name)
    
    def update_open_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        Update the open positions.
        """
        self.dashboard.update_open_positions(positions)
