"""
Enhanced console dashboard for real-time trading monitoring.
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
import random

class ConsoleDashboard:
    """Console-based dashboard for displaying trading information."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dashboard with configuration."""
        self.config = config
        self.stdscr = None
        self.running = True
        self.last_update = time.time()
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update the dashboard data."""
        self.config.update(data)
        self.last_update = time.time()
        
    def add_price(self, price: float) -> None:
        """Add a price point to the history."""
        self.config['price_history'].append(price)
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a trade to the history."""
        self.config['recent_trades'].append(trade)
        
    def _draw_frame(self) -> None:
        """Draw the main dashboard frame."""
        try:
            # Clear the screen
            self.stdscr.clear()
            
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Draw main frame
            self._draw_box(0, 0, max_y-1, max_x-1)
            
            # Draw header section
            header_height = 8  # Height for logo and basic info
            self._draw_box(0, 0, header_height, max_x-1)
            self._draw_header()
            
            # Draw portfolio section
            portfolio_start = header_height
            portfolio_height = 10
            self._draw_box(portfolio_start, 0, portfolio_start + portfolio_height, max_x//2-1)
            self._draw_portfolio_info()
            
            # Draw performance metrics section
            self._draw_box(portfolio_start, max_x//2, portfolio_start + portfolio_height, max_x-1)
            self._draw_performance_metrics()
            
            # Draw price chart section
            chart_start = portfolio_start + portfolio_height
            chart_height = max_y - chart_start - 3  # Leave space for system info
            self._draw_box(chart_start, 0, chart_start + chart_height, max_x-1)
            self._draw_price_chart()
            
            # Draw system info section
            system_start = max_y - 3
            self._draw_box(system_start, 0, max_y-1, max_x-1)
            self._draw_system_info()
            
            # Refresh the screen
            self.stdscr.refresh()
            
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_box(self, start_y: int, start_x: int, end_y: int, end_x: int) -> None:
        """Draw a box with double-line borders."""
        try:
            # Draw corners
            self.safe_addch(start_y, start_x, '╔')
            self.safe_addch(start_y, end_x, '╗')
            self.safe_addch(end_y, start_x, '╚')
            self.safe_addch(end_y, end_x, '╝')
            
            # Draw horizontal lines
            for x in range(start_x + 1, end_x):
                self.safe_addch(start_y, x, '═')
                self.safe_addch(end_y, x, '═')
            
            # Draw vertical lines
            for y in range(start_y + 1, end_y):
                self.safe_addch(y, start_x, '║')
                self.safe_addch(y, end_x, '║')
        except curses.error:
            pass
            
    def _draw_header(self) -> None:
        """Draw the dashboard header."""
        try:
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Draw ELVIS logo with pink color and bold
            self._draw_elvis_logo()
            
            # Draw mode and strategy info (moved down to avoid overlap)
            mode = "PRODUCTION" if self.config.get('PRODUCTION_MODE', False) else "PAPER"
            mode_color = curses.color_pair(1) if mode == "PRODUCTION" else curses.color_pair(3)
            self.safe_addstr(8, 2, f"Mode: {mode}", mode_color)
            self.safe_addstr(8, 20, f"Strategy: {self.config.get('ml_model', 'Unknown')}", curses.color_pair(4))
            
            # Draw current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.safe_addstr(8, max_x - len(current_time) - 2, current_time, curses.color_pair(5))
            
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_elvis_logo(self) -> None:
        """Draw the ELVIS logo."""
        try:
            logo = [
                "███████╗██╗     ██╗   ██╗██╗███████╗",
                "██╔════╝██║     ██║   ██║██║██╔════╝",
                "█████╗  ██║     ██║   ██║██║███████╗",
                "██╔══╝  ██║     ██║   ██║██║╚════██║",
                "███████╗███████╗╚██████╔╝██║███████║",
                "╚══════╝╚══════╝ ╚═════╝ ╚═╝╚══════╝"
            ]
            
            # Center the logo
            start_y = 1
            for i, line in enumerate(logo):
                if start_y + i < self.stdscr.getmaxyx()[0]:
                    x = (self.stdscr.getmaxyx()[1] - len(line)) // 2
                    # Use pink color (color pair 6) with bold attribute
                    self.safe_addstr(start_y + i, x, line, curses.color_pair(6) | curses.A_BOLD)
                    
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_portfolio_info(self) -> None:
        """Draw portfolio information."""
        try:
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Calculate position (moved down by 4 lines)
            y = 9  # Changed from 5 to 9
            x = 2
            
            # Draw portfolio value
            portfolio_value = self.config.get('portfolio_value', 0.0)
            self.safe_addstr(y, x, f"Portfolio Value: ${portfolio_value:,.2f}", curses.color_pair(4))
            
            # Draw position size
            position_size = self.config.get('position_size', 0.0)
            position_color = curses.color_pair(1) if position_size > 0 else curses.color_pair(2) if position_size < 0 else curses.A_NORMAL
            self.safe_addstr(y + 1, x, f"Position Size: {position_size:,.4f}", position_color)
                
                # Draw realized PnL
            realized_pnl = self.config.get('realized_pnl', 0.0)
            pnl_color = curses.color_pair(1) if realized_pnl > 0 else curses.color_pair(2) if realized_pnl < 0 else curses.A_NORMAL
            self.safe_addstr(y + 2, x, f"Realized PnL: ${realized_pnl:,.2f}", pnl_color)
            
            # Draw open positions
            self.safe_addstr(y + 4, x, "Open Positions:", curses.color_pair(4))
            open_positions = self.config.get('open_positions', [])
            if open_positions:
                for i, pos in enumerate(open_positions):
                    if y + 5 + i < max_y:
                        pos_color = curses.color_pair(1) if pos.get('pnl', 0) > 0 else curses.color_pair(2) if pos.get('pnl', 0) < 0 else curses.A_NORMAL
                        self.safe_addstr(y + 5 + i, x + 2, f"{pos.get('symbol', 'Unknown')}: {pos.get('size', 0):,.4f} @ ${pos.get('entry_price', 0):,.2f} (PnL: ${pos.get('pnl', 0):,.2f})", pos_color)
            else:
                self.safe_addstr(y + 5, x + 2, "No open positions", curses.color_pair(3))
                
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_performance_metrics(self) -> None:
        """Draw performance metrics."""
        try:
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Calculate position (moved down by 4 lines)
            y = 9  # Changed from 5 to 9
            x = max_x // 2
            
            # Draw trade statistics
            self.safe_addstr(y, x, "Trade Statistics:", curses.color_pair(4))
            self.safe_addstr(y + 1, x + 2, f"Total Trades: {self.config.get('total_trades', 0)}")
            self.safe_addstr(y + 2, x + 2, f"Win Rate: {self.config.get('win_rate', 0.0):.2%}")
            self.safe_addstr(y + 3, x + 2, f"Profit Factor: {self.config.get('profit_factor', 0.0):.2f}")
            
            # Draw risk metrics
            self.safe_addstr(y + 5, x, "Risk Metrics:", curses.color_pair(4))
            self.safe_addstr(y + 6, x + 2, f"Sharpe Ratio: {self.config.get('sharpe_ratio', 0.0):.2f}")
            self.safe_addstr(y + 7, x + 2, f"Max Drawdown: {self.config.get('max_drawdown', 0.0):.2%}")
            
            # Draw market regime
            self.safe_addstr(y + 9, x, "Market Regime:", curses.color_pair(4))
            regime = self.config.get('market_regime', 'Unknown')
            confidence = self.config.get('regime_confidence', 0.0)
            self.safe_addstr(y + 10, x + 2, f"{regime} (Confidence: {confidence:.2%})", curses.color_pair(3))
            
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_price_chart(self) -> None:
        """Draw the price chart."""
        try:
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Calculate position
            y = max_y - 15
            x = 2
            
            # Draw chart title
            self.safe_addstr(y, x, "Price Chart:", curses.color_pair(4))
            
            # Get price history
            price_history = self.config.get('price_history', [])
            if not price_history:
                self.safe_addstr(y + 1, x + 2, "No price data available", curses.color_pair(3))
                return
                
            # Calculate chart dimensions
            chart_height = 10
            chart_width = min(50, max_x - x - 10)
            
            # Calculate price range
            min_price = min(price_history)
            max_price = max(price_history)
            price_range = max_price - min_price
            
            if price_range == 0:
                price_range = 1
                
            # Draw price points
            for i in range(min(chart_width, len(price_history))):
                price = price_history[-(i + 1)]
                normalized_price = (price - min_price) / price_range
                chart_y = y + chart_height - int(normalized_price * (chart_height - 1))
                
                if chart_y >= y + 1 and chart_y < y + chart_height:
                    self.safe_addch(chart_y, x + 2 + i, 'o', curses.color_pair(1))
                    
            # Draw current price
            current_price = price_history[-1]
            self.safe_addstr(y + chart_height + 1, x + 2, f"Current Price: ${current_price:,.2f}", curses.color_pair(4))
            
            # Draw price scale
            self.safe_addstr(y + 1, x + 2, f"${max_price:,.2f}", curses.color_pair(4))
            self.safe_addstr(y + chart_height, x + 2, f"${min_price:,.2f}", curses.color_pair(4))
            
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def _draw_system_info(self) -> None:
        """Draw system information."""
        try:
            # Get terminal dimensions
            max_y, max_x = self.stdscr.getmaxyx()
            
            # Calculate position
            y = max_y - 3
            x = 2
            
            # Get actual CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=0.1)  # Get CPU usage with a small interval
            memory_usage = psutil.virtual_memory().percent
            
            # Draw CPU and memory usage with color coding
            cpu_color = curses.color_pair(1) if cpu_usage < 50 else curses.color_pair(2) if cpu_usage < 80 else curses.color_pair(3)
            mem_color = curses.color_pair(1) if memory_usage < 50 else curses.color_pair(2) if memory_usage < 80 else curses.color_pair(3)
            
            self.safe_addstr(y, x, f"CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}%", curses.color_pair(5))
            
            # Draw uptime and API calls
            uptime = self.config.get('uptime', 0)
            api_calls = self.config.get('api_calls', 0)
            self.safe_addstr(y + 1, x, f"Uptime: {uptime}s | API Calls: {api_calls}", curses.color_pair(5))
            
        except curses.error:
            # Handle curses errors gracefully
            pass
            
    def run(self) -> None:
        """Run the dashboard."""
        try:
            self.stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            curses.init_pair(6, 213, -1)  # Pink color for ELVIS logo
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(1)
            
            # Add refresh rate control
            refresh_rate = 0.2  # 5 FPS to reduce flickering
            last_refresh = time.time()
            
            # Initialize price history with some sample data
            if 'price_history' not in self.config:
                self.config['price_history'] = [50000.0]  # Initial price
                
            # Store last terminal size
            last_terminal_size = self.stdscr.getmaxyx()
            
            while self.running:
                try:
                    current_time = time.time()
                    current_terminal_size = self.stdscr.getmaxyx()
                    
                    # Only redraw if enough time has passed or terminal size changed
                    if (current_time - last_refresh >= refresh_rate or 
                        current_terminal_size != last_terminal_size):
                        
                        # Add a small random price change for testing
                        last_price = self.config['price_history'][-1]
                        new_price = last_price + (random.random() - 0.5) * 100
                        self.config['price_history'].append(new_price)
                        
                        # Keep only the last 50 prices
                        if len(self.config['price_history']) > 50:
                            self.config['price_history'] = self.config['price_history'][-50:]
                            
                        # Clear the screen
                        self.stdscr.clear()
                        
                        # Draw the frame
                        self._draw_frame()
                        
                        # Update last refresh time and terminal size
                        last_refresh = current_time
                        last_terminal_size = current_terminal_size
                    
                    # Check for quit command
                    c = self.stdscr.getch()
                    if c == ord('q'):
                        self.running = False
                        
                except curses.error:
                    # Handle curses errors gracefully
                    pass
                    
        except Exception as e:
            self.running = False
            raise e
            
        finally:
            if self.stdscr:
                curses.nocbreak()
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()
                
    def safe_addstr(self, y: int, x: int, text: str, attr=curses.A_NORMAL) -> None:
        """Safely add a string to the screen."""
        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass
            
    def safe_addch(self, y: int, x: int, ch: str, attr=curses.A_NORMAL) -> None:
        """Safely add a character to the screen."""
        try:
            self.stdscr.addch(y, x, ch, attr)
        except curses.error:
            pass

class ConsoleDashboardManager:
    """Manages the console dashboard."""
    
    def __init__(self, logger):
        """Initialize the dashboard manager."""
        self.logger = logger
        self.dashboard = ConsoleDashboard({
            'PRODUCTION_MODE': False,  # Default to testnet mode
            'portfolio_value': 0.0,
            'position_size': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_price': 0.0,
            'open_positions': [],
            'recent_trades': deque(maxlen=10),
            'market_regime': 'Unknown',
            'regime_confidence': 0.0,
            'ml_model': 'Ensemble',
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime': 0,
            'api_calls': 0,
            'price_history': deque(maxlen=50),
            'volume_history': deque(maxlen=50),
            'strategy_signals': {},
            'market_data': {'prices': []}
        })
        self.running = False
        
    def start_dashboard(self):
        """Start the dashboard."""
        self.logger.info("Starting console dashboard...")
        self.running = True
        self.dashboard.run()
        
    def stop_dashboard(self):
        """Stop the dashboard."""
        self.logger.info("Stopping console dashboard...")
        self.running = False
        self.dashboard.running = False
        
    def is_running(self):
        """Check if the dashboard is running."""
        return self.running
        
    def update_portfolio_value(self, value: float):
        """Update the portfolio value."""
        self.dashboard.update({'portfolio_value': value})
        
    def update_position(self, size: float, entry_price: float, current_price: float):
        """Update the position."""
        self.dashboard.update({
            'position_size': size,
            'entry_price': entry_price,
            'current_price': current_price
        })
        
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics."""
        self.dashboard.update(metrics)
        
    def update_strategy_signals(self, signals: Dict[str, Any]):
        """Update the strategy signals."""
        self.dashboard.update({'strategy_signals': signals})
        
    def update_open_positions(self, positions: List[Dict[str, Any]]):
        """Update the open positions."""
        self.dashboard.update({'open_positions': positions})
        
    def add_trade(self, trade: Dict[str, Any]):
        """Add a trade to the history."""
        self.dashboard.add_trade(trade)
        
    def update_market_data(self, data: Dict[str, Any]):
        """Update the market data."""
        self.dashboard.update({'market_data': data})
        
    def set_model_name(self, model_name: str):
        """Set the ML model name."""
        self.dashboard.update({'ml_model': model_name})
