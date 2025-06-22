import curses
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque
import psutil
import pandas as pd
import ta
from utils.logging_utils import setup_logger
import numpy as np

class ConsoleDashboard:
    """
    ConsoleDashboard provides a curses-based terminal UI for displaying trading system metrics,
    system resource usage, and recent trading activity. It is designed to be extensible for
    future enhancements such as multi-timeframe views, technical indicators, and interactive features.
    """

    def __init__(self, config=None, logger=None, price_fetcher=None):
        """
        Initialize the ConsoleDashboard.

        Args:
            config (dict): Configuration dictionary containing dynamic data to display.
            logger (logging.Logger): Logger instance for logging dashboard events and errors.
            price_fetcher (PriceFetcher): The price fetcher instance.
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.price_fetcher = price_fetcher
        self.animation_frame = 0
        self.stdscr = None
        self.running = False
        self.timeframe = "5m"  # Default timeframe
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.indicators = ["RSI", "MACD", "BBANDS"] # Default indicators
        self.drawing_mode = False
        self.draw_points = []
        self.lines = []

    def _draw_frame(self):
        """
        Draw a single frame of the dashboard UI, including header, system info, and layout boxes.
        Handles terminal resizing and ensures minimum size requirements.
        """
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()
            if max_y < 40 or max_x < 120:
                self.stdscr.addstr(0, 0, "Terminal too small, resize to at least 120x40")
                self.stdscr.refresh()
                return

            # Main layout boxes
            self._draw_box(0, 0, max_y - 1, max_x - 1) # Main border
            self._draw_header()

            # Define layout sections
            left_pane_width = 35
            chart_pane_width = max_x - left_pane_width - 30
            right_pane_width = 28
            
            chart_pane_x = left_pane_width + 1
            right_pane_x = chart_pane_x + chart_pane_width + 1

            # Draw panes
            self._draw_box(8, 1, max_y - 2, left_pane_width) # Left pane
            self._draw_box(8, chart_pane_x, max_y - 2, chart_pane_x + chart_pane_width) # Chart pane
            self._draw_box(8, right_pane_x, max_y - 2, max_x - 2) # Right pane

            # Draw content in panes
            self._draw_info_pane(9, 3)
            self._draw_chart_pane(9, chart_pane_x + 2, max_y - 10, chart_pane_width - 2)
            self._draw_volume_profile_pane(9, right_pane_x + 2, max_y - 20, right_pane_width - 2)
            self._draw_position_sizing_pane(max_y - 10, right_pane_x + 2, 8, right_pane_width - 2)

            self.animation_frame = (self.animation_frame + 1) % 10
            self.stdscr.refresh()
        except Exception as e:
            self.logger.error(f"Error drawing frame: {e}")

    def _draw_box(self, start_y: int, start_x: int, end_y: int, end_x: int):
        """
        Draw a rectangular box using ASCII characters.

        Args:
            start_y (int): Starting row.
            start_x (int): Starting column.
            end_y (int): Ending row.
            end_x (int): Ending column.
        """
        try:
            self.safe_addch(start_y, start_x, '+')
            self.safe_addch(start_y, end_x, '+')
            self.safe_addch(end_y, start_x, '+')
            self.safe_addch(end_y, end_x, '+')
            for x in range(start_x + 1, end_x):
                self.safe_addch(start_y, x, '-')
                self.safe_addch(end_y, x, '-')
            for y in range(start_y + 1, end_y):
                self.safe_addch(y, start_x, '|')
                self.safe_addch(y, end_x, '|')
        except curses.error:
            pass

    def _draw_header(self):
        """
        Draw the dashboard header with a stylized logo centered horizontally.
        """
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            logo = [
                "███████╗██╗     ██╗   ██╗██╗███████╗",
                "██╔════╝██║     ██║   ██║██║██╔════╝",
                "█████╗  ██║     ██║   ██║██║███████╗",
                "██╔══╝  ██║     ██║   ██║██║╚════██║",
                "███████╗███████╗╚██████╔╝██║███████║",
                "╚══════╝╚══════╝ ╚═════╝ ╚═╝╚══════╝"
            ]
            start_y = 1
            for i, line in enumerate(logo):
                x = (max_x - len(line)) // 2
                self.safe_addstr(start_y + i, x, line, curses.A_BOLD)
        except curses.error:
            pass

    def _draw_info_pane(self, start_y: int, start_x: int):
        """Draws the left pane with general info, PnL, and system status."""
        y = start_y
        
        # Time and Status
        self.safe_addstr(y, start_x, f"Time: {datetime.now().strftime('%H:%M:%S')}", curses.A_BOLD)
        self.safe_addstr(y + 1, start_x, f"Status: RUNNING", curses.A_BOLD)
        
        # Portfolio
        y += 3
        self.safe_addstr(y, start_x, "--- Portfolio ---", curses.A_BOLD)
        
        risk_manager = self.config.get('risk_manager')
        portfolio_value = self.config.get('portfolio_value', 0.0)
        unrealized_pnl = risk_manager.unrealized_pnl if risk_manager else 0.0
        realized_pnl = risk_manager.realized_pnl if risk_manager else 0.0
        
        self.safe_addstr(y + 1, start_x, f"Value: ${portfolio_value:,.2f}")
        self.safe_addstr(y + 2, start_x, f"Unrealized PnL: ${unrealized_pnl:,.2f}")
        self.safe_addstr(y + 3, start_x, f"Realized PnL: ${realized_pnl:,.2f}")

        # Positions
        y += 5
        self.safe_addstr(y, start_x, "--- Open Positions ---", curses.A_BOLD)
        open_positions = self.config.get('open_positions', [])
        if not open_positions:
            self.safe_addstr(y + 1, start_x, "None")
        else:
            for i, pos in enumerate(open_positions[:5]): # Show top 5
                self.safe_addstr(y + 1 + i, start_x, f"{pos['symbol']} {pos['amount']} @ {pos['price']}")

        # System Monitoring
        y += 8
        self.safe_addstr(y, start_x, "--- System Health ---", curses.A_BOLD)
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        system_monitor = self.config.get('system_monitor')
        
        self.safe_addstr(y + 1, start_x, f"CPU: {cpu_usage:.1f}%")
        self.safe_addstr(y + 2, start_x, f"Memory: {memory.percent:.1f}%")
        
        if system_monitor:
            latency = system_monitor.get_network_latency()
            self.safe_addstr(y + 3, start_x, f"Latency: {latency:.2f} ms")
            
            # This is a placeholder for API rate limits, as it requires a response object
            # rate_limits = system_monitor.get_api_rate_limits(response.headers)
            # self.safe_addstr(y + 4, start_x, f"API Limit: {rate_limits['limit']}")
            
            error_rates = system_monitor.get_error_rates()
            self.safe_addstr(y + 5, start_x, f"Errors: {sum(error_rates.values())}")

        # Performance Metrics
        y += 4
        self.safe_addstr(y, start_x, "--- Performance ---", curses.A_BOLD)
        performance_monitor = self.config.get('performance_monitor')
        if performance_monitor:
            sharpe = performance_monitor.calculate_rolling_sharpe()
            drawdown = performance_monitor.calculate_rolling_drawdown()
            sortino = performance_monitor.calculate_sortino_ratio()
            calmar = performance_monitor.calculate_calmar_ratio()
            var = risk_manager.calculate_var() if risk_manager else 0.0
            self.safe_addstr(y + 1, start_x, f"Sharpe: {sharpe:.2f}")
            self.safe_addstr(y + 2, start_x, f"Sortino: {sortino:.2f}")
            self.safe_addstr(y + 3, start_x, f"Calmar: {calmar:.2f}")
            self.safe_addstr(y + 4, start_x, f"Drawdown: {drawdown:.2%}")
            self.safe_addstr(y + 5, start_x, f"VaR (95%): ${var:,.2f}")

        # Trade Distribution
        y += 6
        self.safe_addstr(y, start_x, "--- Trade Distribution ---", curses.A_BOLD)
        trade_analyzer = self.config.get('trade_analyzer')
        if trade_analyzer:
            win_loss = trade_analyzer.get_win_loss_distribution()
            avg_pnl = trade_analyzer.get_average_pnl()
            self.safe_addstr(y + 1, start_x, f"Wins: {win_loss['wins']} | Losses: {win_loss['losses']}")
            self.safe_addstr(y + 2, start_x, f"Avg Win: ${avg_pnl['avg_win']:.2f} | Avg Loss: ${avg_pnl['avg_loss']:.2f}")

        # Position-level Risk
        y += 4
        self.safe_addstr(y, start_x, "--- Position Risk ---", curses.A_BOLD)
        if risk_manager:
            position_risk = risk_manager.get_position_level_risk()
            for symbol, risk in position_risk.items():
                self.safe_addstr(y + 1, start_x, f"{symbol}: ${risk:,.2f}")
                y += 1
>>>>>>> main

    def _draw_position_sizing_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the position sizing visualization pane."""
        self.safe_addstr(start_y, start_x, "--- Position Sizing ---", curses.A_BOLD)
        
        risk_manager = self.config.get('risk_manager')
        if not risk_manager:
            return
            
        portfolio_value = self.config.get('portfolio_value', 0.0)
        
        y = start_y + 2
        
        for symbol, position in risk_manager.open_positions.items():
            position_value = position.get('quantity', 0) * self.price_fetcher.get_current_price(symbol)
            size_percentage = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            bar_width = int(size_percentage / 100 * (width - 10))
            self.safe_addstr(y, start_x, f"{symbol}: {size_percentage:.2f}%")
            self.safe_addstr(y + 1, start_x, f"[{'#' * bar_width}{'-' * (width - 10 - bar_width)}]")
            y += 3
        
    def _draw_chart_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the right pane with the price chart and technical indicators."""
        self.safe_addstr(start_y, start_x, f"--- BTC/USDT Chart ({self.timeframe}) ---", curses.A_BOLD)
        
        df = self.price_fetcher.get_historical_klines("BTCUSDT", self.timeframe) if self.price_fetcher else pd.DataFrame()
        
        if df.empty:
            self.safe_addstr(start_y + 2, start_x, "Loading historical data...")
            return
        
        # Ensure correct data types
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])

        # Calculate indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()

        # Draw Chart (simplified ASCII chart)
        chart_height = height - 5
        chart_width = width - 4
        
        prices = df['close'].tail(chart_width).to_numpy()
        min_price, max_price = prices.min(), prices.max()
        price_range = max_price - min_price if max_price > min_price else 1

        for i, price in enumerate(prices):
            y_pos = chart_height - int(((price - min_price) / price_range) * (chart_height -1))
            self.safe_addch(start_y + 2 + y_pos, start_x + 2 + i, '*')

        # Draw lines
        for p1, p2 in self.lines:
            x1, y1 = p1
            x2, y2 = p2
            # This is a very basic line drawing algorithm, needs improvement
            dx = x2 - x1
            dy = y2 - y1
            for x in range(x1, x2 + 1):
                y = y1 + dy * (x - x1) / dx if dx != 0 else y1
                self.safe_addch(int(y), x, '.')

        # Draw Indicators
        indicator_y = start_y + height - 2
        rsi_val = df['rsi'].iloc[-1]
        macd_val = df['macd'].iloc[-1]
        self.safe_addstr(indicator_y, start_x, f"RSI: {rsi_val:.2f} | MACD: {macd_val:.2f}")

    def _draw_volume_profile_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the market depth pane."""
        self.safe_addstr(start_y, start_x, "--- Market Depth ---", curses.A_BOLD)
        
        order_book = self.price_fetcher.get_order_book("BTCUSDT") if self.price_fetcher else None
        
        if not order_book:
            self.safe_addstr(start_y + 2, start_x, "Loading order book...")
            return

        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        
        bids['cumulative'] = bids['qty'].cumsum()
        asks['cumulative'] = asks['qty'].cumsum()
        
        max_cumulative = max(bids['cumulative'].max(), asks['cumulative'].max())
        
        y = start_y + 2
        
        # Display asks (top 5)
        for i, row in asks.head(5).iloc[::-1].iterrows():
            bar_width = int((row['cumulative'] / max_cumulative) * (width - 15))
            self.safe_addstr(y + 5 - i, start_x, f"{row['price']:.2f} | {'#' * bar_width}")

        y += 6
        
        # Display bids (top 5)
        for i, row in bids.head(5).iterrows():
            bar_width = int((row['cumulative'] / max_cumulative) * (width - 15))
            self.safe_addstr(y + i, start_x, f"{row['price']:.2f} | {'#' * bar_width}")

    def run(self):
        """
        Main loop to run the dashboard UI. Handles keyboard input and periodic redraws.
        Press 'q' to quit the dashboard.
        """
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            self.running = True
            curses.mousemask(1)

            while self.running:
                self._draw_frame()
                c = self.stdscr.getch()
                if c == ord('q'):
                    self.running = False
                elif c == ord('d'):
                    self.drawing_mode = not self.drawing_mode
                    self.draw_points = []
                    self.logger.info(f"Drawing mode {'enabled' if self.drawing_mode else 'disabled'}.")
                elif c >= ord('1') and c <= ord('6'):
                    self.timeframe = self.timeframes[c - ord('1')]
                    self.logger.info(f"Switched to {self.timeframe} timeframe.")
                elif c == curses.KEY_MOUSE:
                    try:
                        _, mx, my, _, _ = curses.getmouse()
                        if self.drawing_mode:
                            self.draw_points.append((mx, my))
                            if len(self.draw_points) == 2:
                                self.lines.append(tuple(self.draw_points))
                                self.draw_points = []
                    except curses.error:
                        pass
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
        finally:
            curses.nocbreak()
            if self.stdscr:
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()

    def safe_addstr(self, y: int, x: int, text: str, attr=None):
        """
        Safely add a string to the curses window, handling out-of-bounds errors.

        Args:
            y (int): Row position.
            x (int): Column position.
            text (str): Text to display.
            attr: Optional curses attribute for styling.
        """
        try:
            if y >= 0 and x >= 0 and y < self.stdscr.getmaxyx()[0] and x + len(text) < self.stdscr.getmaxyx()[1]:
                if attr is not None:
                    self.stdscr.addstr(y, x, text, attr)
                else:
                    self.stdscr.addstr(y, x, text)
        except curses.error:
            pass

    def safe_addch(self, y: int, x: int, ch: str, attr=None):
        """
        Safely add a character to the curses window, handling out-of-bounds errors.

        Args:
            y (int): Row position.
            x (int): Column position.
            ch (str): Character to display.
            attr: Optional curses attribute for styling.
        """
        try:
            if y >= 0 and x >= 0 and y < self.stdscr.getmaxyx()[0] and x < self.stdscr.getmaxyx()[1]:
                if attr is not None:
                    self.stdscr.addch(y, x, ch, attr)
                else:
                    self.stdscr.addch(y, x, ch)
        except curses.error:
            pass

# Entry point

def main(stdscr):
    """
    Entry point for running the ConsoleDashboard standalone.

    Args:
        stdscr: curses standard screen object.
    """
    from utils.price_fetcher import PriceFetcher
    from binance.client import Client
    
    logger = setup_logger("ConsoleDashboard")
    
    # Mock PriceFetcher for standalone mode
    class MockPriceFetcher:
        def get_historical_klines(self, symbol, interval, limit=200):
            # Generate some random data for testing
            data = []
            price = 50000
            for i in range(limit):
                price += (np.random.rand() - 0.5) * 100
                data.append([
                    int(time.time() * 1000) - (limit - i) * 60000,
                    price, price + 50, price - 50, price, np.random.rand() * 10,
                    0,0,0,0,0,0
                ])
            df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(6)])
            return df

        def get_order_book(self, symbol, limit=100):
            bids = [[50000 - i*10, np.random.rand()*10] for i in range(5)]
            asks = [[50010 + i*10, np.random.rand()*10] for i in range(5)]
            return {'bids': bids, 'asks': asks}

    class MockRiskManager:
        def __init__(self):
            self.unrealized_pnl = 150.10
            self.realized_pnl = 450.20
        
        def calculate_var(self):
            return 100.0

    class MockPerformanceMonitor:
        def calculate_rolling_sharpe(self):
            return 1.5
        
        def calculate_rolling_drawdown(self):
            return -0.10

        def calculate_sortino_ratio(self):
            return 2.0

        def calculate_calmar_ratio(self):
            return 3.0

    class MockTradeAnalyzer:
        def get_win_loss_distribution(self):
            return {'wins': 10, 'losses': 5}
        
        def get_average_pnl(self):
            return {'avg_win': 100.0, 'avg_loss': -50.0}

    class MockSystemMonitor:
        def get_network_latency(self):
            return 50.0
            
        def get_error_rates(self):
            return {'binance': 1}

    config = {
        'portfolio_value': 10520.30,
        'open_positions': [
            {'symbol': 'BTC/USDT', 'amount': 0.1, 'price': 50000},
            {'symbol': 'ETH/USDT', 'amount': 2, 'price': 2500}
        ],
        'risk_manager': MockRiskManager(),
        'performance_monitor': MockPerformanceMonitor(),
        'trade_analyzer': MockTradeAnalyzer(),
        'system_monitor': MockSystemMonitor()
    }
    
    price_fetcher = MockPriceFetcher()
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    dashboard.stdscr = stdscr
    dashboard.run()

if __name__ == "__main__":
    curses.wrapper(main)
