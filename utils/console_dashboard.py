import curses
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque
import psutil
from utils.logging_utils import setup_logger

class ConsoleDashboard:
    """
    ConsoleDashboard provides a curses-based terminal UI for displaying trading system metrics,
    system resource usage, and recent trading activity. It is designed to be extensible for
    future enhancements such as multi-timeframe views, technical indicators, and interactive features.
    """

    def __init__(self, config=None, logger=None):
        """
        Initialize the ConsoleDashboard.

        Args:
            config (dict): Configuration dictionary containing dynamic data to display.
            logger (logging.Logger): Logger instance for logging dashboard events and errors.
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.animation_frame = 0
        self.stdscr = None
        self.running = False

    def _draw_frame(self):
        """
        Draw a single frame of the dashboard UI, including header, system info, and layout boxes.
        Handles terminal resizing and ensures minimum size requirements.
        """
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()
            if max_y < 40 or max_x < 100:
                self.stdscr.addstr(0, 0, "Terminal too small, resize to at least 100x40")
                self.stdscr.refresh()
                return

            self._draw_box(0, 0, max_y - 1, max_x - 1)
            self._draw_header()
            self._draw_system_info(10, 2, max_x - 4)
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

    def _draw_system_info(self, start_y: int, start_x: int, width: int):
        """
        Draw system and trading information including time, services, portfolio value,
        open positions, last trade, CPU and memory usage.

        Args:
            start_y (int): Starting row for info display.
            start_x (int): Starting column for info display.
            width (int): Width available for display.
        """
        try:
            y = start_y
            # Current Time
            self.safe_addstr(y, start_x, f"Time: {datetime.now().strftime('%H:%M:%S')}")
            y += 2

            # Running Services
            self.safe_addstr(y, start_x, "Running Services:", curses.A_BOLD)
            y += 1
            services = self.config.get('services', [])
            for service in services:
                self.safe_addstr(y, start_x + 2, f"- {service} [RUNNING]")
                y += 1

            y += 1

            # Portfolio Value
            portfolio_value = self.config.get('portfolio_value', 0.0)
            self.safe_addstr(y, start_x, f"Portfolio Value: ${portfolio_value:,.2f}")
            y += 1

            # Open Positions
            open_positions = self.config.get('open_positions', [])
            self.safe_addstr(y, start_x, f"Open Positions: {len(open_positions)}")
            y += 1

            # Last Trade
            recent_trades = self.config.get('recent_trades', [])
            if recent_trades:
                last_trade = recent_trades[-1]
                trade_info = f"{last_trade.get('side', '')} {last_trade.get('price', 0)} x{last_trade.get('quantity', 0)}"
                self.safe_addstr(y, start_x, f"Last Trade: {trade_info}")
            else:
                self.safe_addstr(y, start_x, "Last Trade: N/A")
            y += 2

            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=None)
            self.safe_addstr(y, start_x, f"CPU Usage: {cpu_usage:.1f}%")
            y += 1

            # Memory Usage
            memory = psutil.virtual_memory()
            self.safe_addstr(y, start_x, f"Memory Usage: {memory.percent:.1f}%")
        except curses.error:
            pass

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
            while self.running:
                self._draw_frame()
                c = self.stdscr.getch()
                if c == ord('q'):
                    self.running = False
                time.sleep(1)
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
    logger = setup_logger("ConsoleDashboard")
    config = {
        'services': ['trade_history_api.py', 'console_dashboard.py', 'main.py'],
        'portfolio_value': 1000.0,
        'open_positions': [],
        'recent_trades': []
    }
    dashboard = ConsoleDashboard(config=config, logger=logger)
    dashboard.stdscr = stdscr
    dashboard.run()

if __name__ == "__main__":
    curses.wrapper(main)
