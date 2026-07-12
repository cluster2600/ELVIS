import curses
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import psutil
import ta

from utils.console_dashboard_support import LogHandler, create_api_tester

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np

from config.config import TRADING_CONFIG
from utils.logging_utils import setup_logger


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
        self.indicators = ["RSI", "MACD", "BBANDS"]  # Default indicators
        self.drawing_mode = False
        self.draw_points = []
        self.lines = []
        self.messages = []  # Store console messages
        self.log_messages = deque(maxlen=100)  # Store recent log messages
        self.last_chart_data = None  # Cache last chart to reduce flashing
        self.chart_buffer = {}  # Store chart display buffer

        # API monitoring lifecycle is centralized in helper module.
        self.api_tester = create_api_tester(logger)

        # Set up log handler to capture logs
        self.log_handler = LogHandler(self)
        self.log_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        # Add handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)

    def _draw_frame(self):
        """
        Draw a single frame of the dashboard UI, including header, system info, and layout boxes.
        Handles terminal resizing and ensures minimum size requirements.
        """
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()
            if max_y < 40 or max_x < 120:
                self.stdscr.addstr(
                    0, 0, "Terminal too small, resize to at least 120x40"
                )
                self.stdscr.refresh()
                return

            # Main layout boxes
            self._draw_box(0, 0, max_y - 1, max_x - 1)  # Main border
            self._draw_header()

            # Define layout sections with larger sidebars and smaller chart
            left_pane_width = 50  # Increased from original 35 to 50 for bigger sidebar
            max_x = int(max_x)  # Ensure integer type
            right_pane_width = 55  # Increased from original 45 to 55 for bigger sidebar
            chart_pane_width = max(
                15, max_x - left_pane_width - right_pane_width - 3
            )  # Smaller chart area

            chart_pane_x = left_pane_width + 1
            right_pane_x = chart_pane_x + chart_pane_width + 1

            # Draw panes
            self._draw_box(8, 1, max_y - 2, left_pane_width)  # Left pane
            self._draw_box(
                8, chart_pane_x, max_y - 2, chart_pane_x + chart_pane_width
            )  # Chart pane
            self._draw_box(8, right_pane_x, max_y - 2, max_x - 2)  # Right pane

            # Add a marker in the right pane to help locate it
            try:
                self.safe_addstr(
                    8,
                    right_pane_x + 1,
                    f"RIGHT PANE (cols {right_pane_x}-{max_x-2})",
                    curses.color_pair(4),
                )
            except:
                pass

            # Draw content in panes with individual error handling
            try:
                self._draw_info_pane(9, 3)
            except Exception as e:
                self.logger.error(f"Error in _draw_info_pane: {e}")

            try:
                self._draw_chart_pane(
                    9, chart_pane_x + 1, max_y - 15, chart_pane_width - 1
                )  # Reduced padding for wider chart
            except Exception as e:
                self.logger.error(f"Error in _draw_chart_pane: {e}")

            try:
                # Give market depth more space and better positioning
                market_depth_height = min(18, max_y - 25)  # More reasonable height
                if market_depth_height > 10:  # Only draw if we have enough space
                    self._draw_volume_profile_pane(
                        9, right_pane_x + 2, market_depth_height, right_pane_width - 2
                    )
                    self.logger.debug(
                        f"Market depth drawn at x={right_pane_x + 2}, h={market_depth_height}"
                    )
                else:
                    self.safe_addstr(
                        9, right_pane_x + 2, "Terminal too small", curses.color_pair(2)
                    )
            except Exception as e:
                self.logger.error(f"Error in _draw_volume_profile_pane: {e}")
                # Draw error message in the right pane so user can see something
                self.safe_addstr(
                    9, right_pane_x + 2, "Market Depth Error", curses.color_pair(2)
                )

            try:
                # API Status widget below market depth
                api_status_y = (
                    9 + market_depth_height + 2
                    if "market_depth_height" in locals()
                    else max_y - 20
                )
                api_status_height = 8  # Fixed height for API status
                if api_status_y + api_status_height < max_y - 10:  # Ensure it fits
                    self._draw_api_status_pane(
                        api_status_y,
                        right_pane_x + 2,
                        api_status_height,
                        right_pane_width - 2,
                    )

                    # Position sizing below API status with adjusted spacing
                    position_y = api_status_y + api_status_height + 1
                    position_height = max(6, max_y - position_y - 3)
                else:
                    # Fallback: Position sizing only if no room for API status
                    position_y = (
                        9 + market_depth_height + 2
                        if "market_depth_height" in locals()
                        else max_y - 12
                    )
                    position_height = max(6, max_y - position_y - 3)

                if position_height > 0:
                    self._draw_position_sizing_pane(
                        position_y,
                        right_pane_x + 2,
                        position_height,
                        right_pane_width - 2,
                    )
            except Exception as e:
                self.logger.error(f"Error in _draw_position_sizing_pane: {e}")

            # Draw console messages at the bottom
            try:
                self._draw_console_messages(max_y - 10, 3, 8, max_x - 6)
            except Exception as e:
                self.logger.error(f"Error in _draw_console_messages: {e}")

            self.animation_frame = (self.animation_frame + 1) % 10
            self.stdscr.refresh()
        except Exception as e:
            self.logger.error(f"Error drawing frame: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

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
            self.safe_addch(start_y, start_x, "+")
            self.safe_addch(start_y, end_x, "+")
            self.safe_addch(end_y, start_x, "+")
            self.safe_addch(end_y, end_x, "+")
            for x in range(start_x + 1, end_x):
                self.safe_addch(start_y, x, "-")
                self.safe_addch(end_y, x, "-")
            for y in range(start_y + 1, end_y):
                self.safe_addch(y, start_x, "|")
                self.safe_addch(y, end_x, "|")
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
                "██╔══╝  ██║     ╚██╗ ██╔╝██║╚════██║",
                "███████╗███████╗ ╚████╔╝ ██║███████║",
                "╚══════╝╚══════╝  ╚═══╝  ╚═╝╚══════╝",
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

        # Time and Status - always live
        try:
            current_time = datetime.now()
        except NameError:
            # Fallback if datetime import fails
            import time

            current_time = time.strftime("%H:%M:%S")
            self.safe_addstr(y, start_x, f"Time: {current_time}", curses.color_pair(4))
            self.safe_addstr(
                y + 1,
                start_x,
                f"Date: {time.strftime('%Y-%m-%d')}",
                curses.color_pair(6),
            )
            self.safe_addstr(
                y + 2,
                start_x,
                f"Status: LIVE TRADING",
                curses.color_pair(1) | curses.A_BOLD,
            )
            y += 3
            self.safe_addstr(
                y, start_x, "--- Portfolio ---", curses.color_pair(3) | curses.A_BOLD
            )
            return  # Exit early with fallback

        # Normal datetime handling
        self.safe_addstr(
            y,
            start_x,
            f"Time: {current_time.strftime('%H:%M:%S')}",
            curses.color_pair(4),
        )
        self.safe_addstr(
            y + 1,
            start_x,
            f"Date: {current_time.strftime('%Y-%m-%d')}",
            curses.color_pair(6),
        )
        self.safe_addstr(
            y + 2,
            start_x,
            f"Status: LIVE TRADING",
            curses.color_pair(1) | curses.A_BOLD,
        )

        # Portfolio
        y += 3
        self.safe_addstr(
            y, start_x, "--- Portfolio ---", curses.color_pair(3) | curses.A_BOLD
        )

        risk_manager = self.config.get("risk_manager")

        # Calculate live P&L directly from database
        try:
            # Get live data from paper trading database
            # Calculate realized P&L from RECENT trades only (after reset)
            from utils.paper_trade_db import (
                get_all_trades,
                get_conn,
                get_open_positions,
            )

            realized_pnl = 0.0

            try:
                conn = get_conn()
                if conn:
                    with conn.cursor() as c:
                        # Get the latest reset timestamp
                        c.execute("""
                            SELECT reset_timestamp FROM trading_session_resets 
                            ORDER BY reset_timestamp DESC LIMIT 1
                        """)
                        reset_result = c.fetchone()

                        if reset_result:
                            reset_timestamp = reset_result[0]
                            # Get P&L from trades after reset only
                            c.execute(
                                """
                                SELECT COALESCE(SUM(pnl), 0) FROM trades 
                                WHERE timestamp >= %s
                                AND pnl BETWEEN -100 AND 100
                            """,
                                (reset_timestamp,),
                            )
                            result = c.fetchone()
                            if result:
                                realized_pnl = (
                                    float(result[0]) if result[0] is not None else 0.0
                                )
                        else:
                            # No reset found, default to 0 (fresh start)
                            realized_pnl = 0.0
                    conn.close()

                # Cap P&L to reasonable range
                realized_pnl = max(-1000.0, min(1000.0, realized_pnl))

            except Exception as e:
                self.logger.warning(
                    f"Could not calculate reset-based realized P&L: {e}"
                )
                realized_pnl = 0.0

            # Calculate unrealized P&L from open positions
            open_positions_raw = get_open_positions()
            unrealized_pnl = 0.0

            for pos in open_positions_raw:
                if len(pos) >= 5:
                    try:
                        # Position format: (id, symbol, side, entry_price, quantity, leverage, entry_time)
                        symbol = pos[1]
                        side = pos[2]  # BUY or SELL
                        entry_price = float(pos[3])
                        quantity = float(pos[4])

                        # Get live current price for this specific symbol
                        current_price = entry_price  # Fallback to entry price
                        if self.price_fetcher:
                            try:
                                # The symbol from the database is already in the correct format (e.g., 'BTCUSDT')
                                if hasattr(self.price_fetcher, "get_current_price"):
                                    fetched_price = (
                                        self.price_fetcher.get_current_price(symbol)
                                    )
                                    if fetched_price and fetched_price > 0:
                                        current_price = float(fetched_price)
                                        self.logger.debug(
                                            f"Portfolio P&L calc - {symbol}: Live ${current_price:.2f} vs Entry ${entry_price:.2f}"
                                        )
                                    else:
                                        self.logger.debug(
                                            f"Price fetcher returned {fetched_price} for {symbol}, using entry price"
                                        )
                                else:
                                    self.logger.debug(
                                        f"Price fetcher has no get_current_price method, using entry price"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error fetching price for {symbol}: {e}"
                                )

                        # Calculate P&L based on position side
                        if side.upper() == "BUY":  # LONG position
                            position_pnl = (current_price - entry_price) * quantity
                        else:  # SHORT position
                            position_pnl = (entry_price - current_price) * quantity
                        unrealized_pnl += position_pnl
                        self.logger.debug(
                            f"Position {symbol}: P&L = ${position_pnl:.2f} (qty: {quantity}, entry: ${entry_price:.2f}, current: ${current_price:.2f})"
                        )
                    except Exception as e:
                        self.logger.warning(f"Error calculating P&L for position: {e}")
                        pass

            # LIVE PORTFOLIO CALCULATION - Get actual balance from executor
            portfolio_value = 1000.0  # Fallback
            btc_price = 107000.0  # Fallback

            try:
                # Get LIVE executor balance
                from core.di import container

                executor = container.get_optional("executor")
                if executor and hasattr(executor, "get_balance"):
                    balance_info = executor.get_balance()
                    usdt_balance = float(balance_info.get("USDT", 1000.0))
                    btc_balance = float(balance_info.get("BTC", 0.0))

                    # Get LIVE BTC price from API
                    if self.price_fetcher and hasattr(
                        self.price_fetcher, "get_current_price"
                    ):
                        fetched_price = self.price_fetcher.get_current_price("BTCUSDT")
                        if fetched_price:
                            btc_price = float(fetched_price)

                    # Calculate total portfolio value from LIVE balances
                    btc_value_in_usdt = btc_balance * btc_price
                    portfolio_value = usdt_balance + btc_value_in_usdt

                    self.logger.debug(
                        f"LIVE Portfolio - USDT: ${usdt_balance:.2f}, BTC: {btc_balance:.6f}, "
                        f"BTC Value: ${btc_value_in_usdt:.2f}, Total: ${portfolio_value:.2f}"
                    )
                else:
                    # Fallback: use starting value plus realized P&L
                    portfolio_value = 1000.0 + realized_pnl

            except Exception as balance_error:
                self.logger.warning(f"Could not get live balance: {balance_error}")
                # Fallback: use starting value plus realized P&L
                portfolio_value = 1000.0 + realized_pnl

            # Add unrealized P&L only if it's reasonable (not inflated)
            if abs(unrealized_pnl) < 5000:  # Sanity check for unrealized P&L
                portfolio_value += unrealized_pnl

            # Don't use config portfolio value as it may be inflated from bad position sizing

            self.logger.debug(
                f"Live P&L calculated - Realized: ${realized_pnl:.2f}, "
                f"Unrealized: ${unrealized_pnl:.2f}, Portfolio: ${portfolio_value:.2f}"
            )

        except Exception as e:
            self.logger.warning(f"Error calculating live P&L: {e}")
            # Use minimal fallback values - avoid stale config data
            portfolio_value = 1000.0  # Starting amount
            unrealized_pnl = 0.0
            realized_pnl = 0.0

        self.safe_addstr(y + 1, start_x, f"Value: ${portfolio_value:,.2f}")

        # Color code PnL based on positive/negative
        unrealized_color = (
            curses.color_pair(1) if unrealized_pnl >= 0 else curses.color_pair(2)
        )
        realized_color = (
            curses.color_pair(1) if realized_pnl >= 0 else curses.color_pair(2)
        )

        self.safe_addstr(
            y + 2, start_x, f"Unrealized PnL: ${unrealized_pnl:,.2f}", unrealized_color
        )
        self.safe_addstr(
            y + 3, start_x, f"Realized PnL: ${realized_pnl:,.2f}", realized_color
        )

        # Positions (Live from Database with Real-time PnL) + Recently Closed
        y += 5
        self.safe_addstr(
            y, start_x, "--- Open Positions ---", curses.color_pair(3) | curses.A_BOLD
        )

        try:
            # Get live positions directly from database
            from utils.paper_trade_db import get_open_positions

            live_positions = get_open_positions()

            # Debug logging for dashboard
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logger.debug(
                f"Dashboard {timestamp}: Found {len(live_positions) if live_positions else 0} positions"
            )

            if live_positions:
                displayed_positions = 0
                for pos in live_positions:  # Show all positions
                    if (
                        len(pos) >= 5
                    ):  # Need at least 5 fields: id, symbol, side, entry_price, quantity
                        symbol = pos[1]
                        side = pos[2]  # BUY or SELL
                        entry_price = float(pos[3])
                        quantity = float(pos[4])

                        # Get live current price for this specific symbol
                        current_price = entry_price  # Fallback to entry price
                        if self.price_fetcher:
                            try:
                                # The symbol from the database is already in the correct format (e.g., 'BTCUSDT')
                                if hasattr(self.price_fetcher, "get_current_price"):
                                    fetched_price = (
                                        self.price_fetcher.get_current_price(symbol)
                                    )
                                    if fetched_price and fetched_price > 0:
                                        current_price = float(fetched_price)
                                        self.logger.debug(
                                            f"Live price for {symbol}: ${current_price:.2f}"
                                        )
                                    else:
                                        self.logger.debug(
                                            f"Price fetcher returned {fetched_price} for {symbol}, using entry price"
                                        )
                                else:
                                    self.logger.debug(
                                        f"Price fetcher has no get_current_price method, using entry price"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error fetching live price for {symbol}: {e}"
                                )
                                pass

                        # Calculate comprehensive real-time P&L with all fees
                        try:
                            leverage = (
                                int(pos[5])
                                if len(pos) > 5
                                else TRADING_CONFIG.get("DEFAULT_LEVERAGE", 50)
                            )
                            entry_time = pos[6] if len(pos) > 6 else None

                            # Try to get executor for comprehensive fee calculation
                            executor = None
                            try:
                                from core.di import container

                                executor = container.get_optional("executor")
                            except:
                                pass

                            if executor and hasattr(
                                executor, "calculate_open_position_pnl"
                            ):
                                # Get comprehensive P&L including all Binance fees
                                pnl_detail = executor.calculate_open_position_pnl(
                                    symbol,
                                    side,
                                    current_price,
                                    entry_price,
                                    quantity,
                                    leverage,
                                    entry_time,
                                )
                                net_pnl = pnl_detail["net_pnl"]
                                gross_pnl = pnl_detail["gross_pnl"]
                                total_fees = (
                                    pnl_detail["ongoing_costs"]
                                    + pnl_detail["estimated_exit_fee"]
                                )
                                funding_fee = pnl_detail["funding_fee"]
                                borrowing_cost = pnl_detail["borrowing_cost"]
                                hours_held = pnl_detail["hours_held"]

                                # Show fee impact
                                fee_impact = f" (Fees: ${total_fees:.4f})"
                                if hours_held > 0:
                                    fee_impact += f" {hours_held:.1f}h"
                            else:
                                # Fallback to simple calculation
                                pnl_multiplier = 1 if side.upper() == "BUY" else -1
                                net_pnl = (
                                    (current_price - entry_price)
                                    * quantity
                                    * pnl_multiplier
                                )
                                gross_pnl = net_pnl
                                fee_impact = " (est)"
                        except Exception as e:
                            # Safe fallback
                            pnl_multiplier = 1 if side.upper() == "BUY" else -1
                            net_pnl = (
                                (current_price - entry_price)
                                * quantity
                                * pnl_multiplier
                            )
                            gross_pnl = net_pnl
                            fee_impact = ""

                        # Display side as received from database
                        display_side = side.upper()

                        # Color code based on net P&L
                        try:
                            pnl_color = (
                                curses.color_pair(1)
                                if net_pnl >= 0
                                else curses.color_pair(2)
                            )
                        except:
                            pnl_color = 0

                        # Display position with live price and comprehensive P&L info
                        # Format quantity based on size for better readability
                        if abs(quantity) >= 1.0:
                            qty_display = f"{abs(quantity):.3f}"
                        elif abs(quantity) >= 0.1:
                            qty_display = f"{abs(quantity):.3f}"
                        else:
                            qty_display = f"{abs(quantity):.6f}"

                        position_text = f"{symbol} {display_side} {qty_display} @ ${entry_price:.0f}"
                        current_price_text = f"Live: ${current_price:.2f}"
                        pnl_text = f"P&L: ${net_pnl:+.2f}{fee_impact}"

                        self.safe_addstr(
                            y + 1 + displayed_positions, start_x, position_text
                        )
                        self.safe_addstr(
                            y + 1 + displayed_positions,
                            start_x + 25,
                            current_price_text,
                            curses.color_pair(4),
                        )
                        self.safe_addstr(
                            y + 2 + displayed_positions, start_x, pnl_text, pnl_color
                        )
                        displayed_positions += 2  # Use 2 lines per position now

            else:
                # Check config positions as fallback
                config_positions = self.config.get("open_positions", [])
                if config_positions:
                    self.safe_addstr(
                        y + 1, start_x, f"Config positions: {len(config_positions)}"
                    )
                    for i, pos in enumerate(config_positions[:3]):
                        symbol = pos.get("symbol", "N/A")
                        size = pos.get("size", 0)
                        entry_price = pos.get("entry_price", 0)
                        side = "LONG" if size > 0 else "SHORT" if size < 0 else "N/A"
                        # Format size for better readability
                        if abs(size) >= 1.0:
                            size_display = f"{abs(size):.3f}"
                        elif abs(size) >= 0.1:
                            size_display = f"{abs(size):.3f}"
                        else:
                            size_display = f"{abs(size):.6f}"
                        self.safe_addstr(
                            y + 2 + i,
                            start_x,
                            f"{symbol} {side} {size_display} @ ${entry_price:.2f}",
                        )
                else:
                    # Show clean message for no positions
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.safe_addstr(y + 1, start_x, f"No open positions ({timestamp})")

        except Exception as e:
            self.logger.warning(f"Error getting live positions: {e}")
            self.safe_addstr(y + 1, start_x, f"Position data error")

        # System Monitoring
        y += 13
        self.safe_addstr(
            y, start_x, "--- System Health ---", curses.color_pair(3) | curses.A_BOLD
        )
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        system_monitor = self.config.get("system_monitor")

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
        self.safe_addstr(
            y, start_x, "--- Performance ---", curses.color_pair(3) | curses.A_BOLD
        )
        performance_monitor = self.config.get("performance_monitor")
        if performance_monitor:
            try:
                sharpe = float(performance_monitor.calculate_rolling_sharpe() or 0.0)
                drawdown = float(
                    performance_monitor.calculate_rolling_drawdown() or 0.0
                )
                sortino = float(performance_monitor.calculate_sortino_ratio() or 0.0)
                calmar = float(performance_monitor.calculate_calmar_ratio() or 0.0)
                var = float(risk_manager.calculate_var() if risk_manager else 0.0)
                self.safe_addstr(y + 1, start_x, f"Sharpe: {sharpe:.2f}")
                self.safe_addstr(y + 2, start_x, f"Sortino: {sortino:.2f}")
                self.safe_addstr(y + 3, start_x, f"Calmar: {calmar:.2f}")
                self.safe_addstr(y + 4, start_x, f"Drawdown: {drawdown:.2%}")
                self.safe_addstr(y + 5, start_x, f"VaR (95%): ${var:,.2f}")
            except (ValueError, TypeError, AttributeError) as e:
                self.safe_addstr(y + 1, start_x, f"Performance data error")
                self.logger.debug(f"Performance metrics error: {e}")

        # Recent Trades (Live from Database) - Shows all trading activity including closed positions
        y += 3
        self.safe_addstr(
            y, start_x, "--- Recent Trades ---", curses.color_pair(3) | curses.A_BOLD
        )

        try:
            # Get recent trades directly from database and show meaningful ones
            from utils.paper_trade_db import get_all_trades

            all_recent_trades = get_all_trades(limit=50)  # Get more trades to filter

            # Separate profitable trades and recent activity
            profitable_trades = []
            recent_activity = []

            for trade in all_recent_trades:
                if len(trade) >= 7:
                    trade_data = {
                        "side": trade[3],
                        "symbol": trade[2],
                        "price": float(trade[4]),
                        "quantity": float(trade[5]),
                        "pnl": float(trade[6]),
                        "timestamp": trade[1],
                    }

                    # Collect trades with actual P&L
                    if trade_data["pnl"] != 0.0:
                        profitable_trades.append(trade_data)

                    # Also collect recent activity (last 10 trades)
                    if len(recent_activity) < 10:
                        recent_activity.append(trade_data)

            # Show profitable trades first, fallback to recent activity
            display_trades = (
                profitable_trades[:3] if profitable_trades else recent_activity[:3]
            )

            if display_trades:
                for i, trade in enumerate(display_trades):
                    side = trade["side"]
                    symbol = trade["symbol"]
                    price = trade["price"]
                    quantity = trade["quantity"]
                    pnl = trade["pnl"]

                    # Color code by side and P&L
                    side_color = (
                        curses.color_pair(1) if side == "BUY" else curses.color_pair(2)
                    )

                    # Show trade details
                    trade_info = f"{side} {quantity:.4f} {symbol} @ ${price:.2f}"
                    self.safe_addstr(y + 1 + i, start_x, trade_info, side_color)

                    # Show P&L or trade value for context
                    if pnl != 0.0:
                        pnl_color = (
                            curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                        )
                        # Add emoji to make profitable closures more visible
                        profit_icon = (
                            "💰" if pnl >= 0.10 else "💸" if pnl <= -5.0 else ""
                        )
                        self.safe_addstr(
                            y + 1 + i,
                            start_x + 25,
                            f"{profit_icon} ${pnl:+.2f}",
                            pnl_color,
                        )
                    else:
                        # Show trade value for position opens
                        trade_value = price * quantity
                        self.safe_addstr(
                            y + 1 + i,
                            start_x + 25,
                            f"${trade_value:.0f}",
                            curses.color_pair(7),
                        )

            else:
                self.safe_addstr(y + 1, start_x, "No recent trades")

        except Exception as e:
            self.logger.warning(f"Error getting live recent trades: {e}")
            self.safe_addstr(y + 1, start_x, "Trade data error")

        # Trade Distribution - Live from Database with comprehensive fee-adjusted stats
        y += 5
        self.safe_addstr(y, start_x, "--- Trade Statistics ---", curses.A_BOLD)

        # Try to get live trade statistics from database
        try:
            from utils.paper_trade_db import get_all_trades

            trades_raw = get_all_trades(
                limit=1000, exclude_test=True
            )  # Exclude TEST trades for accurate statistics

            if trades_raw:
                # Calculate comprehensive live statistics
                wins = 0
                losses = 0
                breakeven = 0
                total_pnl = 0.0
                total_volume = 0.0
                win_pnls = []
                loss_pnls = []

                # Count different types of trades for better analysis
                buy_trades = 0
                sell_trades = 0

                for trade in trades_raw:
                    if len(trade) >= 8:  # Include fee data
                        side = trade[3]
                        price = float(trade[4])
                        quantity = float(trade[5])
                        pnl = float(trade[6])
                        fee = float(trade[7])

                        # Calculate trade volume
                        trade_volume = price * quantity
                        total_volume += trade_volume

                        # Count trade types
                        if side == "BUY":
                            buy_trades += 1
                        elif side == "SELL":
                            sell_trades += 1

                        # Count all trades for P&L analysis (fees already included in stored P&L)
                        if side in ["BUY", "SELL"]:
                            # Use stored P&L directly (fees already factored in)
                            net_pnl = pnl

                            # Count all real trades for total P&L
                            total_pnl += net_pnl

                            # Analyze both BUY and SELL trades for win/loss (scalping strategy)
                            if side in ["BUY", "SELL"]:
                                # P&L already includes fees, so use directly
                                # FIXED: Lower thresholds for smaller position sizes and include non-zero trades
                                if (
                                    net_pnl > 0.01
                                ):  # Profitable trade (lowered from 0.10 to 0.01)
                                    wins += 1
                                    win_pnls.append(net_pnl)
                                elif (
                                    net_pnl < -0.01
                                ):  # Loss trade (lowered from -0.10 to -0.01)
                                    losses += 1
                                    loss_pnls.append(net_pnl)
                                else:  # Small result (breakeven or zero P&L opening trades)
                                    breakeven += 1

                # Calculate statistics
                completed_trades = wins + losses + breakeven
                total_trades = len(trades_raw)
                win_rate = (
                    (wins / completed_trades * 100) if completed_trades > 0 else 0.0
                )
                avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
                avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
                avg_trade_value = (
                    total_volume / total_trades if total_trades > 0 else 0.0
                )

                # Profit factor calculation
                total_wins = sum(win_pnls) if win_pnls else 0.0
                total_losses = abs(sum(loss_pnls)) if loss_pnls else 0.0
                profit_factor = (
                    total_wins / total_losses
                    if total_losses > 0
                    else float("inf") if total_wins > 0 else 0.0
                )

                # Show comprehensive trade breakdown (already filtered, so no TEST trades)
                total_trades = len(trades_raw)

                self.safe_addstr(y + 1, start_x, f"Paper Trades: {total_trades}")

                # Show win rate with realistic assessment
                trading_outcomes = wins + losses  # Exclude breakeven for win rate
                if trading_outcomes > 0:
                    meaningful_win_rate = wins / trading_outcomes * 100
                    win_rate_color = (
                        curses.color_pair(1)
                        if meaningful_win_rate > 50
                        else (
                            curses.color_pair(6)
                            if meaningful_win_rate > 30
                            else curses.color_pair(2)
                        )
                    )
                    self.safe_addstr(
                        y + 2,
                        start_x,
                        f"Win Rate: {meaningful_win_rate:.1f}% ({wins}W/{losses}L)",
                        win_rate_color,
                    )
                else:
                    # More informative message about why winrate is N/A
                    if total_trades > 0:
                        self.safe_addstr(
                            y + 2,
                            start_x,
                            f"Win Rate: N/A ({total_trades} opens, 0 closed)",
                            curses.color_pair(6),
                        )
                    else:
                        self.safe_addstr(
                            y + 2,
                            start_x,
                            "Win Rate: N/A (no trades yet)",
                            curses.color_pair(6),
                        )

                self.safe_addstr(
                    y + 3,
                    start_x,
                    f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}",
                )

                # Color code total P&L
                pnl_color = (
                    curses.color_pair(1) if total_pnl >= 0 else curses.color_pair(2)
                )
                self.safe_addstr(
                    y + 4, start_x, f"Net P&L: ${total_pnl:.2f}", pnl_color
                )

                # Show profit factor with meaningful color coding
                pf_color = (
                    curses.color_pair(1)
                    if profit_factor > 1.5
                    else (
                        curses.color_pair(6)
                        if profit_factor > 1.0
                        else curses.color_pair(2)
                    )
                )
                profit_factor_display = (
                    f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
                )
                self.safe_addstr(
                    y + 5, start_x, f"Profit Factor: {profit_factor_display}", pf_color
                )

                self.safe_addstr(
                    y + 6,
                    start_x,
                    f"Avg Trade: ${avg_trade_value:.0f} | BE: {breakeven}",
                )
            else:
                self.safe_addstr(y + 1, start_x, "No trades yet")

        except Exception as e:
            # Fallback to trade analyzer
            trade_analyzer = self.config.get("trade_analyzer")
            recent_trades = self.config.get("recent_trades", [])
            if trade_analyzer:
                try:
                    win_loss = trade_analyzer.get_win_loss_distribution()
                    avg_pnl = trade_analyzer.get_average_pnl()
                    wins = int(win_loss.get("wins", 0))
                    losses = int(win_loss.get("losses", 0))
                    avg_win = float(avg_pnl.get("avg_win", 0.0))
                    avg_loss = float(avg_pnl.get("avg_loss", 0.0))
                    self.safe_addstr(y + 1, start_x, f"Wins: {wins} | Losses: {losses}")
                    self.safe_addstr(
                        y + 2,
                        start_x,
                        f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}",
                    )
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    self.safe_addstr(y + 1, start_x, f"Trade data error")
                    self.logger.debug(f"Trade analyzer error: {e}")
            elif recent_trades:
                # Calculate basic stats from recent trades if trade_analyzer not available
                wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
                losses = len(recent_trades) - wins
                total_pnl = sum(t.get("pnl", 0) for t in recent_trades)
                self.safe_addstr(y + 1, start_x, f"Wins: {wins} | Losses: {losses}")
                self.safe_addstr(y + 2, start_x, f"Total PnL: ${total_pnl:.2f}")
            else:
                self.safe_addstr(y + 1, start_x, "No trade data available")

        # Position-level Risk
        y += 6
        self.safe_addstr(y, start_x, "--- Position Risk ---", curses.A_BOLD)

        # Get current portfolio and risk data
        portfolio_value = float(self.config.get("portfolio_value", 10000.0))
        open_positions = self.config.get("open_positions", [])

        if open_positions:
            total_position_value = 0
            for i, pos in enumerate(open_positions):  # Show all positions
                symbol = pos.get("symbol", "N/A")
                size = pos.get("size", 0)
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", entry_price)

                # Calculate position value as risk
                position_value = abs(size) * current_price
                total_position_value += position_value

                # Calculate percentage of portfolio
                position_pct = (
                    (position_value / portfolio_value * 100)
                    if portfolio_value > 0
                    else 0
                )

                self.safe_addstr(
                    y + 1 + i,
                    start_x,
                    f"{symbol}: ${position_value:,.2f} ({position_pct:.1f}%)",
                )

            # Show total exposure
            total_exposure_pct = (
                (total_position_value / portfolio_value * 100)
                if portfolio_value > 0
                else 0
            )
            self.safe_addstr(
                y + 4,
                start_x,
                f"Total Exposure: ${total_position_value:,.2f} ({total_exposure_pct:.1f}%)",
            )
        else:
            # Show risk metrics even without open positions
            try:
                # Get recent trading activity for risk assessment
                from utils.paper_trade_db import get_all_trades

                recent_trades = get_all_trades(limit=20)

                if recent_trades:
                    # Calculate recent risk metrics
                    recent_volumes = []
                    recent_pnls = []

                    for trade in recent_trades:
                        if len(trade) >= 7:
                            volume = float(trade[4]) * float(
                                trade[5]
                            )  # price * quantity
                            pnl = float(trade[6])
                            recent_volumes.append(volume)
                            recent_pnls.append(pnl)

                    if recent_volumes:
                        avg_trade_size = sum(recent_volumes) / len(recent_volumes)
                        max_trade_size = max(recent_volumes)

                        # Calculate risk per trade as percentage of portfolio
                        avg_risk_pct = (
                            (avg_trade_size / portfolio_value * 100)
                            if portfolio_value > 0
                            else 0
                        )
                        max_risk_pct = (
                            (max_trade_size / portfolio_value * 100)
                            if portfolio_value > 0
                            else 0
                        )

                        self.safe_addstr(y + 1, start_x, f"No open positions")
                        self.safe_addstr(
                            y + 2,
                            start_x,
                            f"Avg Trade Size: ${avg_trade_size:,.2f} ({avg_risk_pct:.1f}%)",
                        )
                        self.safe_addstr(
                            y + 3,
                            start_x,
                            f"Max Trade Size: ${max_trade_size:,.2f} ({max_risk_pct:.1f}%)",
                        )

                        # Calculate volatility from recent PnLs
                        if len(recent_pnls) > 1:
                            import statistics

                            pnl_std = (
                                statistics.stdev(recent_pnls)
                                if len(recent_pnls) > 1
                                else 0
                            )
                            self.safe_addstr(
                                y + 4, start_x, f"PnL Volatility: ${pnl_std:.2f}"
                            )
                    else:
                        self.safe_addstr(y + 1, start_x, "No recent trading data")
                else:
                    self.safe_addstr(y + 1, start_x, "No positions or trades")

            except Exception as e:
                self.safe_addstr(y + 1, start_x, f"Risk data unavailable")
                self.logger.debug(f"Risk calculation error: {e}")

    def _draw_api_status_pane(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draws the API connection status widget with visual indicators."""
        self.safe_addstr(start_y, start_x, "--- API Status ---", curses.A_BOLD)

        try:
            y = start_y + 1

            # Check if API tester exists and is properly initialized
            if not hasattr(self, "api_tester") or self.api_tester is None:
                self.safe_addstr(
                    y, start_x, "API tester not initialized", curses.color_pair(2)
                )
                return

            # Get current API statuses with safety checks
            try:
                api_statuses = getattr(self.api_tester, "api_statuses", None)
                if api_statuses is None:
                    self.safe_addstr(
                        y, start_x, "API statuses None", curses.color_pair(2)
                    )
                    return

                overall_health = self.api_tester.get_overall_health()
                if overall_health is None:
                    self.safe_addstr(
                        y, start_x, "Health data None", curses.color_pair(2)
                    )
                    return

            except AttributeError as e:
                self.safe_addstr(
                    y,
                    start_x,
                    f"Missing attribute: {str(e)[:10]}",
                    curses.color_pair(2),
                )
                return
            except Exception as e:
                self.safe_addstr(
                    y, start_x, f"API error: {str(e)[:10]}", curses.color_pair(2)
                )
                return

            # Check if statuses are valid
            if not api_statuses or not isinstance(api_statuses, dict):
                self.safe_addstr(y, start_x, "No API status data", curses.color_pair(6))
                return

            # Debug: Log Vault status if it's not connected
            vault_status = api_statuses.get("vault")
            if vault_status and vault_status.status.value != "connected":
                self.logger.debug(
                    f"Vault status in dashboard: {vault_status.status.value}, error: {vault_status.error_message}"
                )

            # Show overall health first
            if overall_health and isinstance(overall_health, dict):
                health_status = overall_health.get("overall_status", "unknown")
                health_percentage = overall_health.get("health_percentage", 0)

                if health_status == "healthy":
                    health_color = curses.color_pair(1)  # Green
                    health_icon = "✅"
                elif health_status == "warning":
                    health_color = curses.color_pair(3)  # Yellow
                    health_icon = "⚠️"
                else:
                    health_color = curses.color_pair(2)  # Red
                    health_icon = "❌"

                self.safe_addstr(
                    y,
                    start_x,
                    f"{health_icon} Overall: {health_percentage:.0f}%",
                    health_color,
                )
            else:
                self.safe_addstr(
                    y, start_x, "❓ Overall: Calculating...", curses.color_pair(6)
                )
            y += 1

            # Show critical services status
            critical_apis = ["binance_spot", "binance_futures", "postgres", "vault"]

            for api_name in critical_apis:
                if y >= start_y + height - 2:  # Prevent overflow
                    break

                try:
                    if api_name in api_statuses:
                        status = api_statuses[api_name]

                        # Check if status object is valid
                        if (
                            not status
                            or not hasattr(status, "status")
                            or not hasattr(status.status, "value")
                        ):
                            indicator = "❓"
                            color = curses.color_pair(6)
                            display_name = api_name.replace("_", " ").title()[:12]
                            response_time = ""
                        else:
                            # Visual indicator based on status
                            status_value = getattr(status.status, "value", "unknown")
                            if status_value == "connected":
                                indicator = "✅"
                                color = curses.color_pair(1)  # Green
                            elif status_value == "testing":
                                indicator = "⏳"
                                color = curses.color_pair(3)  # Yellow
                            elif status_value == "error":
                                indicator = "❌"
                                color = curses.color_pair(2)  # Red
                            else:
                                indicator = "❌"
                                color = curses.color_pair(6)  # White

                            # Format API name for display
                            display_name = api_name.replace("_", " ").title()
                            if len(display_name) > 12:
                                display_name = display_name[:12]

                            # Show response time if available
                            response_time = ""
                            if (
                                hasattr(status, "response_time")
                                and status.response_time
                                and status.response_time > 0
                            ):
                                try:
                                    response_time = f"{status.response_time*1000:.0f}ms"
                                except (TypeError, ValueError):
                                    response_time = ""

                        self.safe_addstr(
                            y, start_x, f"{indicator} {display_name:<12}", color
                        )
                        if response_time and len(response_time) < 8:
                            self.safe_addstr(
                                y, start_x + 15, response_time, curses.color_pair(6)
                            )
                    else:
                        # API not found in statuses
                        display_name = api_name.replace("_", " ").title()[:12]
                        self.safe_addstr(
                            y, start_x, f"❓ {display_name:<12}", curses.color_pair(6)
                        )

                    y += 1

                except Exception as api_error:
                    # Log individual API errors but continue
                    self.logger.debug(f"Error displaying {api_name}: {api_error}")
                    self.safe_addstr(
                        y, start_x, f"❌ {api_name[:12]:<12}", curses.color_pair(2)
                    )
                    y += 1

            # Show secondary services in compact format
            y += 1
            secondary_apis = ["binance_testnet", "redis", "telegram", "prometheus"]
            secondary_status = []

            for api_name in secondary_apis:
                if api_name in api_statuses:
                    status = api_statuses[api_name]
                    if not status or not hasattr(status, "status"):
                        secondary_status.append(f"{api_name[:3].upper()}❓")
                    elif status.status.value == "connected":
                        secondary_status.append(f"{api_name[:3].upper()}✅")
                    elif status.status.value == "testing":
                        secondary_status.append(f"{api_name[:3].upper()}⏳")
                    else:
                        secondary_status.append(f"{api_name[:3].upper()}❌")

            if secondary_status:
                compact_status = " ".join(secondary_status)
                self.safe_addstr(
                    y, start_x, f"Other: {compact_status}", curses.color_pair(6)
                )
                y += 1

            # Show last update time
            if api_statuses:
                valid_statuses = [
                    s for s in api_statuses.values() if s and s.last_checked
                ]
                if valid_statuses:
                    last_checked = max(
                        valid_statuses, key=lambda x: x.last_checked
                    ).last_checked
                    time_str = last_checked.strftime("%H:%M:%S")
                    self.safe_addstr(
                        y, start_x, f"Updated: {time_str}", curses.color_pair(6)
                    )
                else:
                    self.safe_addstr(
                        y, start_x, "Initializing...", curses.color_pair(6)
                    )

        except Exception as e:
            self.safe_addstr(start_y + 1, start_x, f"API status error: {str(e)[:20]}")
            self.logger.debug(f"API status widget error: {e}")

    def _draw_position_sizing_pane(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draws the live position sizing visualization pane with real-time data."""
        self.safe_addstr(
            start_y, start_x, "--- Live Position Sizing ---", curses.A_BOLD
        )

        try:
            y = start_y + 2

            # Get live market data
            current_price = 107000.0  # Default fallback
            if self.price_fetcher:
                try:
                    live_price = self.price_fetcher.get_current_price("BTCUSDT")
                    if live_price and live_price > 0:
                        current_price = float(live_price)
                except Exception as e:
                    self.logger.debug(f"Live price fetch error: {e}")

            # Get live portfolio value from database - use reset-based calculation
            try:
                from utils.paper_trade_db import get_conn

                realized_pnl = 0.0

                try:
                    conn = get_conn()
                    if conn:
                        with conn.cursor() as c:
                            # Get the latest reset timestamp
                            c.execute("""
                                SELECT reset_timestamp FROM trading_session_resets 
                                ORDER BY reset_timestamp DESC LIMIT 1
                            """)
                            reset_result = c.fetchone()

                            if reset_result:
                                reset_timestamp = reset_result[0]
                                # Get P&L from trades after reset only
                                c.execute(
                                    """
                                    SELECT COALESCE(SUM(pnl), 0) FROM trades 
                                    WHERE timestamp >= %s
                                    AND pnl BETWEEN -100 AND 100
                                """,
                                    (reset_timestamp,),
                                )
                                result = c.fetchone()
                                if result:
                                    realized_pnl = (
                                        float(result[0])
                                        if result[0] is not None
                                        else 0.0
                                    )
                            else:
                                # No reset found, default to 0 (fresh start)
                                realized_pnl = 0.0
                        conn.close()

                    # Cap P&L to reasonable range
                    realized_pnl = max(-1000.0, min(1000.0, realized_pnl))

                except Exception as db_e:
                    realized_pnl = 0.0

                # Calculate live portfolio value from LIVE executor balance
                try:
                    from core.di import container

                    executor = container.get_optional("executor")
                    if executor and hasattr(executor, "get_balance"):
                        balance_info = executor.get_balance()
                        starting_value = float(balance_info.get("USDT", 1000.0))
                    else:
                        starting_value = 1000.0  # Fallback if no executor
                except:
                    starting_value = 1000.0  # Fallback
                portfolio_value = starting_value + realized_pnl

                # Add unrealized PnL from open positions
                from utils.paper_trade_db import get_open_positions

                open_positions_raw = get_open_positions()

                unrealized_pnl = 0.0
                total_position_value = 0.0

                for pos in open_positions_raw:
                    if (
                        len(pos) >= 5
                    ):  # Need at least 5 fields: id, symbol, side, entry_price, quantity
                        try:
                            symbol = pos[1]
                            side = pos[2]  # BUY or SELL
                            entry_price = float(pos[3])  # Correct index for entry_price
                            quantity = float(pos[4])  # Correct index for quantity

                            # Get live price for this position
                            live_pos_price = entry_price  # Fallback
                            if self.price_fetcher:
                                try:
                                    fetched_price = (
                                        self.price_fetcher.get_current_price(symbol)
                                    )
                                    if fetched_price and fetched_price > 0:
                                        live_pos_price = float(fetched_price)
                                except:
                                    pass

                            # Calculate position value and P&L
                            position_value = abs(quantity) * live_pos_price
                            total_position_value += position_value

                            # Calculate P&L based on position side
                            if side.upper() == "BUY":  # LONG position
                                position_pnl = (live_pos_price - entry_price) * quantity
                            else:  # SHORT position
                                position_pnl = (entry_price - live_pos_price) * quantity
                            unrealized_pnl += position_pnl

                        except Exception as e:
                            self.logger.debug(f"Position value calculation error: {e}")

                # Add unrealized P&L to portfolio value
                portfolio_value += unrealized_pnl

            except Exception as e:
                # Use minimal fallback - avoid stale config data
                portfolio_value = 1000.0  # Starting amount
                total_position_value = 0.0
                self.logger.debug(f"Live portfolio calculation error: {e}")

            # Get aggressive leverage settings (100x maximum power)
            leverage = int(self.config.get("leverage", 100))  # Default to 100x leverage

            # Calculate live market volatility (simple estimate)
            volatility = 0.02  # Default 2%
            if self.price_fetcher:
                try:
                    # Get recent price data for volatility calculation
                    klines = self.price_fetcher.get_historical_klines(
                        "BTCUSDT", "1m", limit=20
                    )
                    if not klines.empty and "close" in klines.columns:
                        closes = klines["close"].astype(float)
                        if len(closes) > 1:
                            returns = closes.pct_change().dropna()
                            volatility = returns.std() * (
                                60**0.5
                            )  # Annualized hourly volatility
                except Exception as e:
                    self.logger.debug(f"Volatility calculation error: {e}")

            # Display live market data with aggressive parameters
            time_str = datetime.now().strftime("%H:%M:%S")
            self.safe_addstr(
                y,
                start_x,
                f"Live Price: ${current_price:,.2f} [{time_str}]",
                curses.color_pair(4),
            )
            self.safe_addstr(
                y + 1,
                start_x,
                f"Portfolio: ${portfolio_value:,.2f}",
                curses.color_pair(6),
            )
            self.safe_addstr(
                y + 2,
                start_x,
                f"Leverage: {leverage}x | Vol: {volatility*100:.1f}%",
                curses.color_pair(6),
            )

            y += 4

            # Show live position utilization with higher risk tolerance
            if total_position_value > 0:
                utilization = (
                    (total_position_value / portfolio_value) * 100
                    if portfolio_value > 0
                    else 0
                )
                util_color = (
                    curses.color_pair(1)
                    if utilization < 70
                    else (
                        curses.color_pair(3)
                        if utilization < 90
                        else curses.color_pair(2)
                    )
                )
                self.safe_addstr(
                    y, start_x, f"Position Utilization: {utilization:.1f}%", util_color
                )

                # Create utilization bar
                max_bar_width = width - 20
                bar_width = max(
                    0, min(max_bar_width, int(utilization / 100 * max_bar_width))
                )
                remaining_width = max(0, max_bar_width - bar_width)
                bar_str = "█" * bar_width + "░" * remaining_width
                self.safe_addstr(y + 1, start_x, f"[{bar_str}]", util_color)
                y += 3
            else:
                self.safe_addstr(
                    y, start_x, "No active positions", curses.color_pair(6)
                )
                y += 2

            # Aggressive risk-based position sizing with high leverage
            self.safe_addstr(y, start_x, "Aggressive Position Sizing:", curses.A_BOLD)
            y += 1

            # More aggressive risk levels for high leverage trading
            base_risks = [0.02, 0.05, 0.10]  # 2%, 5%, 10% risk levels

            for i, base_risk in enumerate(base_risks):
                # Use full risk without volatility adjustment for aggressive trading
                risk_amount = portfolio_value * base_risk
                position_value = risk_amount * leverage
                position_size = position_value / current_price

                # Color code based on risk level (more tolerant for aggressive trading)
                risk_color = (
                    curses.color_pair(1)
                    if base_risk < 0.05
                    else (
                        curses.color_pair(3)
                        if base_risk < 0.08
                        else curses.color_pair(2)
                    )
                )

                self.safe_addstr(
                    y,
                    start_x,
                    f"{base_risk*100:.1f}%: {position_size:.4f} BTC",
                    risk_color,
                )
                self.safe_addstr(
                    y, start_x + 25, f"(${position_value:,.0f})", risk_color
                )
                y += 1

            y += 1

            # Fixed take-profit at 1.5% as requested
            self.safe_addstr(y, start_x, "Risk Management (Fixed):", curses.A_BOLD)
            y += 1

            # Fixed 1.5% take profit and aggressive stop loss
            take_profit_pct = 0.015  # Fixed 1.5% take profit
            stop_loss_pct = 0.01  # Aggressive 1% stop loss

            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            self.safe_addstr(
                y,
                start_x,
                f"Stop Loss: ${stop_loss_price:,.0f} (-{stop_loss_pct*100:.1f}%)",
                curses.color_pair(2),
            )
            self.safe_addstr(
                y + 1,
                start_x,
                f"Take Profit: ${take_profit_price:,.0f} (+{take_profit_pct*100:.1f}%)",
                curses.color_pair(1),
            )

            y += 3

            # High leverage margin and liquidation info
            if leverage > 1:
                self.safe_addstr(y, start_x, "High Leverage Margin:", curses.A_BOLD)
                y += 1

                # Calculate liquidation price for high leverage
                margin_ratio = 1 / leverage
                liquidation_distance = (
                    margin_ratio * 0.9
                )  # 90% of margin before liquidation
                liquidation_price = current_price * (1 - liquidation_distance)

                # Color code based on liquidation risk (more aggressive thresholds)
                liq_color = (
                    curses.color_pair(1)
                    if liquidation_distance > 0.02
                    else (
                        curses.color_pair(3)
                        if liquidation_distance > 0.01
                        else curses.color_pair(2)
                    )
                )

                self.safe_addstr(
                    y, start_x, f"Liquidation: ${liquidation_price:,.0f}", liq_color
                )
                self.safe_addstr(
                    y, start_x + 25, f"(-{liquidation_distance*100:.2f}%)", liq_color
                )

                # Show margin usage with high leverage tolerance
                if total_position_value > 0:
                    margin_used = total_position_value / leverage
                    margin_available = portfolio_value - margin_used
                    margin_usage_pct = (
                        (margin_used / portfolio_value) * 100
                        if portfolio_value > 0
                        else 0
                    )

                    margin_color = (
                        curses.color_pair(1)
                        if margin_usage_pct < 50
                        else (
                            curses.color_pair(3)
                            if margin_usage_pct < 80
                            else curses.color_pair(2)
                        )
                    )
                    self.safe_addstr(
                        y + 1,
                        start_x,
                        f"Margin Used: {margin_usage_pct:.1f}%",
                        margin_color,
                    )
                    self.safe_addstr(
                        y + 2,
                        start_x,
                        f"Available: ${margin_available:,.0f}",
                        margin_color,
                    )

        except Exception as e:
            self.safe_addstr(start_y + 2, start_x, f"Live sizing error: {str(e)[:20]}")
            self.logger.debug(f"Live position sizing pane error: {e}")
            import traceback

            self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def _draw_chart_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the candlestick chart pane with OHLC data and technical indicators."""
        self.safe_addstr(
            start_y,
            start_x,
            f"--- BTC/USDT Candlestick Chart ({self.timeframe}) ---",
            curses.color_pair(3) | curses.A_BOLD,
        )

        # Get real OHLC data from price fetcher if available
        current_price = float(self.config.get("current_price", 97000.0))
        ohlc_data = None

        # Try to get real data from price fetcher
        if self.price_fetcher:
            try:
                data = self.price_fetcher.get_historical_klines(
                    "BTCUSDT", "1m", limit=40
                )
                if not data.empty and all(
                    col in data.columns for col in ["open", "high", "low", "close"]
                ):
                    ohlc_data = data[["open", "high", "low", "close"]].tail(40)
                    current_price = (
                        ohlc_data["close"].iloc[-1]
                        if not ohlc_data.empty
                        else current_price
                    )
            except Exception as e:
                self.logger.debug(f"Could not fetch real OHLC data: {e}")

        # Get from config if available
        if ohlc_data is None and "ohlc_data" in self.config:
            ohlc_data = self.config["ohlc_data"]

        # Ensure any OHLC data is converted to proper numeric types early
        if ohlc_data is not None and not ohlc_data.empty:
            try:
                # Force all OHLC columns to be numeric, converting any strings
                for col in ["open", "high", "low", "close"]:
                    if col in ohlc_data.columns:
                        ohlc_data[col] = pd.to_numeric(ohlc_data[col], errors="coerce")
                # Drop any rows with NaN values after conversion
                ohlc_data = ohlc_data.dropna()
                self.logger.debug(
                    f"OHLC data after early type conversion: {ohlc_data.dtypes.to_dict()}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to convert OHLC data types: {e}")
                ohlc_data = None

        # Create mock OHLC data if none available
        if ohlc_data is None or len(ohlc_data) == 0:
            import numpy as np

            np.random.seed(42)  # Consistent mock data
            base_price = current_price
            mock_data = []

            for i in range(40):
                # Create realistic OHLC candles
                open_price = float(base_price)
                change_pct = np.random.normal(0, 0.002)  # 0.2% volatility
                close_price = float(open_price * (1 + change_pct))

                # High and low based on volatility
                volatility = abs(change_pct) + np.random.uniform(0.001, 0.005)
                high_price = float(max(open_price, close_price) * (1 + volatility / 2))
                low_price = float(min(open_price, close_price) * (1 - volatility / 2))

                mock_data.append(
                    {
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "close": float(close_price),
                    }
                )
                base_price = close_price

            ohlc_data = pd.DataFrame(mock_data)

        if ohlc_data is None or len(ohlc_data) < 2:
            self.safe_addstr(
                start_y + 2,
                start_x,
                "Waiting for candlestick data...",
                curses.color_pair(3),
            )
            return

        # Chart dimensions
        chart_height = height - 5
        chart_width = min(
            width - 2, len(ohlc_data)
        )  # Minimal margin for maximum chart width within pane

        # Use the most recent candles that fit
        candles = ohlc_data.tail(chart_width)
        if len(candles) < 2:
            return

        # Calculate price range for scaling - ensure numeric types
        try:
            # Ensure candles are already properly typed from early conversion above
            if len(candles) == 0:
                self.safe_addstr(
                    start_y + 2,
                    start_x,
                    "Invalid price data format",
                    curses.color_pair(2),
                )
                return

            all_prices = pd.concat([candles["high"], candles["low"]]).dropna()
            if len(all_prices) == 0:
                self.safe_addstr(
                    start_y + 2, start_x, "No valid price data", curses.color_pair(2)
                )
                return

            min_price, max_price = float(all_prices.min()), float(all_prices.max())
            price_range = max_price - min_price if max_price > min_price else 1.0

            # Validate the calculated values
            if not all(
                isinstance(x, (int, float)) and not pd.isna(x)
                for x in [min_price, max_price, price_range]
            ):
                self.safe_addstr(
                    start_y + 2,
                    start_x,
                    "Price calculation error",
                    curses.color_pair(2),
                )
                return

        except Exception as e:
            self.logger.error(f"Error processing price data: {e}")
            self.safe_addstr(
                start_y + 2, start_x, f"Data error: {str(e)[:30]}", curses.color_pair(2)
            )
            return

        # Update chart more frequently for real-time feel
        try:
            last_close = float(candles["close"].iloc[-1])
            # Create a price string that updates every $1 change for more responsive chart
            stable_close = round(
                last_close, 0
            )  # Round to nearest dollar (was nearest 10)
            price_str = (
                f"{min_price:.0f}-{max_price:.0f}-{len(candles)}-{stable_close:.0f}"
            )
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error creating price string: {e}")
            return

        # Cache chart buffer to reduce redrawing
        if self.last_chart_data != price_str or price_str not in self.chart_buffer:
            self.last_chart_data = price_str

            # Draw candlesticks
            for i, (idx, candle) in enumerate(candles.iterrows()):
                if i >= chart_width:
                    break

                # Ensure numeric values with validation
                try:
                    open_price = float(candle["open"])
                    high_price = float(candle["high"])
                    low_price = float(candle["low"])
                    close_price = float(candle["close"])

                    # Validate that all values are valid numbers
                    if any(
                        pd.isna(x) or not isinstance(x, (int, float))
                        for x in [open_price, high_price, low_price, close_price]
                    ):
                        continue  # Skip this candle if invalid data

                except (ValueError, TypeError, KeyError):
                    continue  # Skip this candle if conversion fails

                # Calculate Y positions (inverted because screen coordinates)
                def price_to_y(price):
                    try:
                        price = float(price)  # Ensure price is numeric
                        normalized = (price - min_price) / price_range
                        return int(
                            start_y
                            + 2
                            + chart_height
                            - 1
                            - (normalized * (chart_height - 1))
                        )
                    except (ValueError, TypeError, ZeroDivisionError):
                        return start_y + 2  # Safe fallback position

                open_y = price_to_y(open_price)
                high_y = price_to_y(high_price)
                low_y = price_to_y(low_price)
                close_y = price_to_y(close_price)

                # Ensure positions are within bounds
                open_y = max(start_y + 2, min(open_y, start_y + chart_height))
                high_y = max(start_y + 2, min(high_y, start_y + chart_height))
                low_y = max(start_y + 2, min(low_y, start_y + chart_height))
                close_y = max(start_y + 2, min(close_y, start_y + chart_height))

                candle_x = start_x + 2 + i

                # Determine candle color (green for up, red for down)
                is_bullish = close_price >= open_price
                candle_color = (
                    curses.color_pair(1) if is_bullish else curses.color_pair(2)
                )

                # Draw the wick (high-low line)
                for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
                    self.safe_addch(y, candle_x, "|", curses.color_pair(6))

                # Draw the body (open-close rectangle)
                body_top = min(open_y, close_y)
                body_bottom = max(open_y, close_y)

                if body_top == body_bottom:  # Doji (open == close)
                    self.safe_addch(
                        body_top, candle_x, "-", candle_color | curses.A_BOLD
                    )
                else:
                    # Draw body with different characters for bullish/bearish
                    body_char = (
                        "█" if is_bullish else "▓"
                    )  # Solid for bullish, lighter for bearish
                    for y in range(body_top, body_bottom + 1):
                        self.safe_addch(
                            y, candle_x, body_char, candle_color | curses.A_BOLD
                        )

                # Mark open and close levels
                self.safe_addch(open_y, candle_x, "○", candle_color)  # Open
                self.safe_addch(
                    close_y, candle_x, "●", candle_color | curses.A_BOLD
                )  # Close

            # Draw price scale on the right
            scale_x = start_x + chart_width + 4
            num_levels = min(8, chart_height // 2)
            for i in range(num_levels):
                if num_levels > 1:
                    scale_price = max_price - (i / (num_levels - 1)) * price_range
                else:
                    scale_price = (max_price + min_price) / 2
                scale_y = (
                    start_y + 2 + int(i * (chart_height - 1) / max(1, num_levels - 1))
                )
                scale_y = max(start_y + 2, min(scale_y, start_y + chart_height - 1))
                self.safe_addstr(
                    scale_y, scale_x, f"${scale_price:.0f}", curses.color_pair(6)
                )

            # Draw volume bars at the bottom (simplified)
            volume_y = start_y + chart_height + 1
            self.safe_addstr(volume_y - 1, start_x, "Volume:", curses.color_pair(6))
            if "volume" in candles.columns:
                try:
                    max_vol = float(candles["volume"].max())
                    if max_vol > 0:
                        for i, (idx, candle) in enumerate(candles.iterrows()):
                            if i >= chart_width:
                                break
                            try:
                                volume = float(candle["volume"])
                                vol_height = max(1, min(3, int((volume / max_vol) * 3)))
                                candle_x = start_x + 2 + i
                                for h in range(vol_height):
                                    self.safe_addch(
                                        volume_y + h,
                                        candle_x,
                                        " ",
                                        curses.color_pair(4),
                                    )
                            except (ValueError, TypeError):
                                continue
                except (ValueError, TypeError):
                    pass  # Skip volume rendering if data is invalid

        # Draw technical indicators at the bottom
        indicator_y = start_y + height - 2

        # Get indicators from config or calculate simple ones
        rsi_val = 50.0
        macd_val = 0.0
        sma_val = current_price

        if "indicators" in self.config:
            indicators = self.config["indicators"]
            rsi_val = indicators.get("rsi", 50.0)
            macd_val = indicators.get("macd", 0.0)
            sma_val = indicators.get("sma_20", current_price)
        elif not ohlc_data.empty:
            # Calculate simple indicators
            closes = ohlc_data["close"]
            if len(closes) >= 14:
                # Simple RSI calculation
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_val = (
                    100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50
                )

            if len(closes) >= 20:
                sma_val = closes.rolling(window=20).mean().iloc[-1]

        # Color code indicators with type safety
        try:
            rsi_val = float(rsi_val)
            macd_val = float(macd_val)
            sma_val = float(sma_val)
            current_price = float(current_price)

            rsi_color = (
                curses.color_pair(1) if 30 <= rsi_val <= 70 else curses.color_pair(2)
            )
            macd_color = curses.color_pair(1) if macd_val > 0 else curses.color_pair(2)
            price_change = (
                ((current_price - sma_val) / sma_val * 100) if sma_val > 0 else 0
            )
            price_color = (
                curses.color_pair(1) if price_change >= 0 else curses.color_pair(2)
            )
        except (ValueError, TypeError, ZeroDivisionError):
            rsi_color = curses.color_pair(6)
            macd_color = curses.color_pair(6)
            price_color = curses.color_pair(6)
            price_change = 0

        self.safe_addstr(indicator_y, start_x, f"RSI: {rsi_val:.1f}", rsi_color)
        self.safe_addstr(indicator_y, start_x + 12, f"MACD: {macd_val:.3f}", macd_color)
        self.safe_addstr(
            indicator_y, start_x + 25, f"Price: ${current_price:.2f}", price_color
        )
        self.safe_addstr(
            indicator_y, start_x + 42, f"SMA20: ${sma_val:.0f}", curses.color_pair(6)
        )

    def _draw_console_messages(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draw console messages at the bottom of the screen."""
        self.safe_addstr(
            start_y,
            start_x,
            "--- Live Trading Logs ---",
            curses.color_pair(3) | curses.A_BOLD,
        )

        # Get recent log messages
        display_messages = (
            list(self.log_messages)[-height + 2 :]
            if len(self.log_messages) > height - 2
            else list(self.log_messages)
        )

        if not display_messages:
            # Show waiting message if no logs yet
            self.safe_addstr(
                start_y + 1,
                start_x,
                "Waiting for trading logs...",
                curses.color_pair(6),
            )
        else:
            for i, message in enumerate(display_messages):
                if i < height - 2:
                    # Truncate message if too long and add color based on content
                    display_text = (
                        message[: width - 2] if len(message) > width - 2 else message
                    )

                    # Color code based on log level or content
                    color = curses.color_pair(6)  # Default white
                    if "ERROR" in message or "Failed" in message:
                        color = curses.color_pair(2)  # Red
                    elif "WARNING" in message or "Warning" in message:
                        color = curses.color_pair(3)  # Yellow
                    elif (
                        "BUY" in message
                        or "SELL" in message
                        or "order executed" in message
                    ):
                        color = curses.color_pair(1)  # Green
                    elif "signal" in message.lower() or "prediction" in message.lower():
                        color = curses.color_pair(4)  # Cyan

                    self.safe_addstr(start_y + 1 + i, start_x, display_text, color)

    def add_message(self, message: str):
        """Add a message to the console messages."""
        if not hasattr(self, "messages"):
            self.messages = []
        self.messages.append(message)
        # Keep only last 50 messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

        # Also add to config for display
        if "recent_messages" not in self.config:
            self.config["recent_messages"] = []
        self.config["recent_messages"].append(message)
        if len(self.config["recent_messages"]) > 50:
            self.config["recent_messages"] = self.config["recent_messages"][-50:]

    def add_log_message(self, message: str):
        """Add a log message to the live log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_messages.append(formatted_message)

    def _draw_volume_profile_pane(
        self, start_y: int, start_x: int, height: int, width: int
    ):
        """Draws the market depth pane."""
        self.safe_addstr(start_y, start_x, "--- Market Depth ---", curses.A_BOLD)

        try:
            self.logger.debug(
                f"Market depth pane: y={start_y}, x={start_x}, h={height}, w={width}"
            )

            if not self.price_fetcher:
                self.logger.warning("No price fetcher available for market depth")
                self.safe_addstr(start_y + 2, start_x, "No price fetcher available")
                return

            self.logger.debug("Fetching order book for market depth...")

            order_book = self.price_fetcher.get_order_book("BTCUSDT", limit=100)

            if not order_book:
                self.logger.warning("No order book data received from API")
                self.safe_addstr(start_y + 2, start_x, "Loading order book...")
                return

            bids_data = order_book.get("bids", [])
            asks_data = order_book.get("asks", [])

            self.logger.debug(
                f"Order book received: {len(bids_data)} bids, {len(asks_data)} asks"
            )

            if not bids_data or not asks_data:
                self.logger.warning("Order book data is empty")
                self.safe_addstr(start_y + 2, start_x, "No order book data")
                return

            y = start_y + 2

            self.logger.debug("Starting to display market depth data...")

            try:
                header_color = curses.color_pair(6) if curses.has_colors() else 0
            except:
                header_color = 0
            self.safe_addstr(y, start_x, "  Price |     Qty | Bar   ", header_color)
            y += 1

            asks_display = asks_data[:5]
            asks_display.reverse()

            try:
                ask_color = curses.color_pair(2) if curses.has_colors() else 0
            except:
                ask_color = 0
            self.safe_addstr(y, start_x, "--- ASKS (Sell) ---", ask_color)
            y += 1

            all_quantities = [float(bid[1]) for bid in bids_data[:5]] + [
                float(ask[1]) for ask in asks_data[:5]
            ]
            max_qty = max(all_quantities) if all_quantities else 1

            for ask in asks_display:
                try:
                    price = float(ask[0])
                    qty = float(ask[1])
                    bar_width = max(0, min(6, int((qty / max_qty) * 6)))
                    bar_str = "█" * bar_width + "░" * (6 - bar_width)

                    self.safe_addstr(
                        y, start_x, f"{price:7.1f} | {qty:7.4f} | {bar_str}", ask_color
                    )
                    y += 1
                except (ValueError, TypeError, IndexError):
                    continue

            if bids_data and asks_data:
                try:
                    best_bid = float(bids_data[0][0])
                    best_ask = float(asks_data[0][0])
                    spread = best_ask - best_bid
                    try:
                        spread_color = (
                            curses.color_pair(3) if curses.has_colors() else 0
                        )
                    except:
                        spread_color = 0
                    self.safe_addstr(
                        y, start_x, f"--- SPREAD: ${spread:.2f} ---", spread_color
                    )
                    y += 1
                except:
                    pass

            try:
                bid_color = curses.color_pair(1) if curses.has_colors() else 0
            except:
                bid_color = 0
            self.safe_addstr(y, start_x, "--- BIDS (Buy) ---", bid_color)
            y += 1

            for bid in bids_data[:5]:
                try:
                    price = float(bid[0])
                    qty = float(bid[1])
                    bar_width = max(0, min(6, int((qty / max_qty) * 6)))
                    bar_str = "█" * bar_width + "░" * (6 - bar_width)

                    self.safe_addstr(
                        y, start_x, f"{price:7.1f} | {qty:7.4f} | {bar_str}", bid_color
                    )
                    y += 1
                except (ValueError, TypeError, IndexError):
                    continue

            self.logger.debug("Market depth display completed successfully")

        except Exception as e:
            self.safe_addstr(start_y + 2, start_x, f"Order book error: {str(e)[:20]}")
            self.logger.error(f"Market depth error: {e}")
            import traceback

            self.logger.error(f"Market depth traceback: {traceback.format_exc()}")

    def run(self, stdscr):
        """
        Main loop to run the dashboard UI. Handles keyboard input and periodic redraws.
        Press 'q' to quit the dashboard.

        Args:
            stdscr: The curses standard screen object passed by curses.wrapper()
        """
        self.stdscr = stdscr
        try:
            curses.start_color()
            curses.use_default_colors()

            # Initialize color pairs
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Green text
            curses.init_pair(2, curses.COLOR_RED, -1)  # Red text
            curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Yellow text
            curses.init_pair(4, curses.COLOR_CYAN, -1)  # Cyan text
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Magenta text
            curses.init_pair(6, curses.COLOR_WHITE, -1)  # White text
            curses.init_pair(7, curses.COLOR_BLUE, -1)  # Blue text

            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            self.running = True
            curses.mousemask(1)

            frame_count = 0
            while self.running:
                # Draw frame every iteration for maximum responsiveness
                self._draw_frame()  # Update every 0.5 seconds for smooth chart updates

                c = self.stdscr.getch()
                if c == ord("q"):
                    self.running = False
                elif c == ord("d"):
                    self.drawing_mode = not self.drawing_mode
                    self.draw_points = []
                    self.logger.info(
                        f"Drawing mode {'enabled' if self.drawing_mode else 'disabled'}."
                    )
                elif c >= ord("1") and c <= ord("6"):
                    self.timeframe = self.timeframes[c - ord("1")]
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

                frame_count += 1
                time.sleep(0.2)  # More frequent updates for live P&L
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
        finally:
            try:
                curses.nocbreak()
                if self.stdscr:
                    self.stdscr.keypad(False)
                curses.echo()

                # Stop API monitoring
                if hasattr(self, "api_tester"):
                    self.api_tester.stop_monitoring()

                # Remove log handler to prevent memory leaks
                if hasattr(self, "log_handler"):
                    root_logger = logging.getLogger()
                    root_logger.removeHandler(self.log_handler)
            except curses.error:
                pass  # Ignore cleanup errors
            # Note: curses.wrapper() handles endwin() automatically

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
            if (
                y >= 0
                and x >= 0
                and y < self.stdscr.getmaxyx()[0]
                and x + len(text) < self.stdscr.getmaxyx()[1]
            ):
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
            if (
                y >= 0
                and x >= 0
                and y < self.stdscr.getmaxyx()[0]
                and x < self.stdscr.getmaxyx()[1]
            ):
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
    from binance.client import Client

    from utils.price_fetcher import PriceFetcher

    logger = setup_logger("ConsoleDashboard")

    # Mock PriceFetcher for standalone mode
    class MockPriceFetcher:
        def get_historical_klines(self, symbol, interval, limit=200):
            # Generate some random data for testing
            import numpy as np

            data = []
            price = 50000
            for i in range(limit):
                price += (np.random.rand() - 0.5) * 100
                data.append(
                    [
                        int(time.time() * 1000) - (limit - i) * 60000,
                        price,
                        price + 50,
                        price - 50,
                        price,
                        np.random.rand() * 10,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            df = pd.DataFrame(
                data,
                columns=["open_time", "open", "high", "low", "close", "volume"]
                + [f"extra_{i}" for i in range(6)],
            )
            return df

        def get_order_book(self, symbol, limit=100):
            import numpy as np

            bids = [[50000 - i * 10, np.random.rand() * 10] for i in range(5)]
            asks = [[50010 + i * 10, np.random.rand() * 10] for i in range(5)]
            return {"bids": bids, "asks": asks}

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
            return {"wins": 10, "losses": 5}

        def get_average_pnl(self):
            return {"avg_win": 100.0, "avg_loss": -50.0}

    class MockSystemMonitor:
        def get_network_latency(self):
            return 50.0

        def get_error_rates(self):
            return {"binance": 1}

    config = {
        "portfolio_value": 10520.30,
        "open_positions": [
            {"symbol": "BTC/USDT", "amount": 0.1, "price": 50000},
            {"symbol": "ETH/USDT", "amount": 2, "price": 2500},
        ],
        "risk_manager": MockRiskManager(),
        "performance_monitor": MockPerformanceMonitor(),
        "trade_analyzer": MockTradeAnalyzer(),
        "system_monitor": MockSystemMonitor(),
    }

    # Use real price fetcher for live market depth data
    try:
        price_fetcher = PriceFetcher(logger)
        logger.info("Using real PriceFetcher for live market data")
    except Exception as e:
        logger.warning(f"Failed to create real PriceFetcher: {e}, using mock data")
        price_fetcher = MockPriceFetcher()

    dashboard = ConsoleDashboard(
        config=config, logger=logger, price_fetcher=price_fetcher
    )
    dashboard.run(stdscr)


class ConsoleDashboardManager:
    """Manager class for the console dashboard that handles threading and lifecycle."""

    def __init__(self, logger, config, price_fetcher=None):
        self.logger = logger
        self.config = config
        # Create real price fetcher if not provided
        if price_fetcher is None:
            try:
                from utils.price_fetcher import PriceFetcher

                price_fetcher = PriceFetcher(logger)
                logger.info("Created real PriceFetcher for ConsoleDashboardManager")
            except Exception as e:
                logger.warning(f"Failed to create PriceFetcher: {e}")
                price_fetcher = None

        self.dashboard = ConsoleDashboard(
            config=config, logger=logger, price_fetcher=price_fetcher
        )
        self.running = False
        self.thread = None

    def start_dashboard(self):
        """Start the dashboard in a separate thread."""
        import sys
        import threading

        # Check if we can actually run a curses dashboard
        if not sys.stdout.isatty():
            self.logger.warning("No TTY available - console dashboard disabled")
            self.logger.info(
                "Running in headless mode - check logs for trading activity"
            )
            self.running = False
            return

        # Check if TERM is set
        import os

        if not os.getenv("TERM"):
            self.logger.warning(
                "TERM environment variable not set - console dashboard disabled"
            )
            self.logger.info(
                "Running in headless mode - check logs for trading activity"
            )
            self.running = False
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self.thread.start()
        self.logger.info("Console dashboard started")

    def stop_dashboard(self):
        """Stop the dashboard."""
        self.running = False
        if self.dashboard:
            self.dashboard.running = False
        self.logger.info("Console dashboard stopped")

    def is_running(self):
        """Check if dashboard is running."""
        return self.running and (self.thread is None or self.thread.is_alive())

    def add_message(self, message):
        """Add a message to the dashboard."""
        if self.dashboard:
            self.dashboard.add_message(message)

    def _run_dashboard(self):
        """Internal method to run the dashboard."""
        try:
            # Check if we have a proper terminal
            import sys

            if not sys.stdout.isatty():
                self.logger.warning("Not running in a TTY, dashboard disabled")
                self.running = False
                return

            curses.wrapper(self.dashboard.run)
        except curses.error as e:
            self.logger.warning(f"Curses terminal UI not available: {e}")
            self.logger.info(
                "Running in headless mode - check logs for trading activity"
            )
            self.running = False
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            self.running = False


if __name__ == "__main__":
    curses.wrapper(main)
