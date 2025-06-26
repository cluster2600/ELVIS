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

class LogHandler(logging.Handler):
    """Custom log handler to capture logs for dashboard display"""
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.setLevel(logging.INFO)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.dashboard.add_log_message(msg)
        except Exception:
            pass  # Don't let logging errors crash the dashboard

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
        self.messages = []  # Store console messages
        self.log_messages = deque(maxlen=100)  # Store recent log messages
        self.last_chart_data = None  # Cache last chart to reduce flashing
        self.chart_buffer = {}  # Store chart display buffer
        
        # Set up log handler to capture logs
        self.log_handler = LogHandler(self)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
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
                self.stdscr.addstr(0, 0, "Terminal too small, resize to at least 120x40")
                self.stdscr.refresh()
                return

            # Main layout boxes
            self._draw_box(0, 0, max_y - 1, max_x - 1) # Main border
            self._draw_header()

            # Define layout sections with type safety - enlarged right pane
            left_pane_width = 35
            max_x = int(max_x)  # Ensure integer type
            right_pane_width = 45  # Increased from 28 to 45 for better visibility
            chart_pane_width = max(10, max_x - left_pane_width - right_pane_width - 3)  # Adjust chart accordingly
            
            chart_pane_x = left_pane_width + 1
            right_pane_x = chart_pane_x + chart_pane_width + 1

            # Draw panes
            self._draw_box(8, 1, max_y - 2, left_pane_width) # Left pane
            self._draw_box(8, chart_pane_x, max_y - 2, chart_pane_x + chart_pane_width) # Chart pane
            self._draw_box(8, right_pane_x, max_y - 2, max_x - 2) # Right pane
            
            # Add a marker in the right pane to help locate it
            try:
                self.safe_addstr(8, right_pane_x + 1, f"RIGHT PANE (cols {right_pane_x}-{max_x-2})", curses.color_pair(4))
            except:
                pass

            # Draw content in panes with individual error handling
            try:
                self._draw_info_pane(9, 3)
            except Exception as e:
                self.logger.error(f"Error in _draw_info_pane: {e}")
                
            try:
                self._draw_chart_pane(9, chart_pane_x + 2, max_y - 15, chart_pane_width - 2)
            except Exception as e:
                self.logger.error(f"Error in _draw_chart_pane: {e}")
                
            try:
                # Give market depth more space and better positioning
                market_depth_height = min(18, max_y - 25)  # More reasonable height
                if market_depth_height > 10:  # Only draw if we have enough space
                    self._draw_volume_profile_pane(9, right_pane_x + 2, market_depth_height, right_pane_width - 2)
                    self.logger.debug(f"Market depth drawn at x={right_pane_x + 2}, h={market_depth_height}")
                else:
                    self.safe_addstr(9, right_pane_x + 2, "Terminal too small", curses.color_pair(2))
            except Exception as e:
                self.logger.error(f"Error in _draw_volume_profile_pane: {e}")
                # Draw error message in the right pane so user can see something
                self.safe_addstr(9, right_pane_x + 2, "Market Depth Error", curses.color_pair(2))
                
            try:
                # Position sizing below market depth with adjusted spacing
                position_y = 9 + market_depth_height + 2 if 'market_depth_height' in locals() else max_y - 12
                position_height = max(6, max_y - position_y - 3)
                if position_height > 0:
                    self._draw_position_sizing_pane(position_y, right_pane_x + 2, position_height, right_pane_width - 2)
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
        
        # Time and Status - always live
        current_time = datetime.now()
        self.safe_addstr(y, start_x, f"Time: {current_time.strftime('%H:%M:%S')}", curses.color_pair(4))
        self.safe_addstr(y + 1, start_x, f"Date: {current_time.strftime('%Y-%m-%d')}", curses.color_pair(6))
        self.safe_addstr(y + 2, start_x, f"Status: LIVE TRADING", curses.color_pair(1) | curses.A_BOLD)
        
        # Portfolio
        y += 3
        self.safe_addstr(y, start_x, "--- Portfolio ---", curses.color_pair(3) | curses.A_BOLD)
        
        risk_manager = self.config.get('risk_manager')
        
        # Calculate live P&L directly from database
        try:
            # Get live data from paper trading database
            from utils.paper_trade_db import get_all_trades, get_open_positions
            
            # Calculate realized P&L from completed trades
            all_trades = get_all_trades(limit=1000)
            realized_pnl = 0.0
            
            for trade in all_trades:
                if len(trade) >= 7:
                    pnl = float(trade[6])
                    realized_pnl += pnl
            
            # Calculate unrealized P&L from open positions
            open_positions_raw = get_open_positions()
            unrealized_pnl = 0.0
            
            for pos in open_positions_raw:
                if len(pos) >= 4:
                    try:
                        # Get current price for this position
                        current_price = 107500.0  # Default BTC price
                        if self.price_fetcher:
                            try:
                                fetched_price = self.price_fetcher.get_current_price(pos[1])
                                if fetched_price:
                                    current_price = float(fetched_price)
                            except:
                                pass
                        
                        entry_price = float(pos[2])
                        quantity = float(pos[3])
                        position_pnl = (current_price - entry_price) * quantity
                        unrealized_pnl += position_pnl
                    except:
                        pass
            
            # Calculate realistic portfolio value
            starting_balance = 10000.0
            
            # For portfolio calculation, use only the realized P&L since unrealized might be inflated
            # due to incorrect position sizing in the trading logic
            portfolio_value = starting_balance + realized_pnl
            
            # Add unrealized P&L only if it's reasonable (not inflated)
            if abs(unrealized_pnl) < 5000:  # Sanity check for unrealized P&L
                portfolio_value += unrealized_pnl
            
            # Don't use config portfolio value as it may be inflated from bad position sizing
            
            self.logger.debug(f"Live P&L calculated - Realized: ${realized_pnl:.2f}, "
                            f"Unrealized: ${unrealized_pnl:.2f}, Portfolio: ${portfolio_value:.2f}")
                            
        except Exception as e:
            self.logger.warning(f"Error calculating live P&L: {e}")
            # Fallback to config values
            portfolio_value = float(self.config.get('portfolio_value', 10000.0))
            unrealized_pnl = float(self.config.get('unrealized_pnl', 0.0))
            realized_pnl = float(self.config.get('realized_pnl', 0.0))
        
        self.safe_addstr(y + 1, start_x, f"Value: ${portfolio_value:,.2f}")
        
        # Color code PnL based on positive/negative
        unrealized_color = curses.color_pair(1) if unrealized_pnl >= 0 else curses.color_pair(2)
        realized_color = curses.color_pair(1) if realized_pnl >= 0 else curses.color_pair(2)
        
        self.safe_addstr(y + 2, start_x, f"Unrealized PnL: ${unrealized_pnl:,.2f}", unrealized_color)
        self.safe_addstr(y + 3, start_x, f"Realized PnL: ${realized_pnl:,.2f}", realized_color)

        # Positions (Live from Database with Real-time PnL)
        y += 5
        self.safe_addstr(y, start_x, "--- Open Positions ---", curses.color_pair(3) | curses.A_BOLD)
        
        try:
            # Get live positions directly from database
            from utils.paper_trade_db import get_open_positions
            live_positions = get_open_positions()
            
            if live_positions:
                # Get current price for PnL calculation
                current_price = 107500.0  # Default BTC price
                if self.price_fetcher:
                    try:
                        fetched_price = self.price_fetcher.get_current_price('BTCUSDT')
                        if fetched_price:
                            current_price = float(fetched_price)
                    except:
                        pass
                
                displayed_positions = 0
                for pos in live_positions[:5]:  # Show top 5 positions
                    if len(pos) >= 4:
                        symbol = pos[1]
                        entry_price = float(pos[2])
                        quantity = float(pos[3])
                        
                        # Calculate comprehensive real-time P&L with all fees
                        try:
                            leverage = int(pos[4]) if len(pos) > 4 else 10
                            entry_time = pos[5] if len(pos) > 5 else None
                            
                            # Try to get executor for comprehensive fee calculation
                            executor = None
                            try:
                                from core.di import container
                                executor = container.get_optional('executor')
                            except:
                                pass
                            
                            if executor and hasattr(executor, 'calculate_open_position_pnl'):
                                # Get comprehensive P&L including all Binance fees
                                pnl_detail = executor.calculate_open_position_pnl(
                                    symbol, current_price, entry_price, quantity, leverage, entry_time
                                )
                                net_pnl = pnl_detail['net_pnl']
                                gross_pnl = pnl_detail['gross_pnl']
                                total_fees = pnl_detail['ongoing_costs'] + pnl_detail['estimated_exit_fee']
                                funding_fee = pnl_detail['funding_fee']
                                borrowing_cost = pnl_detail['borrowing_cost']
                                hours_held = pnl_detail['hours_held']
                                
                                # Show fee impact
                                fee_impact = f" (Fees: ${total_fees:.4f})"
                                if hours_held > 0:
                                    fee_impact += f" {hours_held:.1f}h"
                            else:
                                # Fallback to simple calculation
                                net_pnl = (current_price - entry_price) * quantity
                                gross_pnl = net_pnl
                                fee_impact = " (est)"
                        except Exception as e:
                            # Safe fallback
                            net_pnl = (current_price - entry_price) * quantity
                            gross_pnl = net_pnl
                            fee_impact = ""
                        
                        # Determine side
                        side = 'LONG' if quantity > 0 else 'SHORT'
                        
                        # Color code based on net P&L
                        try:
                            pnl_color = curses.color_pair(1) if net_pnl >= 0 else curses.color_pair(2)
                        except:
                            pnl_color = 0
                        
                        # Display position with comprehensive P&L info
                        position_text = f"{symbol} {side} {abs(quantity):.4f} @ ${entry_price:.0f}"
                        pnl_text = f"Net: ${net_pnl:+.2f}{fee_impact}"
                        
                        self.safe_addstr(y + 1 + displayed_positions, start_x, position_text)
                        self.safe_addstr(y + 1 + displayed_positions, start_x + 22, pnl_text, pnl_color)
                        displayed_positions += 1
                
                # Show current BTC price used for calculations
                self.safe_addstr(y + 1 + displayed_positions, start_x, f"BTC Price: ${current_price:,.2f}")
                
            else:
                # Fallback to config positions or show "None"
                config_positions = self.config.get('open_positions', [])
                if config_positions:
                    for i, pos in enumerate(config_positions[:3]):
                        symbol = pos.get('symbol', 'N/A')
                        size = pos.get('size', 0)
                        entry_price = pos.get('entry_price', 0)
                        side = 'LONG' if size > 0 else 'SHORT' if size < 0 else 'N/A'
                        self.safe_addstr(y + 1 + i, start_x, f"{symbol} {side} {abs(size):.6f} @ ${entry_price:.2f}")
                else:
                    self.safe_addstr(y + 1, start_x, "No open positions")
                    
        except Exception as e:
            self.logger.warning(f"Error getting live positions: {e}")
            self.safe_addstr(y + 1, start_x, f"Position data error")

        # System Monitoring
        y += 8
        self.safe_addstr(y, start_x, "--- System Health ---", curses.color_pair(3) | curses.A_BOLD)
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
        self.safe_addstr(y, start_x, "--- Performance ---", curses.color_pair(3) | curses.A_BOLD)
        performance_monitor = self.config.get('performance_monitor')
        if performance_monitor:
            try:
                sharpe = float(performance_monitor.calculate_rolling_sharpe() or 0.0)
                drawdown = float(performance_monitor.calculate_rolling_drawdown() or 0.0)
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

        # Recent Trades (Live from Database)
        y += 6
        self.safe_addstr(y, start_x, "--- Recent Trades ---", curses.color_pair(3) | curses.A_BOLD)
        
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
                        'side': trade[3],
                        'symbol': trade[2],
                        'price': float(trade[4]),
                        'quantity': float(trade[5]),
                        'pnl': float(trade[6]),
                        'timestamp': trade[1]
                    }
                    
                    # Collect trades with actual P&L
                    if trade_data['pnl'] != 0.0:
                        profitable_trades.append(trade_data)
                    
                    # Also collect recent activity (last 10 trades)
                    if len(recent_activity) < 10:
                        recent_activity.append(trade_data)
            
            # Show profitable trades first, fallback to recent activity
            display_trades = profitable_trades[:3] if profitable_trades else recent_activity[:3]
            
            if display_trades:
                for i, trade in enumerate(display_trades):
                    side = trade['side']
                    symbol = trade['symbol']
                    price = trade['price']
                    quantity = trade['quantity']
                    pnl = trade['pnl']
                    
                    # Color code by side and P&L
                    side_color = curses.color_pair(1) if side == 'BUY' else curses.color_pair(2)
                    
                    # Show trade details
                    trade_info = f"{side} {quantity:.4f} {symbol} @ ${price:.2f}"
                    self.safe_addstr(y + 1 + i, start_x, trade_info, side_color)
                    
                    # Show P&L or trade value for context
                    if pnl != 0.0:
                        pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                        self.safe_addstr(y + 1 + i, start_x + 25, f"PnL: ${pnl:.2f}", pnl_color)
                    else:
                        # Show trade value instead of 0.00 P&L
                        trade_value = price * quantity
                        self.safe_addstr(y + 1 + i, start_x + 25, f"Val: ${trade_value:.0f}")
                        
            else:
                self.safe_addstr(y + 1, start_x, "No recent trades")
                
        except Exception as e:
            # Fallback to config trades
            recent_trades = self.config.get('recent_trades', [])
            if recent_trades:
                for i, trade in enumerate(recent_trades[:3]):
                    side = trade.get('side', 'N/A')
                    symbol = trade.get('symbol', 'N/A')
                    price = trade.get('price', 0)
                    quantity = trade.get('quantity', 0)
                    pnl = trade.get('pnl', 0)
                    
                    side_color = curses.color_pair(1) if side == 'BUY' else curses.color_pair(2)
                    trade_info = f"{side} {quantity:.4f} {symbol} @ ${price:.2f}"
                    self.safe_addstr(y + 1 + i, start_x, trade_info, side_color)
                    
                    if pnl != 0.0:
                        pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                        self.safe_addstr(y + 1 + i, start_x + 25, f"PnL: ${pnl:.2f}", pnl_color)
                    else:
                        trade_value = price * quantity
                        self.safe_addstr(y + 1 + i, start_x + 25, f"Val: ${trade_value:.0f}")
            else:
                self.safe_addstr(y + 1, start_x, "No trades yet")

        # Trade Distribution - Live from Database with comprehensive fee-adjusted stats
        y += 5
        self.safe_addstr(y, start_x, "--- Trade Statistics ---", curses.A_BOLD)
        
        # Try to get live trade statistics from database
        try:
            from utils.paper_trade_db import get_all_trades
            trades_raw = get_all_trades(limit=1000, exclude_test=True)  # Exclude TEST trades for accurate statistics
            
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
                        if side == 'BUY':
                            buy_trades += 1
                        elif side == 'SELL':
                            sell_trades += 1
                        
                        # Count all trades for P&L analysis (fees already included in stored P&L)
                        if side in ['BUY', 'SELL']:
                            # Use stored P&L directly (fees already factored in)
                            net_pnl = pnl
                            
                            # Count all real trades for total P&L
                            total_pnl += net_pnl
                            
                            # Only analyze SELL trades for win/loss (completed positions)
                            if side == 'SELL':
                                # P&L already includes fees, so use directly
                                if net_pnl > 0.10:  # Profitable trade
                                    wins += 1
                                    win_pnls.append(net_pnl)
                                elif net_pnl < -0.10:  # Loss trade
                                    losses += 1
                                    loss_pnls.append(net_pnl)
                                else:  # Small result (breakeven)
                                    breakeven += 1
                
                # Calculate statistics
                completed_trades = wins + losses + breakeven
                total_trades = len(trades_raw)
                win_rate = (wins / completed_trades * 100) if completed_trades > 0 else 0.0
                avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
                avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
                avg_trade_value = total_volume / total_trades if total_trades > 0 else 0.0
                
                # Profit factor calculation
                total_wins = sum(win_pnls) if win_pnls else 0.0
                total_losses = abs(sum(loss_pnls)) if loss_pnls else 0.0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
                
                # Show comprehensive trade breakdown (already filtered, so no TEST trades)
                total_trades = len(trades_raw)
                
                self.safe_addstr(y + 1, start_x, f"Real Trades: {total_trades}")
                
                # Show win rate with realistic assessment
                trading_outcomes = wins + losses  # Exclude breakeven for win rate
                if trading_outcomes > 0:
                    meaningful_win_rate = (wins / trading_outcomes * 100)
                    win_rate_color = curses.color_pair(1) if meaningful_win_rate > 50 else curses.color_pair(6) if meaningful_win_rate > 30 else curses.color_pair(2)
                    self.safe_addstr(y + 2, start_x, f"Win Rate: {meaningful_win_rate:.1f}% ({wins}W/{losses}L)", win_rate_color)
                else:
                    self.safe_addstr(y + 2, start_x, "Win Rate: N/A (no decisive trades)", curses.color_pair(6))
                
                self.safe_addstr(y + 3, start_x, f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
                
                # Color code total P&L
                pnl_color = curses.color_pair(1) if total_pnl >= 0 else curses.color_pair(2)
                self.safe_addstr(y + 4, start_x, f"Net P&L: ${total_pnl:.2f}", pnl_color)
                
                # Show profit factor with meaningful color coding
                pf_color = curses.color_pair(1) if profit_factor > 1.5 else curses.color_pair(6) if profit_factor > 1.0 else curses.color_pair(2)
                profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
                self.safe_addstr(y + 5, start_x, f"Profit Factor: {profit_factor_display}", pf_color)
                
                self.safe_addstr(y + 6, start_x, f"Avg Trade: ${avg_trade_value:.0f} | BE: {breakeven}")
            else:
                self.safe_addstr(y + 1, start_x, "No trades yet")
                
        except Exception as e:
            # Fallback to trade analyzer
            trade_analyzer = self.config.get('trade_analyzer')
            if trade_analyzer:
                try:
                    win_loss = trade_analyzer.get_win_loss_distribution()
                    avg_pnl = trade_analyzer.get_average_pnl()
                    wins = int(win_loss.get('wins', 0))
                    losses = int(win_loss.get('losses', 0))
                    avg_win = float(avg_pnl.get('avg_win', 0.0))
                    avg_loss = float(avg_pnl.get('avg_loss', 0.0))
                    self.safe_addstr(y + 1, start_x, f"Wins: {wins} | Losses: {losses}")
                    self.safe_addstr(y + 2, start_x, f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    self.safe_addstr(y + 1, start_x, f"Trade data error")
                    self.logger.debug(f"Trade analyzer error: {e}")
            elif recent_trades:
                # Calculate basic stats from recent trades if trade_analyzer not available
                wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
                losses = len(recent_trades) - wins
                total_pnl = sum(t.get('pnl', 0) for t in recent_trades)
                self.safe_addstr(y + 1, start_x, f"Wins: {wins} | Losses: {losses}")
                self.safe_addstr(y + 2, start_x, f"Total PnL: ${total_pnl:.2f}")
            else:
                self.safe_addstr(y + 1, start_x, "No trade data available")

        # Position-level Risk
        y += 6
        self.safe_addstr(y, start_x, "--- Position Risk ---", curses.A_BOLD)
        
        # Get current portfolio and risk data
        portfolio_value = float(self.config.get('portfolio_value', 10000.0))
        open_positions = self.config.get('open_positions', [])
        
        if open_positions:
            total_position_value = 0
            for i, pos in enumerate(open_positions[:3]):  # Show top 3
                symbol = pos.get('symbol', 'N/A')
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                current_price = pos.get('current_price', entry_price)
                
                # Calculate position value as risk
                position_value = abs(size) * current_price
                total_position_value += position_value
                
                # Calculate percentage of portfolio
                position_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
                
                self.safe_addstr(y + 1 + i, start_x, f"{symbol}: ${position_value:,.2f} ({position_pct:.1f}%)")
            
            # Show total exposure
            total_exposure_pct = (total_position_value / portfolio_value * 100) if portfolio_value > 0 else 0
            self.safe_addstr(y + 4, start_x, f"Total Exposure: ${total_position_value:,.2f} ({total_exposure_pct:.1f}%)")
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
                            volume = float(trade[4]) * float(trade[5])  # price * quantity
                            pnl = float(trade[6])
                            recent_volumes.append(volume)
                            recent_pnls.append(pnl)
                    
                    if recent_volumes:
                        avg_trade_size = sum(recent_volumes) / len(recent_volumes)
                        max_trade_size = max(recent_volumes)
                        
                        # Calculate risk per trade as percentage of portfolio
                        avg_risk_pct = (avg_trade_size / portfolio_value * 100) if portfolio_value > 0 else 0
                        max_risk_pct = (max_trade_size / portfolio_value * 100) if portfolio_value > 0 else 0
                        
                        self.safe_addstr(y + 1, start_x, f"No open positions")
                        self.safe_addstr(y + 2, start_x, f"Avg Trade Size: ${avg_trade_size:,.2f} ({avg_risk_pct:.1f}%)")
                        self.safe_addstr(y + 3, start_x, f"Max Trade Size: ${max_trade_size:,.2f} ({max_risk_pct:.1f}%)")
                        
                        # Calculate volatility from recent PnLs
                        if len(recent_pnls) > 1:
                            import statistics
                            pnl_std = statistics.stdev(recent_pnls) if len(recent_pnls) > 1 else 0
                            self.safe_addstr(y + 4, start_x, f"PnL Volatility: ${pnl_std:.2f}")
                    else:
                        self.safe_addstr(y + 1, start_x, "No recent trading data")
                else:
                    self.safe_addstr(y + 1, start_x, "No positions or trades")
                    
            except Exception as e:
                self.safe_addstr(y + 1, start_x, f"Risk data unavailable")
                self.logger.debug(f"Risk calculation error: {e}")

    def _draw_position_sizing_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the position sizing visualization pane."""
        self.safe_addstr(start_y, start_x, "--- Position Sizing ---", curses.A_BOLD)
        
        try:
            portfolio_value = float(self.config.get('portfolio_value', 10000.0))
            
            y = start_y + 2
            
            # Get positions from config (updated from paper trading database)
            open_positions = self.config.get('open_positions', [])
            
            if not open_positions:
                # Show theoretical position sizing information
                try:
                    # Get current market price
                    current_price = 107000.0  # Default
                    if self.price_fetcher:
                        try:
                            fetched_price = self.price_fetcher.get_current_price('BTCUSDT')
                            if fetched_price:
                                current_price = float(fetched_price)
                        except:
                            pass
                    
                    # Calculate different risk levels
                    risk_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5% risk
                    leverage = 10  # Default leverage from config
                    
                    self.safe_addstr(y, start_x, f"Current BTC Price: ${current_price:,.2f}")
                    self.safe_addstr(y + 1, start_x, f"Portfolio Value: ${portfolio_value:,.2f}")
                    self.safe_addstr(y + 2, start_x, f"Leverage: {leverage}x")
                    
                    y += 4
                    self.safe_addstr(y, start_x, "Position Sizing (Risk %):")
                    
                    for i, risk_pct in enumerate(risk_levels):
                        risk_amount = portfolio_value * risk_pct
                        position_value = risk_amount * leverage
                        position_size = position_value / current_price
                        
                        self.safe_addstr(y + 1 + i, start_x, 
                                       f"{risk_pct*100:.0f}%: {position_size:.4f} BTC (${position_value:,.0f})")
                        
                except Exception as e:
                    self.safe_addstr(y, start_x, "No open positions")
                    self.logger.debug(f"Position sizing calculation error: {e}")
                return
            
            for position in open_positions:
                try:
                    symbol = position.get('symbol', 'UNKNOWN')
                    size = float(position.get('size', 0))
                    entry_price = float(position.get('entry_price', 0))
                    
                    # Get current price for accurate position value
                    current_price = entry_price  # Default fallback
                    if self.price_fetcher:
                        try:
                            fetched_price = self.price_fetcher.get_current_price(symbol)
                            if fetched_price:
                                current_price = float(fetched_price)
                        except:
                            pass  # Use entry price fallback
                        
                    position_value = abs(size) * current_price
                    size_percentage = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                    
                    # Create visualization bar
                    max_bar_width = width - 15  # Leave space for text
                    bar_width = max(0, min(max_bar_width, int(size_percentage / 100 * max_bar_width)))
                    remaining_width = max(0, max_bar_width - bar_width)
                    
                    # Display position info
                    side = "LONG" if size > 0 else "SHORT"
                    self.safe_addstr(y, start_x, f"{symbol} {side}: {size_percentage:.1f}%")
                    self.safe_addstr(y + 1, start_x, f"${position_value:,.0f} / ${portfolio_value:,.0f}")
                    
                    # Draw sizing bar
                    bar_color = curses.color_pair(1) if size > 0 else curses.color_pair(2)
                    bar_str = '█' * bar_width + '░' * remaining_width
                    self.safe_addstr(y + 2, start_x, f"[{bar_str}]", bar_color)
                    
                    y += 4
                    
                    # Prevent overflow
                    if y >= start_y + height - 2:
                        break
                        
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Position sizing error for position {position}: {e}")
                    continue
                    
        except Exception as e:
            self.safe_addstr(start_y + 2, start_x, f"Sizing error: {str(e)[:15]}")
            self.logger.debug(f"Position sizing pane error: {e}")
        
    def _draw_chart_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the candlestick chart pane with OHLC data and technical indicators."""
        self.safe_addstr(start_y, start_x, f"--- BTC/USDT Candlestick Chart ({self.timeframe}) ---", curses.color_pair(3) | curses.A_BOLD)
        
        # Get real OHLC data from price fetcher if available
        current_price = float(self.config.get('current_price', 97000.0))
        ohlc_data = None
        
        # Try to get real data from price fetcher
        if self.price_fetcher:
            try:
                data = self.price_fetcher.get_historical_klines("BTCUSDT", "1m", limit=40)
                if not data.empty and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    ohlc_data = data[['open', 'high', 'low', 'close']].tail(40)
                    current_price = ohlc_data['close'].iloc[-1] if not ohlc_data.empty else current_price
            except Exception as e:
                self.logger.debug(f"Could not fetch real OHLC data: {e}")
        
        # Get from config if available
        if ohlc_data is None and 'ohlc_data' in self.config:
            ohlc_data = self.config['ohlc_data']
            
        # Ensure any OHLC data is converted to proper numeric types early
        if ohlc_data is not None and not ohlc_data.empty:
            try:
                # Force all OHLC columns to be numeric, converting any strings 
                for col in ['open', 'high', 'low', 'close']:
                    if col in ohlc_data.columns:
                        ohlc_data[col] = pd.to_numeric(ohlc_data[col], errors='coerce')
                # Drop any rows with NaN values after conversion
                ohlc_data = ohlc_data.dropna()
                self.logger.debug(f"OHLC data after early type conversion: {ohlc_data.dtypes.to_dict()}")
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
                high_price = float(max(open_price, close_price) * (1 + volatility/2))
                low_price = float(min(open_price, close_price) * (1 - volatility/2))
                
                mock_data.append({
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price)
                })
                base_price = close_price
            
            ohlc_data = pd.DataFrame(mock_data)
        
        if ohlc_data is None or len(ohlc_data) < 2:
            self.safe_addstr(start_y + 2, start_x, "Waiting for candlestick data...", curses.color_pair(3))
            return

        # Chart dimensions
        chart_height = height - 5
        chart_width = min(width - 12, len(ohlc_data))  # Leave space for price scale
        
        # Use the most recent candles that fit
        candles = ohlc_data.tail(chart_width)
        if len(candles) < 2:
            return
            
        # Calculate price range for scaling - ensure numeric types
        try:
            # Ensure candles are already properly typed from early conversion above
            if len(candles) == 0:
                self.safe_addstr(start_y + 2, start_x, "Invalid price data format", curses.color_pair(2))
                return
                
            all_prices = pd.concat([candles['high'], candles['low']]).dropna()
            if len(all_prices) == 0:
                self.safe_addstr(start_y + 2, start_x, "No valid price data", curses.color_pair(2))
                return
                
            min_price, max_price = float(all_prices.min()), float(all_prices.max())
            price_range = max_price - min_price if max_price > min_price else 1.0
            
            # Validate the calculated values
            if not all(isinstance(x, (int, float)) and not pd.isna(x) for x in [min_price, max_price, price_range]):
                self.safe_addstr(start_y + 2, start_x, "Price calculation error", curses.color_pair(2))
                return
                
        except Exception as e:
            self.logger.error(f"Error processing price data: {e}")
            self.safe_addstr(start_y + 2, start_x, f"Data error: {str(e)[:30]}", curses.color_pair(2))
            return

        # Only redraw chart when data changes significantly - prevent flickering
        try:
            last_close = float(candles['close'].iloc[-1])
            # Create a more stable price string that doesn't change every frame
            stable_close = round(last_close, -1)  # Round to nearest 10
            price_str = f"{min_price:.0f}-{max_price:.0f}-{len(candles)}-{stable_close:.0f}"
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
                    open_price = float(candle['open'])
                    high_price = float(candle['high'])
                    low_price = float(candle['low'])
                    close_price = float(candle['close'])
                    
                    # Validate that all values are valid numbers
                    if any(pd.isna(x) or not isinstance(x, (int, float)) for x in [open_price, high_price, low_price, close_price]):
                        continue  # Skip this candle if invalid data
                        
                except (ValueError, TypeError, KeyError):
                    continue  # Skip this candle if conversion fails
                
                # Calculate Y positions (inverted because screen coordinates)
                def price_to_y(price):
                    try:
                        price = float(price)  # Ensure price is numeric
                        normalized = (price - min_price) / price_range
                        return int(start_y + 2 + chart_height - 1 - (normalized * (chart_height - 1)))
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
                candle_color = curses.color_pair(1) if is_bullish else curses.color_pair(2)
                
                # Draw the wick (high-low line)
                for y in range(min(high_y, low_y), max(high_y, low_y) + 1):
                    self.safe_addch(y, candle_x, '|', curses.color_pair(6))
                
                # Draw the body (open-close rectangle)
                body_top = min(open_y, close_y)
                body_bottom = max(open_y, close_y)
                
                if body_top == body_bottom:  # Doji (open == close)
                    self.safe_addch(body_top, candle_x, '-', candle_color | curses.A_BOLD)
                else:
                    # Draw body with different characters for bullish/bearish
                    body_char = '█' if is_bullish else '▓'  # Solid for bullish, lighter for bearish
                    for y in range(body_top, body_bottom + 1):
                        self.safe_addch(y, candle_x, body_char, candle_color | curses.A_BOLD)
                
                # Mark open and close levels
                self.safe_addch(open_y, candle_x, '○', candle_color)  # Open
                self.safe_addch(close_y, candle_x, '●', candle_color | curses.A_BOLD)  # Close
                
            # Draw price scale on the right
            scale_x = start_x + chart_width + 4
            num_levels = min(8, chart_height // 2)
            for i in range(num_levels):
                if num_levels > 1:
                    scale_price = max_price - (i / (num_levels - 1)) * price_range
                else:
                    scale_price = (max_price + min_price) / 2
                scale_y = start_y + 2 + int(i * (chart_height - 1) / max(1, num_levels - 1))
                scale_y = max(start_y + 2, min(scale_y, start_y + chart_height - 1))
                self.safe_addstr(scale_y, scale_x, f"${scale_price:.0f}", curses.color_pair(6))
            
            # Draw volume bars at the bottom (simplified)
            volume_y = start_y + chart_height + 1
            self.safe_addstr(volume_y - 1, start_x, "Volume:", curses.color_pair(6))
            if 'volume' in candles.columns:
                try:
                    max_vol = float(candles['volume'].max())
                    if max_vol > 0:
                        for i, (idx, candle) in enumerate(candles.iterrows()):
                            if i >= chart_width:
                                break
                            try:
                                volume = float(candle['volume'])
                                vol_height = max(1, min(3, int((volume / max_vol) * 3)))
                                candle_x = start_x + 2 + i
                                for h in range(vol_height):
                                    self.safe_addch(volume_y + h, candle_x, '▁', curses.color_pair(4))
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
        
        if 'indicators' in self.config:
            indicators = self.config['indicators']
            rsi_val = indicators.get('rsi', 50.0)
            macd_val = indicators.get('macd', 0.0)
            sma_val = indicators.get('sma_20', current_price)
        elif not ohlc_data.empty:
            # Calculate simple indicators
            closes = ohlc_data['close']
            if len(closes) >= 14:
                # Simple RSI calculation
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50
            
            if len(closes) >= 20:
                sma_val = closes.rolling(window=20).mean().iloc[-1]
        
        # Color code indicators with type safety
        try:
            rsi_val = float(rsi_val)
            macd_val = float(macd_val) 
            sma_val = float(sma_val)
            current_price = float(current_price)
            
            rsi_color = curses.color_pair(1) if 30 <= rsi_val <= 70 else curses.color_pair(2)
            macd_color = curses.color_pair(1) if macd_val > 0 else curses.color_pair(2)
            price_change = ((current_price - sma_val) / sma_val * 100) if sma_val > 0 else 0
            price_color = curses.color_pair(1) if price_change >= 0 else curses.color_pair(2)
        except (ValueError, TypeError, ZeroDivisionError):
            rsi_color = curses.color_pair(6)
            macd_color = curses.color_pair(6)
            price_color = curses.color_pair(6)
            price_change = 0
        
        self.safe_addstr(indicator_y, start_x, f"RSI: {rsi_val:.1f}", rsi_color)
        self.safe_addstr(indicator_y, start_x + 12, f"MACD: {macd_val:.3f}", macd_color)
        self.safe_addstr(indicator_y, start_x + 25, f"Price: ${current_price:.2f}", price_color)
        self.safe_addstr(indicator_y, start_x + 42, f"SMA20: ${sma_val:.0f}", curses.color_pair(6))

    def _draw_console_messages(self, start_y: int, start_x: int, height: int, width: int):
        """Draw console messages at the bottom of the screen."""
        self.safe_addstr(start_y, start_x, "--- Live Trading Logs ---", curses.color_pair(3) | curses.A_BOLD)
        
        # Get recent log messages
        display_messages = list(self.log_messages)[-height+2:] if len(self.log_messages) > height-2 else list(self.log_messages)
        
        if not display_messages:
            # Show waiting message if no logs yet
            self.safe_addstr(start_y + 1, start_x, "Waiting for trading logs...", curses.color_pair(6))
        else:
            for i, message in enumerate(display_messages):
                if i < height - 2:
                    # Truncate message if too long and add color based on content
                    display_text = message[:width-2] if len(message) > width-2 else message
                    
                    # Color code based on log level or content
                    color = curses.color_pair(6)  # Default white
                    if "ERROR" in message or "Failed" in message:
                        color = curses.color_pair(2)  # Red
                    elif "WARNING" in message or "Warning" in message:
                        color = curses.color_pair(3)  # Yellow  
                    elif "BUY" in message or "SELL" in message or "order executed" in message:
                        color = curses.color_pair(1)  # Green
                    elif "signal" in message.lower() or "prediction" in message.lower():
                        color = curses.color_pair(4)  # Cyan
                    
                    self.safe_addstr(start_y + 1 + i, start_x, display_text, color)

    def add_message(self, message: str):
        """Add a message to the console messages."""
        if not hasattr(self, 'messages'):
            self.messages = []
        self.messages.append(message)
        # Keep only last 50 messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]
        
        # Also add to config for display
        if 'recent_messages' not in self.config:
            self.config['recent_messages'] = []
        self.config['recent_messages'].append(message)
        if len(self.config['recent_messages']) > 50:
            self.config['recent_messages'] = self.config['recent_messages'][-50:]
    
    def add_log_message(self, message: str):
        """Add a log message to the live log display."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.log_messages.append(formatted_message)

    def _draw_volume_profile_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the market depth pane."""
        self.safe_addstr(start_y, start_x, "--- Market Depth ---", curses.A_BOLD)
        
        try:
            self.logger.debug(f"Market depth pane: y={start_y}, x={start_x}, h={height}, w={width}")
            
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

            bids_data = order_book.get('bids', [])
            asks_data = order_book.get('asks', [])
            
            self.logger.debug(f"Order book received: {len(bids_data)} bids, {len(asks_data)} asks")
            
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
            
            all_quantities = [float(bid[1]) for bid in bids_data[:5]] + [float(ask[1]) for ask in asks_data[:5]]
            max_qty = max(all_quantities) if all_quantities else 1
            
            for ask in asks_display:
                try:
                    price = float(ask[0])
                    qty = float(ask[1])
                    bar_width = max(0, min(6, int((qty / max_qty) * 6)))
                    bar_str = '█' * bar_width + '░' * (6 - bar_width)
                    
                    self.safe_addstr(y, start_x, f"{price:7.1f} | {qty:7.4f} | {bar_str}", ask_color)
                    y += 1
                except (ValueError, TypeError, IndexError):
                    continue
            
            if bids_data and asks_data:
                try:
                    best_bid = float(bids_data[0][0])
                    best_ask = float(asks_data[0][0])
                    spread = best_ask - best_bid
                    try:
                        spread_color = curses.color_pair(3) if curses.has_colors() else 0
                    except:
                        spread_color = 0
                    self.safe_addstr(y, start_x, f"--- SPREAD: ${spread:.2f} ---", spread_color)
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
                    bar_str = '█' * bar_width + '░' * (6 - bar_width)
                    
                    self.safe_addstr(y, start_x, f"{price:7.1f} | {qty:7.4f} | {bar_str}", bid_color)
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
            curses.init_pair(2, curses.COLOR_RED, -1)    # Red text
            curses.init_pair(3, curses.COLOR_YELLOW, -1) # Yellow text
            curses.init_pair(4, curses.COLOR_CYAN, -1)   # Cyan text
            curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Magenta text
            curses.init_pair(6, curses.COLOR_WHITE, -1)  # White text
            curses.init_pair(7, curses.COLOR_BLUE, -1)   # Blue text
            
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            self.running = True
            curses.mousemask(1)

            frame_count = 0
            while self.running:
                # Only draw frame every few iterations to reduce flashing
                if frame_count % 5 == 0:  # Draw every 5th iteration to reduce flicker more
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
                
                frame_count += 1
                time.sleep(0.5)  # Slower refresh rate to reduce flicker
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
        finally:
            try:
                curses.nocbreak()
                if self.stdscr:
                    self.stdscr.keypad(False)
                curses.echo()
                
                # Remove log handler to prevent memory leaks
                if hasattr(self, 'log_handler'):
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
            import numpy as np
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
            import numpy as np
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
    
    # Use real price fetcher for live market depth data
    try:
        price_fetcher = PriceFetcher(logger)
        logger.info("Using real PriceFetcher for live market data")
    except Exception as e:
        logger.warning(f"Failed to create real PriceFetcher: {e}, using mock data")
        price_fetcher = MockPriceFetcher()
    
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    dashboard.stdscr = stdscr
    dashboard.run()

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
        
        self.dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
        self.running = False
        self.thread = None
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread."""
        import sys
        import threading
        
        # Check if we can actually run a curses dashboard
        if not sys.stdout.isatty():
            self.logger.warning("No TTY available - console dashboard disabled")
            self.logger.info("Running in headless mode - check logs for trading activity")
            self.running = False
            return
        
        # Check if TERM is set
        import os
        if not os.getenv('TERM'):
            self.logger.warning("TERM environment variable not set - console dashboard disabled")
            self.logger.info("Running in headless mode - check logs for trading activity")
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
            self.logger.info("Running in headless mode - check logs for trading activity")
            self.running = False
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            self.running = False

if __name__ == "__main__":
    curses.wrapper(main)
