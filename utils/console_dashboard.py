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

            # Define layout sections with type safety
            left_pane_width = 35
            max_x = int(max_x)  # Ensure integer type
            chart_pane_width = max(10, max_x - left_pane_width - 30)  # Ensure positive
            right_pane_width = 28
            
            chart_pane_x = left_pane_width + 1
            right_pane_x = chart_pane_x + chart_pane_width + 1

            # Draw panes
            self._draw_box(8, 1, max_y - 2, left_pane_width) # Left pane
            self._draw_box(8, chart_pane_x, max_y - 2, chart_pane_x + chart_pane_width) # Chart pane
            self._draw_box(8, right_pane_x, max_y - 2, max_x - 2) # Right pane

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
                self._draw_volume_profile_pane(9, right_pane_x + 2, max_y - 20, right_pane_width - 2)
            except Exception as e:
                self.logger.error(f"Error in _draw_volume_profile_pane: {e}")
                
            try:
                self._draw_position_sizing_pane(max_y - 15, right_pane_x + 2, 8, right_pane_width - 2)
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
        
        # Get live data from config (updated dynamically in main loop)
        try:
            # Prioritize values from config which are updated live from trading loop
            portfolio_value = float(self.config.get('portfolio_value', 10000.0))
            unrealized_pnl = float(self.config.get('unrealized_pnl', 0.0))
            realized_pnl = float(self.config.get('realized_pnl', 0.0))
            
            # Fallback to risk manager if config values are default
            if portfolio_value == 10000.0 and risk_manager:
                try:
                    unrealized_pnl = float(risk_manager.unrealized_pnl or 0.0)
                    realized_pnl = float(risk_manager.realized_pnl or 0.0)
                    # Calculate portfolio value as starting balance + PnL
                    starting_balance = 10000.0  # Default paper trading balance
                    portfolio_value = starting_balance + realized_pnl + unrealized_pnl
                except (ValueError, TypeError, AttributeError):
                    pass  # Keep config values
                    
            self.logger.debug(f"Dashboard displaying - Portfolio: ${portfolio_value:.2f}, "
                            f"Unrealized: ${unrealized_pnl:.2f}, Realized: ${realized_pnl:.2f}")
                            
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Error parsing portfolio values: {e}")
            portfolio_value = 10000.0
            unrealized_pnl = 0.0
            realized_pnl = 0.0
        
        self.safe_addstr(y + 1, start_x, f"Value: ${portfolio_value:,.2f}")
        
        # Color code PnL based on positive/negative
        unrealized_color = curses.color_pair(1) if unrealized_pnl >= 0 else curses.color_pair(2)
        realized_color = curses.color_pair(1) if realized_pnl >= 0 else curses.color_pair(2)
        
        self.safe_addstr(y + 2, start_x, f"Unrealized PnL: ${unrealized_pnl:,.2f}", unrealized_color)
        self.safe_addstr(y + 3, start_x, f"Realized PnL: ${realized_pnl:,.2f}", realized_color)

        # Positions
        y += 5
        self.safe_addstr(y, start_x, "--- Open Positions ---", curses.color_pair(3) | curses.A_BOLD)
        
        # Get live positions from config or risk manager
        open_positions = self.config.get('open_positions', [])
        if not open_positions and risk_manager and hasattr(risk_manager, 'open_positions'):
            # Fallback to risk manager format (handle both list and dict)
            risk_positions = risk_manager.open_positions
            if risk_positions:
                if isinstance(risk_positions, dict):
                    # Dictionary format
                    for i, (symbol, pos_data) in enumerate(list(risk_positions.items())[:5]): # Show top 5
                        side = pos_data.get('side', 'N/A')
                        amount = pos_data.get('quantity', 0)
                        price = pos_data.get('entry_price', 0)
                        self.safe_addstr(y + 1 + i, start_x, f"{symbol} {side} {amount:.4f} @ ${price:.2f}")
                elif isinstance(risk_positions, list):
                    # List format
                    for i, pos_data in enumerate(risk_positions[:5]): # Show top 5
                        symbol = pos_data.get('symbol', 'N/A')
                        side = 'LONG' if pos_data.get('quantity', 0) > 0 else 'SHORT' if pos_data.get('quantity', 0) < 0 else 'N/A'
                        amount = abs(pos_data.get('quantity', 0))
                        price = pos_data.get('entry_price', 0)
                        self.safe_addstr(y + 1 + i, start_x, f"{symbol} {side} {amount:.4f} @ ${price:.2f}")
            else:
                self.safe_addstr(y + 1, start_x, "None")
        elif open_positions:
            # Use config format (from trading dashboard)
            for i, pos in enumerate(open_positions[:5]):  # Show top 5
                symbol = pos.get('symbol', 'N/A')
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                pnl = pos.get('pnl', 0)
                side = 'LONG' if size > 0 else 'SHORT' if size < 0 else 'N/A'
                self.safe_addstr(y + 1 + i, start_x, f"{symbol} {side} {abs(size):.6f} @ ${entry_price:.2f} | PnL: ${pnl:.2f}")
        else:
            self.safe_addstr(y + 1, start_x, "None")

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
        recent_trades = self.config.get('recent_trades', [])
        if recent_trades:
            for i, trade in enumerate(recent_trades[:3]):  # Show last 3 trades
                side = trade.get('side', 'N/A')
                symbol = trade.get('symbol', 'N/A')
                price = trade.get('price', 0)
                quantity = trade.get('quantity', 0)
                pnl = trade.get('pnl', 0)
                
                # Color code by side
                side_color = curses.color_pair(1) if side == 'BUY' else curses.color_pair(2)
                pnl_color = curses.color_pair(1) if pnl >= 0 else curses.color_pair(2)
                
                self.safe_addstr(y + 1 + i, start_x, f"{side} {quantity:.4f} {symbol} @ ${price:.2f}", side_color)
                self.safe_addstr(y + 1 + i, start_x + 25, f"PnL: ${pnl:.2f}", pnl_color)
        else:
            self.safe_addstr(y + 1, start_x, "No trades yet")

        # Trade Distribution
        y += 5
        self.safe_addstr(y, start_x, "--- Trade Statistics ---", curses.A_BOLD)
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

        # Position-level Risk
        y += 4
        self.safe_addstr(y, start_x, "--- Position Risk ---", curses.A_BOLD)
        
        # Calculate risk from open positions
        open_positions = self.config.get('open_positions', [])
        if open_positions:
            for i, pos in enumerate(open_positions[:3]):  # Show top 3
                symbol = pos.get('symbol', 'N/A')
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                current_price = pos.get('current_price', entry_price)
                
                # Calculate position value as risk
                position_value = abs(size) * current_price
                self.safe_addstr(y + 1 + i, start_x, f"{symbol}: ${position_value:,.2f}")
        elif risk_manager and hasattr(risk_manager, 'get_position_level_risk'):
            # Fallback to risk manager
            position_risk = risk_manager.get_position_level_risk()
            for i, (symbol, risk) in enumerate(position_risk.items()):
                self.safe_addstr(y + 1 + i, start_x, f"{symbol}: ${risk:,.2f}")
        else:
            self.safe_addstr(y + 1, start_x, "No positions")

    def _draw_position_sizing_pane(self, start_y: int, start_x: int, height: int, width: int):
        """Draws the position sizing visualization pane."""
        self.safe_addstr(start_y, start_x, "--- Position Sizing ---", curses.A_BOLD)
        
        try:
            portfolio_value = float(self.config.get('portfolio_value', 10000.0))
            
            y = start_y + 2
            
            # Get positions from config (updated from paper trading database)
            open_positions = self.config.get('open_positions', [])
            
            if not open_positions:
                self.safe_addstr(y, start_x, "No open positions")
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
            if not self.price_fetcher:
                self.safe_addstr(start_y + 2, start_x, "No price fetcher available")
                return
                
            # Always fetch fresh order book data (no cache for live updates)
            order_book = self.price_fetcher.get_order_book("BTCUSDT", limit=10)
            
            if not order_book:
                self.safe_addstr(start_y + 2, start_x, "Loading order book...")
                return

            # Ensure order book data is properly formatted
            bids_data = order_book.get('bids', [])
            asks_data = order_book.get('asks', [])
            
            if not bids_data or not asks_data:
                self.safe_addstr(start_y + 2, start_x, "No order book data")
                return
            
            # Simplify - don't use pandas for this display, work directly with the data    
            y = start_y + 2
            
            # Display header
            self.safe_addstr(y, start_x, "Price      | Qty      | Bar", curses.color_pair(6))
            y += 1
            
            # Display asks (sell orders) at top - reverse order for proper display
            asks_display = asks_data[:5]  # Top 5 asks
            asks_display.reverse()  # Show highest price first
            
            self.safe_addstr(y, start_x, "--- ASKS (Sell) ---", curses.color_pair(2))
            y += 1
            
            # Calculate max quantity for bar scaling
            all_quantities = [float(bid[1]) for bid in bids_data[:5]] + [float(ask[1]) for ask in asks_data[:5]]
            max_qty = max(all_quantities) if all_quantities else 1
            
            for ask in asks_display:
                try:
                    price = float(ask[0])
                    qty = float(ask[1])
                    bar_width = max(0, min(8, int((qty / max_qty) * 8)))
                    bar_str = '█' * bar_width + '░' * (8 - bar_width)
                    
                    self.safe_addstr(y, start_x, f"{price:8.0f} | {qty:8.3f} | {bar_str}", curses.color_pair(2))
                    y += 1
                except (ValueError, TypeError, IndexError):
                    continue
            
            # Spread display
            if bids_data and asks_data:
                try:
                    best_bid = float(bids_data[0][0])
                    best_ask = float(asks_data[0][0])
                    spread = best_ask - best_bid
                    self.safe_addstr(y, start_x, f"--- SPREAD: ${spread:.2f} ---", curses.color_pair(3))
                    y += 1
                except:
                    pass
            
            # Display bids (buy orders) at bottom
            self.safe_addstr(y, start_x, "--- BIDS (Buy) ---", curses.color_pair(1))
            y += 1
            
            for bid in bids_data[:5]:  # Top 5 bids
                try:
                    price = float(bid[0])
                    qty = float(bid[1])
                    bar_width = max(0, min(8, int((qty / max_qty) * 8)))
                    bar_str = '█' * bar_width + '░' * (8 - bar_width)
                    
                    self.safe_addstr(y, start_x, f"{price:8.0f} | {qty:8.3f} | {bar_str}", curses.color_pair(1))
                    y += 1
                except (ValueError, TypeError, IndexError):
                    continue
                    
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
    
    price_fetcher = MockPriceFetcher()
    dashboard = ConsoleDashboard(config=config, logger=logger, price_fetcher=price_fetcher)
    dashboard.stdscr = stdscr
    dashboard.run()

class ConsoleDashboardManager:
    """Manager class for the console dashboard that handles threading and lifecycle."""
    
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.dashboard = ConsoleDashboard(config=config, logger=logger)
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
