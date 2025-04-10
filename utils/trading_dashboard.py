"""
Enhanced Trading Dashboard for ELVIS with comprehensive market data and trading features.
"""

import sys
import logging
import time
from pathlib import Path
from collections import deque
import psutil
import pandas as pd
import numpy as np
import talib
import curses
from logging.handlers import RotatingFileHandler
from binance.client import Client
from datetime import datetime, timedelta

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
else:
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

from trading.execution.binance_executor import BinanceExecutor
from trading.performance_monitor import PerformanceMonitor
from config import API_CONFIG, TRADING_CONFIG

# Configure logging to file
log_file = "/Users/maxime/BTC_BOT/BTC_BOT/logs/trading_dashboard.log"
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

class ConsoleDashboard:
    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.animation_frame = 0
        self.stdscr = None
        self.running = False
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'daily_return': 0.0,
            'monthly_return': 0.0,
            'yearly_return': 0.0
        }
        self.strategy_signals = {
            'Technical Strategy': 'HOLD',
            'Mean Reversion': 'HOLD',
            'Trend Following': 'HOLD'
        }
        
    def update_performance_metrics(self, metrics):
        """Update performance metrics for display."""
        self.performance_metrics.update(metrics)
        
    def update_strategy_signals(self, signals):
        """Update strategy signals for display."""
        self.strategy_signals.update(signals)
        
    def _draw_frame(self) -> None:
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()
            if max_y < 40 or max_x < 100:
                self.stdscr.addstr(0, 0, "Terminal too small, resize to at least 100x40")
                return
            
            # Header (10 lines for ASCII art)
            header_height = 10
            self._draw_box(0, 0, header_height, max_x-1)
            self._draw_header()
            
            # Main content
            content_start = header_height + 1
            content_height = max_y - content_start - 5  # Reserve 5 lines for footer
            half_width = max_x // 2
            
            # Left column: Portfolio Info and Performance Metrics
            self._draw_box(content_start, 0, content_start + 5, half_width - 1)
            self._draw_portfolio_info(content_start + 1, 0, half_width - 1)
            self._draw_box(content_start + 6, 0, content_start + content_height // 2, half_width - 1)
            self._draw_performance_metrics(content_start + 7, 0, half_width - 1)
            
            # Right column: Open Positions, Recent Trades, and Strategy Signals
            self._draw_box(content_start, half_width, content_start + content_height // 3, max_x - 1)
            self._draw_open_positions(content_start + 1, half_width, max_x - 1)
            self._draw_box(content_start + content_height // 3 + 1, half_width, content_start + 2 * content_height // 3, max_x - 1)
            self._draw_recent_trades(content_start + content_height // 3 + 2, half_width, max_x - 1)
            self._draw_box(content_start + 2 * content_height // 3 + 1, half_width, content_start + content_height, max_x - 1)
            self._draw_strategy_signals(content_start + 2 * content_height // 3 + 2, half_width, max_x - 1)
            
            # Footer: System Info
            footer_start = max_y - 5
            self._draw_box(footer_start, 0, max_y - 1, max_x - 1)
            self._draw_system_info(footer_start + 1, 0, max_x - 1)
            
            self.animation_frame = (self.animation_frame + 1) % 10
            self.stdscr.refresh()
        except Exception as e:
            self.logger.error(f"Error drawing frame: {e}")
            
    def _draw_box(self, start_y: int, start_x: int, end_y: int, end_x: int) -> None:
        try:
            corner_char = '╔' if self.animation_frame < 5 else '╗'
            self.safe_addch(start_y, start_x, corner_char)
            self.safe_addch(start_y, end_x, corner_char)
            self.safe_addch(end_y, start_x, '╚')
            self.safe_addch(end_y, end_x, '╝')
            for x in range(start_x + 1, end_x):
                self.safe_addch(start_y, x, '═')
                self.safe_addch(end_y, x, '═')
            for y in range(start_y + 1, end_y):
                self.safe_addch(y, start_x, '║')
                self.safe_addch(y, end_x, '║')
        except curses.error:
            pass
            
    def _draw_header(self) -> None:
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            logo = [
                "███████╗██╗     ██╗   ██╗██╗███████╗",
                "██╔════╝██║     ██║   ██║██║██╔════╝",
                "█████╗  ██║     ██║   ██║██║███████╗",
                "██╔══╝  ██║      ██║ ██║ ██║╚════██║",
                "███████╗███████╗╚ ████╔╝ ██║███████║",
                "╚══════╝╚══════╝ ╚════╝  ╚═╝╚══════╝"
            ]
            start_y = 1
            for i, line in enumerate(logo):
                if start_y + i < max_y:
                    x = (max_x - len(line)) // 2
                    color = curses.color_pair(6) if self.animation_frame < 5 else curses.color_pair(5)
                    self.safe_addstr(start_y + i, x, line, color | curses.A_BOLD)
            
            mode = "PRODUCTION" if self.config.get('PRODUCTION_MODE', False) else "TESTNET"
            mode_color = curses.color_pair(1) if mode == "PRODUCTION" else curses.color_pair(3)
            mode_text = f"{mode} MODE"
            self.safe_addstr(7, max_x - len(mode_text) - 2, mode_text, mode_color | curses.A_BOLD | curses.A_REVERSE)
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            time_text = f"Last updated: {current_time}"
            self.safe_addstr(7, 2, time_text, curses.color_pair(5))
        except curses.error:
            self.logger.error("Curses error in drawing header")
            
    def _draw_portfolio_info(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Portfolio Information", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            portfolio_value = self.config.get('portfolio_value', 0.0)
            self.safe_addstr(y, start_x + 2, f"Portfolio Value: ${portfolio_value:,.2f}", curses.color_pair(4))
            y += 1
            open_positions = self.config.get('open_positions', [])
            position_text = "NO open position" if not open_positions else f"{len(open_positions)} open position(s)"
            self.safe_addstr(y, start_x + 2, position_text, curses.color_pair(3) if not open_positions else curses.color_pair(4))
        except curses.error:
            self.logger.error("Curses error in drawing portfolio info")
            
    def _draw_performance_metrics(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Performance Metrics", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            metrics = self.performance_metrics
            self.safe_addstr(y, start_x + 2, f"Total Trades: {metrics['total_trades']}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Winning Trades: {metrics['winning_trades']}", curses.color_pair(1))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Losing Trades: {metrics['losing_trades']}", curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Win Rate: {metrics['win_rate']:.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Profit Factor: {metrics['profit_factor']:.4f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Max Drawdown: {metrics['max_drawdown']:.2f}%", curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Avg Profit: {metrics['avg_profit']:.2f}%", curses.color_pair(1))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Avg Loss: {metrics['avg_loss']:.2f}%", curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Daily Return: {metrics['daily_return']:.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Monthly Return: {metrics['monthly_return']:.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Yearly Return: {metrics['yearly_return']:.2f}%", curses.color_pair(4))
        except curses.error:
            self.logger.error("Curses error in drawing performance metrics")
            
    def _draw_open_positions(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Open Positions", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            self.safe_addstr(y, start_x + 2, "Sym  Size  Entry  PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            open_positions = self.config.get('open_positions', [])
            for i, pos in enumerate(open_positions[:5]):
                pos_color = curses.color_pair(1) if pos['pnl'] > 0 else curses.color_pair(2)
                text = f"{pos['symbol'][:4]} {pos['size']:.3f} ${pos['entry_price']:,.0f} ${pos['pnl']:,.2f}"
                self.safe_addstr(y + i, start_x + 2, text, pos_color)
            if not open_positions:
                self.safe_addstr(y, start_x + 2, "No open positions", curses.color_pair(3))
        except curses.error:
            self.logger.error("Curses error in drawing open positions")
            
    def _draw_recent_trades(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Recent Trades", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            self.safe_addstr(y, start_x + 2, "Time        Symbol  Side  Price  Quantity  PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            trades = list(self.config.get('recent_trades', []))[-5:]  # Limit to 5 for space
            for i, trade in enumerate(trades):
                trade_time = time.strftime("%H:%M:%S", time.localtime(trade['time'] / 1000))
                trade_color = curses.color_pair(1) if trade['pnl'] > 0 else curses.color_pair(2)
                text = f"{trade_time} {trade['symbol'][:7]} {trade['type']:4} ${trade['price']:,.0f} {trade['size']:.3f} ${trade['pnl']:,.2f}"
                self.safe_addstr(y + i, start_x + 2, text, trade_color)
            if not trades:
                self.safe_addstr(y, start_x + 2, "No recent trades", curses.color_pair(3))
        except curses.error:
            self.logger.error("Curses error in drawing recent trades")
            
    def _draw_strategy_signals(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Strategy Signals", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            for strategy, signal in self.strategy_signals.items():
                signal_color = curses.color_pair(1) if signal == "BUY" else curses.color_pair(2) if signal == "SELL" else curses.color_pair(4)
                text = f"{strategy}: {signal}"
                self.safe_addstr(y, start_x + 2, text, signal_color)
                y += 1
        except curses.error:
            self.logger.error("Curses error in drawing strategy signals")
            
    def _draw_system_info(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            y = start_y
            uptime = self.config.get('uptime', 0)
            uptime_str = f"{uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}"
            self.safe_addstr(y, start_x + 2, f"CPU: {self.config.get('cpu_usage', 0.0):,.1f}%", curses.color_pair(3))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Mem: {self.config.get('memory_usage', 0.0):,.1f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Uptime: {uptime_str}", curses.color_pair(5))
        except curses.error:
            self.logger.error("Curses error in drawing system info")
            
    def safe_addstr(self, y: int, x: int, text: str, attr=None) -> None:
        try:
            if y >= 0 and x >= 0 and y < self.stdscr.getmaxyx()[0] and x + len(text) < self.stdscr.getmaxyx()[1]:
                if attr is not None:
                    self.stdscr.addstr(y, x, text, attr)
                else:
                    self.stdscr.addstr(y, x, text)
        except curses.error:
            pass
            
    def safe_addch(self, y: int, x: int, ch: str, attr=None) -> None:
        try:
            if y >= 0 and x >= 0 and y < self.stdscr.getmaxyx()[0] and x < self.stdscr.getmaxyx()[1]:
                if attr is not None:
                    self.stdscr.addch(y, x, ch, attr)
                else:
                    self.stdscr.addch(y, x, ch)
        except curses.error:
            pass

class TradingDashboard:
    def __init__(self, logger=None, dashboard_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.dashboard_manager = dashboard_manager
        self.logger.info("Starting dashboard initialization")
        
        self.executor = BinanceExecutor(self.logger)
        self.executor.initialize()
        self.spot_client = Client(API_CONFIG['TESTNET_FUTURES_API'], API_CONFIG['TESTNET_FUTURES_SECRET'], testnet=True)
        
        self.is_testnet = not TRADING_CONFIG.get('PRODUCTION_MODE', False)
        self.logger.info(f"Running in {'Testnet' if self.is_testnet else 'Production'} mode")
        
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Initialize configuration with expanded data structures
        self.config = {
            'PRODUCTION_MODE': not self.is_testnet,
            'portfolio_value': 0.0,
            'available_margin': 0.0,
            'current_price': 0.0,
            'spot_price': 0.0,
            'price_spread': 0.0,
            'order_book': {'bids': [], 'asks': []},
            'indicators': {},
            'leverage': self.executor.current_leverage,
            'funding_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime': 0,
            'price_history': deque(maxlen=50),
            'volume_history': deque(maxlen=50),
            'recent_trades': deque(maxlen=10),
            'open_positions': [],
            'local_positions': [],  # Persistent simulated positions
            'pending_orders': [],
            'liquidation_price': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'account_risk_level': 'LOW',
            'daily_return': 0.0,
            'weekly_return': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sentiment': {
                'funding_rate': 0.0,
                'open_interest': 0.0,
                'long_short_ratio': 0.0,
                'fear_greed_index': 50
            },
            'alerts': [],
            'candlestick_data': {
                '1m': deque(maxlen=100),
                '5m': deque(maxlen=100),
                '15m': deque(maxlen=100),
                '1h': deque(maxlen=100),
                '1d': deque(maxlen=100)
            },
            'futures_contracts': {
                'weekly': {'price': 0.0, 'expiry': ''},
                'monthly': {'price': 0.0, 'expiry': ''},
                'quarterly': {'price': 0.0, 'expiry': ''}
            }
        }
        
        self.logger.info("Dashboard initialized")
        self.trade_size = 0.002
        self.ema_short_period = 9
        self.ema_long_period = 21
        self.rsi_period = 14
        self.rsi_oversold = 45
        self.rsi_overbought = 55
        self.pending_orders = []
        self.alerts = {'price_high': 80000, 'price_low': 70000}
        self.trade_history = deque(maxlen=100)
        
    def run(self):
        self.logger.info("Starting trading dashboard...")
        try:
            stdscr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            curses.init_pair(6, 213, -1)
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            stdscr.nodelay(True)
            
            dashboard = ConsoleDashboard(self.config, self.logger)
            dashboard.stdscr = stdscr
            dashboard.running = True
            
            start_time = time.time()
            while dashboard.running:
                self._update_all_data()
                self._execute_trading_strategy()
                self.config['uptime'] = int(time.time() - start_time)
                
                # Update performance metrics
                performance_metrics = self.performance_monitor.calculate_metrics()
                performance_metrics.update({
                    'total_trades': len(self.performance_monitor.trades),
                    'winning_trades': sum(1 for t in self.performance_monitor.trades if t['pnl'] > 0),
                    'losing_trades': sum(1 for t in self.performance_monitor.trades if t['pnl'] < 0),
                    'avg_profit': np.mean([t['pnl'] for t in self.performance_monitor.trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in self.performance_monitor.trades) else 0.0,
                    'avg_loss': np.mean([t['pnl'] for t in self.performance_monitor.trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in self.performance_monitor.trades) else 0.0,
                    'daily_return': 0.0,  # Placeholder
                    'monthly_return': 0.0,  # Placeholder
                    'yearly_return': 0.0  # Placeholder
                })
                dashboard.update_performance_metrics(performance_metrics)
                
                # Update strategy signals
                strategy_signals = {
                    'Technical Strategy': 'HOLD',
                    'Mean Reversion': 'HOLD',
                    'Trend Following': 'HOLD'
                }
                if self.config.get('indicators', {}).get('rsi', 0.0) < self.rsi_oversold:
                    strategy_signals['Technical Strategy'] = 'BUY'
                elif self.config.get('indicators', {}).get('rsi', 0.0) > self.rsi_overbought:
                    strategy_signals['Technical Strategy'] = 'SELL'
                dashboard.update_strategy_signals(strategy_signals)
                
                dashboard._draw_frame()
                key = stdscr.getch()
                if key == ord('q'):
                    dashboard.running = False
                elif key == ord('b'):
                    self._manual_buy()
                elif key == ord('s'):
                    self._manual_sell()
                elif key == ord('c'):
                    self._close_all_positions()
                elif key == ord('+'):
                    self._adjust_leverage(1)
                elif key == ord('-'):
                    self._adjust_leverage(-1)
                time.sleep(5)
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
        finally:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            self.logger.info("Trading dashboard stopped")
            
    def _update_all_data(self) -> None:
        self.logger.info("Updating all data from Binance")
        self.config['PRODUCTION_MODE'] = not self.is_testnet
        
        try:
            account = self.executor.client.account()
            self.config['portfolio_value'] = float(account['totalWalletBalance'])
            self.config['available_margin'] = float(account['availableBalance'])
            PORTFOLIO_VALUE_GAUGE.set(self.config['portfolio_value'])
        except Exception as e:
            self.logger.error(f"Failed to fetch account info: {e}")
        
        symbol = 'BTCUSDT'
        try:
            # Fetch current price
            ticker = self.executor.client.mark_price(symbol=symbol)
            self.config['current_price'] = float(ticker['markPrice'])
            self.config['price_history'].append(self.config['current_price'])
            PRICE_GAUGE.set(self.config['current_price'])
            
            # Fetch spot price
            spot_ticker = self.spot_client.get_symbol_ticker(symbol=symbol)
            self.config['spot_price'] = float(spot_ticker['price'])
            
            # Calculate price spread
            self.config['price_spread'] = self.config['current_price'] - self.config['spot_price']
            
            # Fetch order book
            order_book = self.executor.client.depth(symbol=symbol, limit=5)
            self.config['order_book'] = {'bids': order_book['bids'], 'asks': order_book['asks']}
            
            # Calculate and set order book metrics
            bids = order_book['bids']
            asks = order_book['asks']
            
            ORDER_BOOK_BIDS_GAUGE.set(len(bids))
            ORDER_BOOK_ASKS_GAUGE.set(len(asks))
            
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            ORDER_BOOK_BID_VOLUME_GAUGE.set(bid_volume)
            ORDER_BOOK_ASK_VOLUME_GAUGE.set(ask_volume)
            
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = best_ask - best_bid
                ORDER_BOOK_SPREAD_GAUGE.set(spread)
            
            # Fetch funding rate
            funding = self.executor.client.funding_rate(symbol=symbol, limit=1)
            self.config['funding_rate'] = float(funding[0]['fundingRate']) * 100
            FUNDING_RATE_GAUGE.set(self.config['funding_rate'])
            
            # Fetch futures contracts
            self._update_futures_contracts(symbol)
            
            # Fetch candlestick data for different timeframes
            self._update_candlestick_data(symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}", exc_info=True)
        
        self._update_technical_indicators()
        try:
            account = self.executor.client.account()
            self.logger.debug(f"Account positions: {account['positions']}")
            real_positions = []
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    size = float(pos['positionAmt'])
                    # Try different field names for entry price
                    entry_price = None
                    for field in ['avgPrice', 'entryPrice', 'price']:
                        if field in pos:
                            entry_price = float(pos[field])
                            break
                    
                    if entry_price is None:
                        self.logger.warning(f"Could not find entry price for position: {pos}")
                        continue
                        
                    leverage = float(pos['leverage'])
                    pnl = float(pos['unrealizedProfit'])
                    liquidation_price = float(pos['liquidationPrice'])
                    real_positions.append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': self.config['current_price'],
                        'leverage': leverage,
                        'pnl': pnl,
                        'liquidation_price': liquidation_price,
                        'time': int(time.time() * 1000)
                    })
            
            self.config['open_positions'] = self.config['local_positions'].copy()
            for real_pos in real_positions:
                matching_local = next((p for p in self.config['open_positions'] if p['symbol'] == real_pos['symbol'] and p['entry_price'] == real_pos['entry_price']), None)
                if matching_local:
                    self.config['open_positions'] = [p for p in self.config['open_positions'] if p != matching_local]
                self.config['open_positions'].append(real_pos)
            
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
            
            OPEN_POSITIONS_GAUGE.set(len(self.config['open_positions']))
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open")
            
            # Calculate unrealized PnL
            self.config['unrealized_pnl'] = sum(pos['pnl'] for pos in self.config['open_positions'])
            
            # Calculate liquidation price (simplified)
            if self.config['open_positions']:
                self.config['liquidation_price'] = min(pos['liquidation_price'] for pos in self.config['open_positions'])
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}", exc_info=True)
            self.config['open_positions'] = self.config['local_positions'].copy()
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
        
        try:
            trades = self.executor.client.get_account_trades(symbol=symbol, limit=10)
            self.config['recent_trades'] = deque([
                {
                    'symbol': t['symbol'],
                    'type': t['side'].lower(),
                    'size': float(t['qty']),
                    'price': float(t['price']),
                    'pnl': float(t['realizedPnl']),
                    'time': t['time']
                }
                for t in trades
            ], maxlen=10)
            for trade in trades:
                self.performance_monitor.add_trade({
                    'timestamp': datetime.fromtimestamp(trade['time'] / 1000).isoformat(),
                    'symbol': trade['symbol'],
                    'side': trade['side'].lower(),
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'pnl': float(trade['realizedPnl'])
                })
            self.logger.debug(f"Fetched recent trades: {len(self.config['recent_trades'])} trades")
            
            # Calculate realized PnL
            self.config['realized_pnl'] = sum(float(t['realizedPnl']) for t in trades)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch recent trades: {e}", exc_info=True)
        
        # Update pending orders metrics
        pending_buy_orders = [o for o in self.pending_orders if o['side'] == 'buy']
        pending_sell_orders = [o for o in self.pending_orders if o['side'] == 'sell']
        PENDING_ORDERS_GAUGE.set(len(self.pending_orders))
        PENDING_ORDERS_BUY_GAUGE.set(len(pending_buy_orders))
        PENDING_ORDERS_SELL_GAUGE.set(len(pending_sell_orders))
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update sentiment data
        self._update_sentiment_data(symbol)
        
        # Check alerts
        self._check_alerts()
        
        self.config['cpu_usage'] = psutil.cpu_percent()
        self.config['memory_usage'] = psutil.virtual_memory().percent
        CPU_USAGE_GAUGE.set(self.config['cpu_usage'])
        MEMORY_USAGE_GAUGE.set(self.config['memory_usage'])
        
    def _update_futures_contracts(self, symbol):
        """Fetch futures contract prices for different expiry periods"""
        try:
            # Fetch all futures contracts
            exchange_info = self.executor.client.exchange_info()
            futures_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].startswith(symbol) and s['symbol'] != symbol]
            
            # Group by expiry
            weekly = []
            monthly = []
            quarterly = []
            
            for sym in futures_symbols:
                if 'W' in sym:  # Weekly
                    weekly.append(sym)
                elif 'M' in sym:  # Monthly
                    monthly.append(sym)
                elif 'Q' in sym:  # Quarterly
                    quarterly.append(sym)
            
            # Get the nearest expiry for each
            if weekly:
                weekly_price = self.executor.client.mark_price(symbol=weekly[0])
                self.config['futures_contracts']['weekly'] = {
                    'price': float(weekly_price['markPrice']),
                    'expiry': weekly[0]
                }
            
            if monthly:
                monthly_price = self.executor.client.mark_price(symbol=monthly[0])
                self.config['futures_contracts']['monthly'] = {
                    'price': float(monthly_price['markPrice']),
                    'expiry': monthly[0]
                }
            
            if quarterly:
                quarterly_price = self.executor.client.mark_price(symbol=quarterly[0])
                self.config['futures_contracts']['quarterly'] = {
                    'price': float(quarterly_price['markPrice']),
                    'expiry': quarterly[0]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to fetch futures contracts: {e}")
    
    def _update_candlestick_data(self, symbol):
        """Fetch candlestick data for different timeframes"""
        timeframes = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        for tf, interval in timeframes.items():
            try:
                klines = self.executor.client.klines(symbol=symbol, interval=interval, limit=100)
                candles = []
                for k in klines:
                    candles.append({
                        'time': k[0],
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                self.config['candlestick_data'][tf] = deque(candles, maxlen=100)
            except Exception as e:
                self.logger.error(f"Failed to fetch {tf} candlestick data: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics like Sharpe ratio, win rate, etc."""
        try:
            # Get metrics from performance monitor
            metrics = self.performance_monitor.get_metrics()
            
            self.config['total_trades'] = metrics.get('total_trades', 0)
            self.config['winning_trades'] = metrics.get('winning_trades', 0)
            self.config['losing_trades'] = metrics.get('losing_trades', 0)
            self.config['win_rate'] = metrics.get('win_rate', 0.0)
            self.config['profit_factor'] = metrics.get('profit_factor', 0.0)
            self.config['sharpe_ratio'] = metrics.get('sharpe_ratio', 0.0)
            self.config['max_drawdown'] = metrics.get('max_drawdown', 0.0)
            
            # Calculate daily and weekly returns
            if len(self.trade_history) > 0:
                today = datetime.now().date()
                week_ago = today - timedelta(days=7)
                
                daily_trades = [t for t in self.trade_history if datetime.fromtimestamp(t['time'] / 1000).date() == today]
                weekly_trades = [t for t in self.trade_history if datetime.fromtimestamp(t['time'] / 1000).date() >= week_ago]
                
                daily_pnl = sum(t['pnl'] for t in daily_trades)
                weekly_pnl = sum(t['pnl'] for t in weekly_trades)
                
                self.config['daily_return'] = daily_pnl / self.config['portfolio_value'] if self.config['portfolio_value'] > 0 else 0.0
                self.config['weekly_return'] = weekly_pnl / self.config['portfolio_value'] if self.config['portfolio_value'] > 0 else 0.0
            
            # Calculate account risk level
            position_value = sum(abs(pos['size'] * pos['current_price']) for pos in self.config['open_positions'])
            margin_ratio = position_value / self.config['portfolio_value'] if self.config['portfolio_value'] > 0 else 0.0
            
            if margin_ratio > 0.8:
                self.config['account_risk_level'] = 'HIGH'
            elif margin_ratio > 0.5:
                self.config['account_risk_level'] = 'MEDIUM'
            else:
                self.config['account_risk_level'] = 'LOW'
                
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    def _update_sentiment_data(self, symbol):
        """Update sentiment data like funding rate, open interest, etc."""
        try:
            # Funding rate is already fetched in _update_all_data
            
            # Fetch open interest
            open_interest = self.executor.client.open_interest(symbol=symbol)
            self.config['sentiment']['open_interest'] = float(open_interest['openInterest'])
            
            # Fetch long/short ratio
            long_short_ratio = self.executor.client.long_short_ratio(symbol=symbol, period='5m', limit=1)
            self.config['sentiment']['long_short_ratio'] = float(long_short_ratio[0]['longShortRatio'])
            
            # Fear & Greed index (simplified calculation)
            # In a real implementation, this would come from an external API
            price_change = (self.config['current_price'] - self.config['price_history'][-2]) / self.config['price_history'][-2] if len(self.config['price_history']) > 1 else 0
            volume_change = 0  # Would need volume history
            
            # Simple calculation based on price change
            if price_change > 0.05:  # 5% increase
                self.config['sentiment']['fear_greed_index'] = 80  # Greed
            elif price_change > 0.02:  # 2% increase
                self.config['sentiment']['fear_greed_index'] = 60  # Greed
            elif price_change < -0.05:  # 5% decrease
                self.config['sentiment']['fear_greed_index'] = 20  # Fear
            elif price_change < -0.02:  # 2% decrease
                self.config['sentiment']['fear_greed_index'] = 40  # Fear
            else:
                self.config['sentiment']['fear_greed_index'] = 50  # Neutral
                
        except Exception as e:
            self.logger.error(f"Failed to update sentiment data: {e}")
    
    def _check_alerts(self):
        """Check and trigger alerts based on conditions"""
        current_price = self.config['current_price']
        
        # Price alerts
        if current_price >= self.alerts['price_high']:
            self._add_alert(f"Price alert: BTC above {self.alerts['price_high']}")
        elif current_price <= self.alerts['price_low']:
            self._add_alert(f"Price alert: BTC below {self.alerts['price_low']}")
        
        # Liquidation alerts
        for pos in self.config['open_positions']:
            if pos['liquidation_price'] > 0 and current_price <= pos['liquidation_price'] * 1.05:  # 5% buffer
                self._add_alert(f"Liquidation warning: {pos['symbol']} position at {pos['liquidation_price']}")
        
        # Risk alerts
        if self.config['account_risk_level'] == 'HIGH':
            self._add_alert("High account risk level detected")
    
    def _add_alert(self, message):
        """Add an alert to the alerts list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.config['alerts'].append(f"[{timestamp}] {message}")
        
        # Keep only the last 10 alerts
        if len(self.config['alerts']) > 10:
            self.config['alerts'] = self.config['alerts'][-10:]

    def _execute_trading_strategy(self):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping trade execution: invalid current price")
            return
        price_data = list(self.config['price_history'])
        if len(price_data) < max(self.ema_long_period, self.rsi_period):
            self.logger.debug(f"Insufficient price data for strategy: {len(price_data)} prices")
            return
        df = pd.DataFrame({'close': price_data})
        buy_signal, sell_signal = self._generate_signals(df)
        position = next((p for p in self.config['open_positions'] if p['symbol'] == 'BTCUSDT'), None)
        min_notional = 100
        quantity = max(self.trade_size, min_notional / current_price)
        
        self.logger.debug(f"Checking signals: Buy={buy_signal}, Sell={sell_signal}, Position={position is not None}")
        
        if buy_signal and not position:
            self.logger.info(f"Signal: BUY at {current_price}")
            try:
                order = self.executor.execute_buy('BTCUSDT', quantity, current_price)
                self.config['total_trades'] += 1
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'buy',
                    'size': quantity,
                    'price': current_price,
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                })
                self.performance_monitor.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'pnl': 0.0
                })
                self._update_positions()
                self.logger.info(f"Buy order executed: {order}")
            except Exception as e:
                self.logger.error(f"Failed to execute BUY order: {e}")
                new_position = {
                    'symbol': 'BTCUSDT',
                    'size': quantity,
                    'entry_price': current_price,
                    'current_price': current_price,
                    'leverage': self.config['leverage'],
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                }
                self.config['local_positions'].append(new_position)
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'buy',
                    'size': quantity,
                    'price': current_price,
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                })
                self.performance_monitor.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'pnl': 0.0
                })
                self.logger.info("Simulated BUY in paper mode")
        elif sell_signal and position:
            self.logger.info(f"Signal: SELL at {current_price}")
            try:
                order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
                self.config['total_trades'] += 1
                if position['pnl'] > 0:
                    self.config['winning_trades'] += 1
                else:
                    self.config['losing_trades'] += 1
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'sell',
                    'size': abs(position['size']),
                    'price': current_price,
                    'pnl': position['pnl'],
                    'time': int(time.time() * 1000)
                })
                self.performance_monitor.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'price': current_price,
                    'quantity': abs(position['size']),
                    'pnl': position['pnl']
                })
                self._update_positions()
                self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                self.logger.info(f"Sell order executed: {order}")
            except Exception as e:
                self.logger.error(f"Failed to execute SELL order: {e}")
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'sell',
                    'size': abs(position['size']),
                    'price': current_price,
                    'pnl': position['pnl'],
                    'time': int(time.time() * 1000)
                })
                self.performance_monitor.add_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'sell',
                    'price': current_price,
                    'quantity': abs(position['size']),
                    'pnl': position['pnl']
                })
                self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != 'BTCUSDT']
                self.logger.info("Simulated SELL in paper mode")

    def _manual_buy(self):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping manual buy: invalid price")
            return
        min_notional = 100
        quantity = max(self.trade_size, min_notional / current_price)
        try:
            order = self.executor.execute_buy('BTCUSDT', quantity, current_price)
            self.config['total_trades'] += 1
            self.config['recent_trades'].append({
                'symbol': 'BTCUSDT',
                'type': 'buy',
                'size': quantity,
                'price': current_price,
                'pnl': 0.0,
                'time': int(time.time() * 1000)
            })
            self.performance_monitor.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'price': current_price,
                'quantity': quantity,
                'pnl': 0.0
            })
            self._update_positions()
            self.logger.info(f"Manual BUY executed: {order}")
        except Exception as e:
            self.logger.error(f"Failed to execute manual BUY: {e}")
            new_position = {
                'symbol': 'BTCUSDT',
                'size': quantity,
                'entry_price': current_price,
                'current_price': current_price,
                'leverage': self.config['leverage'],
                'pnl': 0.0,
                'time': int(time.time() * 1000)
            }
            self.config['local_positions'].append(new_position)
            self.config['recent_trades'].append({
                'symbol': 'BTCUSDT',
                'type': 'buy',
                'size': quantity,
                'price': current_price,
                'pnl': 0.0,
                'time': int(time.time() * 1000)
            })
            self.performance_monitor.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'price': current_price,
                'quantity': quantity,
                'pnl': 0.0
            })
            self.logger.info("Simulated manual BUY in paper mode")

    def _manual_sell(self):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping manual sell: invalid price")
            return
        position = next((p for p in self.config['open_positions'] if p['symbol'] == 'BTCUSDT'), None)
        if not position:
            self.logger.debug("No position to sell")
            return
        try:
            order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
            self.config['total_trades'] += 1
            if position['pnl'] > 0:
                self.config['winning_trades'] += 1
            else:
                self.config['losing_trades'] += 1
            self.config['recent_trades'].append({
                'symbol': 'BTCUSDT',
                'type': 'sell',
                'size': abs(position['size']),
                'price': current_price,
                'pnl': position['pnl'],
                'time': int(time.time() * 1000)
            })
            self.performance_monitor.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'sell',
                'price': current_price,
                'quantity': abs(position['size']),
                'pnl': position['pnl']
            })
            self._update_positions()
            self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
            self.logger.info(f"Manual SELL executed: {order}")
        except Exception as e:
            self.logger.error(f"Failed to execute manual SELL: {e}")
            self.config['recent_trades'].append({
                'symbol': 'BTCUSDT',
                'type': 'sell',
                'size': abs(position['size']),
                'price': current_price,
                'pnl': position['pnl'],
                'time': int(time.time() * 1000)
            })
            self.performance_monitor.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'side': 'sell',
                'price': current_price,
                'quantity': abs(position['size']),
                'pnl': position['pnl']
            })
            self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
            self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != 'BTCUSDT']
            self.logger.info("Simulated manual SELL in paper mode")

    def _adjust_leverage(self, delta: int):
        new_leverage = max(1, min(125, self.config['leverage'] + delta))
        try:
            self.executor.set_leverage('BTCUSDT', new_leverage)
            self.config['leverage'] = new_leverage
            self.logger.info(f"Leverage adjusted to {new_leverage}x")
        except Exception as e:
            self.logger.error(f"Failed to adjust leverage: {e}")
            self.config['leverage'] = new_leverage
            self.logger.info(f"Simulated leverage adjustment to {new_leverage}x in paper mode")

    def _close_all_positions(self):
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping close all: invalid price")
            return
        for position in self.config['open_positions']:
            if position['symbol'] == 'BTCUSDT':
                try:
                    order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
                    self.config['total_trades'] += 1
                    if position['pnl'] > 0:
                        self.config['winning_trades'] += 1
                    else:
                        self.config['losing_trades'] += 1
                    self.config['recent_trades'].append({
                        'symbol': 'BTCUSDT',
                        'type': 'sell',
                        'size': abs(position['size']),
                        'price': current_price,
                        'pnl': position['pnl'],
                        'time': int(time.time() * 1000)
                    })
                    self.performance_monitor.add_trade({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': 'BTCUSDT',
                        'side': 'sell',
                        'price': current_price,
                        'quantity': abs(position['size']),
                        'pnl': position['pnl']
                    })
                    self._update_positions()
                    self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                    self.logger.info(f"Closed position: {order}")
                except Exception as e:
                    self.logger.error(f"Failed to close position: {e}")
                    self.config['recent_trades'].append({
                        'symbol': 'BTCUSDT',
                        'type': 'sell',
                        'size': abs(position['size']),
                        'price': current_price,
                        'pnl': position['pnl'],
                        'time': int(time.time() * 1000)
                    })
                    self.performance_monitor.add_trade({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': 'BTCUSDT',
                        'side': 'sell',
                        'price': current_price,
                        'quantity': abs(position['size']),
                        'pnl': position['pnl']
                    })
                    self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                    self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != 'BTCUSDT']
                    self.logger.info("Simulated close in paper mode")

    def _update_positions(self):
        try:
            # Fetch real positions from Binance
            account = self.executor.client.account()
            real_positions = []
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    size = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    leverage = float(pos['leverage'])
                    pnl = float(pos['unrealizedProfit'])
                    real_positions.append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': self.config['current_price'],
                        'leverage': leverage,
                        'pnl': pnl,
                        'time': int(time.time() * 1000)
                    })
            
            # Start with local positions
            self.config['open_positions'] = self.config['local_positions'].copy()
            
            # Add real positions, updating any matching local positions
            for real_pos in real_positions:
                matching_local = next((p for p in self.config['open_positions'] if p['symbol'] == real_pos['symbol'] and p['entry_price'] == real_pos['entry_price']), None)
                if matching_local:
                    self.config['open_positions'] = [p for p in self.config['open_positions'] if p != matching_local]
                self.config['open_positions'].append(real_pos)
            
            # Update PnL for all positions
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
            
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open - {self.config['open_positions']}")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
            self.config['open_positions'] = self.config['local_positions'].copy()
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']

    def _generate_signals(self, data: pd.DataFrame) -> tuple:
        if data.empty or len(data) < self.rsi_period:
            self.logger.warning("Insufficient data for signals")
            return False, False
        try:
            df = data.copy()
            df['ema_short'] = talib.EMA(df['close'].values, timeperiod=self.ema_short_period)
            df['ema_long'] = talib.EMA(df['close'].values, timeperiod=self.ema_long_period)
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            latest = df.iloc[-1]
            buy_signal = (
                latest['ema_short'] > latest['ema_long'] and 
                latest['rsi'] < self.rsi_oversold
            )
            sell_signal = (
                latest['ema_short'] < latest['ema_long'] and 
                latest['rsi'] > self.rsi_overbought
            )
            self.logger.info(f"Signal Check: EMA9={latest['ema_short']:.2f}, EMA21={latest['ema_long']:.2f}, RSI={latest['rsi']:.2f}, Buy={buy_signal}, Sell={sell_signal}")
            return buy_signal, sell_signal
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return False, False

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    dashboard = TradingDashboard(logger=logger)
    dashboard.run()