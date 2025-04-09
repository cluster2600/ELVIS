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
from binance.client import Client  # For spot price

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
else:
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

from trading.execution.binance_executor import BinanceExecutor
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
        
    def _draw_frame(self) -> None:
        try:
            self.stdscr.clear()
            max_y, max_x = self.stdscr.getmaxyx()
            self._draw_box(0, 0, max_y-1, max_x-1)
            
            # Header (10 lines)
            header_height = 10
            self._draw_box(0, 0, header_height, max_x-1)
            self._draw_header()
            
            # Split into two columns
            content_start = header_height + 1
            content_height = max_y - content_start - 3
            half_width = max_x // 2
            
            # Left column: Market Data, Technical Indicators
            self._draw_box(content_start, 0, content_start + content_height//2, half_width - 1)
            self._draw_market_data(content_start, half_width)
            self._draw_box(content_start + content_height//2 + 1, 0, content_start + content_height, half_width - 1)
            self._draw_technical_indicators(content_start + content_height//2 + 1, half_width)
            
            # Right column: Account Info, Risk & Performance
            self._draw_box(content_start, half_width, content_start + content_height//2, max_x - 1)
            self._draw_account_info(content_start, half_width, max_x)
            self._draw_box(content_start + content_height//2 + 1, half_width, content_start + content_height, max_x - 1)
            self._draw_risk_metrics(content_start + content_height//2 + 1, half_width, max_x)
            
            # Footer (System Info)
            system_start = max_y - 3
            self._draw_box(system_start, 0, max_y-1, max_x-1)
            self._draw_system_info()
            
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
            self._draw_elvis_logo()
            mode = "PRODUCTION" if self.config.get('PRODUCTION_MODE', False) else "TESTNET"
            mode_color = curses.color_pair(1) if mode == "PRODUCTION" else curses.color_pair(3)
            mode_text = f"=== {mode} MODE ==="
            self.safe_addstr(7, (max_x - len(mode_text)) // 2, mode_text, mode_color | curses.A_BOLD | curses.A_REVERSE)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.safe_addstr(8, max_x - len(current_time) - 2, current_time, curses.color_pair(5))
            btc_price = self.config.get('current_price', 0.0)
            price_text = f"BTC Price: ${btc_price:,.2f}"
            self.safe_addstr(9, 2, price_text, curses.color_pair(4))
        except curses.error:
            self.logger.error("Curses error in drawing header")
            
    def _draw_elvis_logo(self) -> None:
        try:
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
                if start_y + i < self.stdscr.getmaxyx()[0]:
                    x = (self.stdscr.getmaxyx()[1] - len(line)) // 2
                    color = curses.color_pair(6) if self.animation_frame < 5 else curses.color_pair(5)
                    self.safe_addstr(start_y + i, x, line, color | curses.A_BOLD)
        except curses.error:
            pass
            
    def _draw_market_data(self, start_y: int, width: int) -> None:
        try:
            self.safe_addstr(start_y, 2, "=== MARKET DATA ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            spot_price = self.config.get('spot_price', 0.0)
            futures_weekly = self.config.get('futures_weekly', 0.0)
            futures_monthly = self.config.get('futures_monthly', 0.0)
            spread = futures_weekly - spot_price
            order_book = self.config.get('order_book', {'bids': [], 'asks': []})
            self.safe_addstr(y, 2, f"Spot BTC: ${spot_price:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"Futures (W): ${futures_weekly:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"Futures (M): ${futures_monthly:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"Spread (W): ${spread:,.2f}", curses.color_pair(1) if spread > 0 else curses.color_pair(2))
            y += 1
            self.safe_addstr(y, 2, f"Bids: {len(order_book['bids'])} | Asks: {len(order_book['asks'])}", curses.color_pair(4))
        except curses.error:
            self.logger.error("Curses error in drawing market data")
            
    def _draw_technical_indicators(self, start_y: int, width: int) -> None:
        try:
            self.safe_addstr(start_y, 2, "=== TECHNICAL INDICATORS ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            indicators = self.config.get('indicators', {})
            self.safe_addstr(y, 2, f"EMA9: {indicators.get('ema_short', 0.0):,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"EMA21: {indicators.get('ema_long', 0.0):,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"RSI: {indicators.get('rsi', 0.0):.2f}", curses.color_pair(4))
            y += 1
            macd = indicators.get('macd', {'macd': 0.0, 'signal': 0.0, 'hist': 0.0})
            self.safe_addstr(y, 2, f"MACD: {macd['macd']:.2f}/{macd['signal']:.2f}", curses.color_pair(4))
            y += 1
            bb = indicators.get('bollinger', {'upper': 0.0, 'middle': 0.0, 'lower': 0.0})
            self.safe_addstr(y, 2, f"BB: {bb['upper']:,.0f}/{bb['lower']:,.0f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, 2, f"Volume: {indicators.get('volume', 0.0):,.0f}", curses.color_pair(4))
        except curses.error:
            self.logger.error("Curses error in drawing technical indicators")
            
    def _draw_account_info(self, start_y: int, start_x: int, max_x: int) -> None:
        try:
            self.safe_addstr(start_y, start_x + 2, "=== ACCOUNT INFO ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            margin = self.config.get('available_margin', 0.0)
            leverage = self.config.get('leverage', 1)
            liquidation = self.config.get('liquidation_price', 0.0)
            self.safe_addstr(y, start_x + 2, f"Margin: ${margin:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Leverage: {leverage}x", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Liquidation: ${liquidation:,.0f}", curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, "Open Positions:", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, "Symbol Size  Entry  PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            for x in range(start_x + 2, max_x - 2):
                self.safe_addch(y, x, '=', curses.color_pair(4))
            y += 1
            open_positions = self.config.get('open_positions', [])
            self.logger.debug(f"Rendering {len(open_positions)} open positions: {open_positions}")
            if open_positions:
                for i, pos in enumerate(open_positions[:3]):  # Limit to 3 for space
                    pos_color = curses.color_pair(1) if pos['pnl'] > 0 else curses.color_pair(2)
                    self.safe_addstr(y + i, start_x + 2, f"{pos['symbol']} {pos['size']:.4f} ${pos['entry_price']:,.0f} ${pos['pnl']:,.2f}", pos_color)
            else:
                self.safe_addstr(y, start_x + 2, "No open positions", curses.color_pair(3))
        except curses.error:
            self.logger.error("Curses error in drawing account info")
            
    def _draw_risk_metrics(self, start_y: int, start_x: int, max_x: int) -> None:
        try:
            self.safe_addstr(start_y, start_x + 2, "=== RISK & PERFORMANCE ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            unrealized_pnl = sum(pos['pnl'] for pos in self.config.get('open_positions', []))
            realized_pnl = sum(trade['pnl'] for trade in self.config.get('recent_trades', []))
            risk_level = self.config.get('risk_level', 0.0)
            pos_size = sum(abs(pos['size']) * pos['entry_price'] for pos in self.config.get('open_positions', []))
            margin = self.config.get('available_margin', 0.0)
            daily_return = self.config.get('daily_return', 0.0)
            self.safe_addstr(y, start_x + 2, f"Unrealized PnL: ${unrealized_pnl:,.2f}", curses.color_pair(1) if unrealized_pnl > 0 else curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Realized PnL: ${realized_pnl:,.2f}", curses.color_pair(1) if realized_pnl > 0 else curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Risk Level: {risk_level:.1f}%", curses.color_pair(2) if risk_level > 50 else curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Pos Size/Margin: ${pos_size:,.0f}/${margin:,.0f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Daily Return: {daily_return:.2f}%", curses.color_pair(1) if daily_return > 0 else curses.color_pair(2))
        except curses.error:
            self.logger.error("Curses error in drawing risk metrics")
            
    def _draw_system_info(self) -> None:
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            y = max_y - 2
            x = 2
            uptime = self.config.get('uptime', 0)
            uptime_str = f"{uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}"
            info = [
                (f"CPU: {self.config.get('cpu_usage', 0.0):.1f}%", curses.color_pair(3)),
                (f"Memory: {self.config.get('memory_usage', 0.0):.1f}%", curses.color_pair(4)),
                (f"Uptime: {uptime_str}", curses.color_pair(5))
            ]
            x_pos = x
            for text, color in info:
                self.safe_addstr(y, x_pos, text, color)
                x_pos += len(text) + 4
        except curses.error:
            pass
            
    def safe_addstr(self, y: int, x: int, text: str, attr=None) -> None:
        try:
            if attr is not None:
                self.stdscr.addstr(y, x, text, attr)
            else:
                self.stdscr.addstr(y, x, text)
        except curses.error:
            pass
            
    def safe_addch(self, y: int, x: int, ch: str, attr=None) -> None:
        try:
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
        # Use futures testnet keys for spot client
        self.spot_client = Client(API_CONFIG['TESTNET_FUTURES_API'], API_CONFIG['TESTNET_FUTURES_SECRET'], testnet=True)
        self.is_testnet = not TRADING_CONFIG.get('PRODUCTION_MODE', False)
        self.logger.info(f"Running in {'Testnet' if self.is_testnet else 'Production'} mode")
        self.config = {
            'PRODUCTION_MODE': not self.is_testnet,
            'portfolio_value': 0.0,
            'current_price': 0.0,
            'spot_price': 0.0,
            'futures_weekly': 0.0,
            'futures_monthly': 0.0,
            'order_book': {'bids': [], 'asks': []},
            'indicators': {},
            'available_margin': 0.0,
            'leverage': 1,
            'liquidation_price': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'risk_level': 0.0,
            'daily_return': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime': 0,
            'price_history': deque(maxlen=50),
            'volume_history': deque(maxlen=50),
            'recent_trades': deque(maxlen=10),
            'open_positions': [],
            'pending_orders': [],
            'local_positions': []
        }
        self.logger.info("Dashboard initialized")
        self.trade_size = 0.001
        self.ema_short_period = 9
        self.ema_long_period = 21
        self.rsi_period = 14
        self.rsi_oversold = 45
        self.rsi_overbought = 55
        self.volatility_threshold = 0.001
        self.stop_loss_pct = TRADING_CONFIG['STOP_LOSS_PCT']
        self.take_profit_pct = TRADING_CONFIG['TAKE_PROFIT_PCT']
        
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
                self._update_real_data()
                self._execute_trading_strategy()
                self.config['uptime'] = int(time.time() - start_time)
                dashboard._draw_frame()
                key = stdscr.getch()
                if key == ord('q'):
                    dashboard.running = False
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
        finally:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            self.logger.info("Trading dashboard stopped")
            
    def _update_real_data(self) -> None:
        self.logger.info("Updating real data from Binance")
        self.config['PRODUCTION_MODE'] = not self.is_testnet
        total_balance = 0.0
        try:
            account = self.executor.client.account()
            total_balance = float(account['totalWalletBalance'])
            self.config['available_margin'] = float(account['availableBalance'])
            self.logger.debug(f"Account data: {account}")
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {e}")
        
        symbol = 'BTCUSDT'
        current_price = 0.0
        spot_price = 0.0
        futures_weekly = 0.0
        futures_monthly = 0.0
        try:
            ticker = self.executor.client.mark_price(symbol)
            current_price = float(ticker['markPrice'])
            spot_ticker = self.spot_client.get_symbol_ticker(symbol=symbol)
            spot_price = float(spot_ticker['price'])
            # Fetch available futures symbols dynamically
            exchange_info = self.executor.client.get_exchange_info()
            futures_symbols = [s['symbol'] for s in exchange_info['symbols'] if 'BTCUSDT' in s['symbol'] and s['symbol'] != 'BTCUSDT']
            if futures_symbols:
                futures_weekly = float(self.executor.client.mark_price(futures_symbols[0])['markPrice'])  # First available
                futures_monthly = float(self.executor.client.mark_price(futures_symbols[-1])['markPrice'])  # Last available
            order_book = self.executor.client.get_order_book(symbol=symbol, limit=5)
            self.config['order_book'] = {'bids': order_book['bids'], 'asks': order_book['asks']}
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
        
        self.config['price_history'].append(current_price)
        self.config['volume_history'].append(float(self.executor.client.klines(symbol=symbol, interval='1m', limit=1)[0][5]))
        
        # Calculate indicators
        price_data = list(self.config['price_history'])
        if len(price_data) >= max(self.ema_long_period, self.rsi_period):
            df = pd.DataFrame({'close': price_data})
            df['ema_short'] = talib.EMA(df['close'].values, timeperiod=self.ema_short_period)
            df['ema_long'] = talib.EMA(df['close'].values, timeperiod=self.ema_long_period)
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            macd, signal, hist = talib.MACD(df['close'].values)
            upper, middle, lower = talib.BBANDS(df['close'].values)
            self.config['indicators'] = {
                'ema_short': df['ema_short'].iloc[-1],
                'ema_long': df['ema_long'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'macd': {'macd': macd[-1], 'signal': signal[-1], 'hist': hist[-1]},
                'bollinger': {'upper': upper[-1], 'middle': middle[-1], 'lower': lower[-1]},
                'volume': self.config['volume_history'][-1]
            }
        
        open_positions = []
        try:
            positions = account['positions']
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    size = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    leverage = float(pos['leverage'])
                    pnl = float(pos['unrealizedProfit'])
                    liquidation_price = float(pos.get('liquidationPrice', 0.0))
                    pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'leverage': leverage,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage,
                        'liquidation_price': liquidation_price
                    })
            self.logger.debug(f"Open positions from API: {open_positions}")
        except Exception as e:
            self.logger.error(f"Failed to process positions: {e}")
        
        # Merge with local positions
        if not open_positions and self.config['local_positions']:
            open_positions = self.config['local_positions']
            for pos in open_positions:
                pos['current_price'] = current_price
                pos['pnl'] = (current_price - pos['entry_price']) * pos['size']
                pos['pnl_percentage'] = (pos['pnl'] / (abs(pos['size']) * pos['entry_price'])) * 100 if pos['size'] != 0 and pos['entry_price'] != 0 else 0
            self.logger.debug(f"Using local positions: {open_positions}")
        
        try:
            trades = self.executor.client.get_account_trades(symbol=symbol, limit=10)
            self.logger.debug(f"Raw trades from Binance: {trades}")
            new_trades = []
            for trade in trades:
                trade_dict = {
                    'symbol': trade['symbol'],
                    'type': trade['side'],
                    'size': float(trade['qty']),
                    'price': float(trade['price']),
                    'pnl': float(trade['realizedPnl']),
                    'time': trade['time']
                }
                if not any(t['time'] == trade_dict['time'] for t in self.config['recent_trades']):
                    new_trades.append(trade_dict)
            for trade in reversed(new_trades):
                self.config['recent_trades'].appendleft(trade)
            self.logger.debug(f"Added {len(new_trades)} new trades to recent_trades")
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e} - Keeping existing trades")
        
        self.config['portfolio_value'] = total_balance
        self.config['current_price'] = current_price
        self.config['spot_price'] = spot_price
        self.config['futures_weekly'] = futures_weekly
        self.config['futures_monthly'] = futures_monthly
        self.config['open_positions'] = open_positions
        self.config['leverage'] = self.executor.current_leverage
        self.config['liquidation_price'] = open_positions[0]['liquidation_price'] if open_positions else 0.0
        self.config['risk_level'] = (sum(abs(p['size']) * p['entry_price'] for p in open_positions) / total_balance * 100) if total_balance > 0 else 0.0
        self.config['daily_return'] = (sum(trade['pnl'] for trade in self.config['recent_trades']) / total_balance * 100) if total_balance > 0 else 0.0
        self.config['cpu_usage'] = psutil.cpu_percent()
        self.config['memory_usage'] = psutil.virtual_memory().percent
        
        self.logger.debug(f"Current recent_trades: {list(self.config['recent_trades'])}")
        self.logger.debug(f"Current open_positions: {self.config['open_positions']}")
        self.logger.info(f"Updated dashboard - Portfolio: ${total_balance:.2f}, Price: ${current_price:.2f}, Trades: {len(self.config['recent_trades'])}")
    
    def _generate_signals(self, data: pd.DataFrame) -> tuple:
        if data.empty:
            self.logger.warning("Empty data provided to generate_signals")
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
                latest['ema_short'] < latest['ema_long'] or 
                latest['rsi'] > self.rsi_overbought
            )
            self.logger.info(f"Signal Check: EMA9={latest['ema_short']:.2f}, EMA21={latest['ema_long']:.2f}, RSI={latest['rsi']:.2f}, Buy={buy_signal}, Sell={sell_signal}")
            return buy_signal, sell_signal
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return False, False

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
        if self.config['total_trades'] == 0 and len(price_data) >= max(self.ema_long_period, self.rsi_period) and not position:
            buy_signal = True
        if buy_signal and not position:
            self.logger.info(f"Signal: BUY at {current_price}")
            try:
                order = self.executor.execute_buy('BTCUSDT', self.trade_size, current_price)
                self.logger.debug(f"Buy order response: {order}")
                self.config['total_trades'] += 1
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'BUY',
                    'size': self.trade_size,
                    'price': current_price,
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                })
                self._update_positions()
                if not any(p['symbol'] == 'BTCUSDT' for p in self.config['open_positions']):
                    new_position = {
                        'symbol': 'BTCUSDT',
                        'size': self.trade_size,
                        'entry_price': current_price,
                        'current_price': current_price,
                        'leverage': self.executor.current_leverage,
                        'pnl': 0.0,
                        'pnl_percentage': 0.0,
                        'liquidation_price': current_price * (1 - 1/self.executor.current_leverage)  # Simplified
                    }
                    self.config['local_positions'].append(new_position)
                    self.config['open_positions'].append(new_position)
                    self.logger.info(f"Forced local position for BTCUSDT: {new_position}")
            except Exception as e:
                self.logger.error(f"Failed to execute BUY order: {e}")
                new_position = {
                    'symbol': 'BTCUSDT',
                    'size': self.trade_size,
                    'entry_price': current_price,
                    'current_price': current_price,
                    'leverage': self.executor.current_leverage,
                    'pnl': 0.0,
                    'pnl_percentage': 0.0,
                    'liquidation_price': current_price * (1 - 1/self.executor.current_leverage)
                }
                self.config['local_positions'].append(new_position)
                self.config['open_positions'].append(new_position)
                self.logger.info(f"Added local position due to BUY failure: {new_position}")
        elif sell_signal and position:
            self.logger.info(f"Signal: SELL at {current_price}")
            try:
                order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
                self.logger.debug(f"Sell order response: {order}")
                self.config['total_trades'] += 1
                if position['pnl'] > 0:
                    self.config['winning_trades'] += 1
                else:
                    self.config['losing_trades'] += 1
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'SELL',
                    'size': abs(position['size']),
                    'price': current_price,
                    'pnl': position['pnl'],
                    'time': int(time.time() * 1000)
                })
                self._update_positions()
                self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != 'BTCUSDT']
            except Exception as e:
                self.logger.error(f"Failed to execute SELL order: {e}")
                self.config['local_positions'] = [p for p in self.config['local_positions'] if p['symbol'] != 'BTCUSDT']
                self.config['open_positions'] = [p for p in self.config['open_positions'] if p['symbol'] != 'BTCUSDT']

    def _update_positions(self):
        try:
            account = self.executor.client.account()
            self.config['open_positions'] = []
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    size = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    leverage = float(pos['leverage'])
                    pnl = float(pos['unrealizedProfit'])
                    liquidation_price = float(pos.get('liquidationPrice', 0.0))
                    pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                    self.config['open_positions'].append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': self.config['current_price'],
                        'leverage': leverage,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage,
                        'liquidation_price': liquidation_price
                    })
            for local_pos in self.config['local_positions']:
                if not any(p['symbol'] == local_pos['symbol'] for p in self.config['open_positions']):
                    local_pos['current_price'] = self.config['current_price']
                    local_pos['pnl'] = (local_pos['current_price'] - local_pos['entry_price']) * local_pos['size']
                    local_pos['pnl_percentage'] = (local_pos['pnl'] / (abs(local_pos['size']) * local_pos['entry_price'])) * 100 if local_pos['size'] != 0 and local_pos['entry_price'] != 0 else 0
                    self.config['open_positions'].append(local_pos)
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open - {self.config['open_positions']}")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
            self.config['open_positions'] = self.config['local_positions'].copy()
            for pos in self.config['open_positions']:
                pos['current_price'] = self.config['current_price']
                pos['pnl'] = (pos['current_price'] - pos['entry_price']) * pos['size']
                pos['pnl_percentage'] = (pos['pnl'] / (abs(pos['size']) * pos['entry_price'])) * 100 if pos['size'] != 0 and pos['entry_price'] != 0 else 0