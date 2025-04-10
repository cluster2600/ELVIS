"""
Utilities package for the ELVIS project.
ELVIS: Enhanced Leveraged Virtual Investment System
"""

import curses
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque

class ConsoleDashboard:
    """Enhanced console-based trading dashboard."""
    
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
            if max_y < 40 or max_x < 100:
                self.stdscr.addstr(0, 0, "Terminal too small, resize to at least 100x40")
                self.stdscr.refresh()
                return

            # Header
            header_height = 8
            self._draw_box(0, 0, header_height, max_x-1)
            self._draw_header()

            # Main content layout
            content_start = header_height + 1
            content_height = max_y - content_start - 5
            third_width = max_x // 3

            # Left: Market Data + Indicators
            self._draw_box(content_start, 0, content_start + content_height, third_width - 1)
            self._draw_market_data(content_start + 1, 0, third_width - 1)
            self._draw_technical_indicators(content_start + 10, 0, third_width - 1)

            # Middle: Account Info + Open Positions
            self._draw_box(content_start, third_width, content_start + content_height, 2 * third_width - 1)
            self._draw_account_info(content_start + 1, third_width, 2 * third_width - 1)
            self._draw_open_positions(content_start + 8, third_width, 2 * third_width - 1)

            # Right: Trades + Actions + Metrics
            self._draw_box(content_start, 2 * third_width, content_start + content_height, max_x - 1)
            self._draw_recent_trades(content_start + 1, 2 * third_width, max_x - 1)
            self._draw_trading_actions(content_start + 10, 2 * third_width, max_x - 1)
            self._draw_performance_metrics(content_start + 15, 2 * third_width, max_x - 1)

            # Footer: Sentiment + System Info
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
                "██╔══╝  ██║     ██║   ██║██║╚════██║",
                "███████╗███████╗╚██████╔╝██║███████║",
                "╚══════╝╚══════╝ ╚═════╝ ╚═╝╚══════╝"
            ]
            start_y = 1
            for i, line in enumerate(logo):
                x = (max_x - len(line)) // 2
                color = curses.color_pair(6) if self.animation_frame < 5 else curses.color_pair(5)
                self.safe_addstr(start_y + i, x, line, color | curses.A_BOLD)
            
            mode = "PRODUCTION" if self.config.get('PRODUCTION_MODE', False) else "TESTNET"
            mode_color = curses.color_pair(1) if mode == "PRODUCTION" else curses.color_pair(3)
            mode_text = f"=== {mode} MODE ==="
            self.safe_addstr(7, (max_x - len(mode_text)) // 2, mode_text, mode_color | curses.A_BOLD | curses.A_REVERSE)
        except curses.error:
            pass

    def _draw_market_data(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Market Data", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            spot_price = self.config.get('spot_price', 0.0)
            futures_price = self.config.get('current_price', 0.0)
            spread = futures_price - spot_price
            self.safe_addstr(y, start_x + 2, f"Spot BTC: ${spot_price:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Futures: ${futures_price:,.2f}", curses.color_pair(4))
            y += 1
            spread_color = curses.color_pair(1) if spread > 0 else curses.color_pair(2)
            self.safe_addstr(y, start_x + 2, f"Spread: ${spread:,.2f}", spread_color)
            y += 1
            order_book = self.config.get('order_book', {'bids': [], 'asks': []})
            if order_book['bids']:
                top_bid = float(order_book['bids'][0][0])
                self.safe_addstr(y, start_x + 2, f"Top Bid: ${top_bid:,.2f}", curses.color_pair(1))
            y += 1
            if order_book['asks']:
                top_ask = float(order_book['asks'][0][0])
                self.safe_addstr(y, start_x + 2, f"Top Ask: ${top_ask:,.2f}", curses.color_pair(2))
        except curses.error:
            self.logger.error("Error drawing market data")

    def _draw_technical_indicators(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Indicators", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            indicators = self.config.get('indicators', {})
            ema_short = indicators.get('ema_short', 0.0)
            ema_long = indicators.get('ema_long', 0.0)
            rsi = indicators.get('rsi', 0.0)
            macd = indicators.get('macd', {'macd': 0.0, 'signal': 0.0})
            sma = indicators.get('sma', 0.0)
            bb_upper = indicators.get('bb_upper', 0.0)
            bb_lower = indicators.get('bb_lower', 0.0)
            volume = indicators.get('volume', 0.0)
            self.safe_addstr(y, start_x + 2, f"EMA 9: {ema_short:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"EMA 21: {ema_long:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"SMA 20: {sma:,.2f}", curses.color_pair(4))
            y += 1
            rsi_color = curses.color_pair(1) if rsi < 30 else curses.color_pair(2) if rsi > 70 else curses.color_pair(4)
            self.safe_addstr(y, start_x + 2, f"RSI: {rsi:.2f}", rsi_color)
            y += 1
            macd_color = curses.color_pair(1) if macd['macd'] > macd['signal'] else curses.color_pair(2)
            self.safe_addstr(y, start_x + 2, f"MACD: {macd['macd']:.2f}/{macd['signal']:.2f}", macd_color)
            y += 1
            self.safe_addstr(y, start_x + 2, f"BB: {bb_upper:,.2f}/{bb_lower:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Vol: {volume:,.0f}", curses.color_pair(4))
        except curses.error:
            self.logger.error("Error drawing indicators")

    def _draw_account_info(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Account Info", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            portfolio_value = self.config.get('portfolio_value', 0.0)
            available_margin = self.config.get('available_margin', portfolio_value)
            leverage = self.config.get('leverage', 1)
            positions = self.config.get('open_positions', [])
            total_pnl = sum(pos['pnl'] for pos in positions)
            liquidation_price = 0  # Simplified calculation placeholder
            if positions:
                pos = positions[0]
                liquidation_price = pos['entry_price'] - (pos['entry_price'] * pos['size'] / (portfolio_value * leverage))
            self.safe_addstr(y, start_x + 2, f"Equity: ${portfolio_value:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Margin: ${available_margin:,.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Leverage: {leverage}x", curses.color_pair(4))
            y += 1
            pnl_color = curses.color_pair(1) if total_pnl > 0 else curses.color_pair(2)
            self.safe_addstr(y, start_x + 2, f"Unreal PnL: ${total_pnl:,.2f}", pnl_color)
            y += 1
            self.safe_addstr(y, start_x + 2, f"Liq Price: ${liquidation_price:,.2f}", curses.color_pair(3))
        except curses.error:
            self.logger.error("Error drawing account info")

    def _draw_open_positions(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Open Positions", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            self.safe_addstr(y, start_x + 2, "Sym Size Entry PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            positions = self.config.get('open_positions', [])
            for i, pos in enumerate(positions[:5]):
                pos_color = curses.color_pair(1) if pos['pnl'] > 0 else curses.color_pair(2)
                text = f"{pos['symbol'][:4]} {pos['size']:.3f} ${pos['entry_price']:,.0f} ${pos['pnl']:,.2f}"
                self.safe_addstr(y + i, start_x + 2, text, pos_color)
            if not positions:
                self.safe_addstr(y, start_x + 2, "No open positions", curses.color_pair(3))
        except curses.error:
            self.logger.error("Error drawing open positions")

    def _draw_recent_trades(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Recent Trades", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            self.safe_addstr(y, start_x + 2, "Time Symbol Side Price Qty PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            trades = list(self.config.get('recent_trades', []))[-5:]
            for i, trade in enumerate(trades):
                trade_time = time.strftime("%H:%M:%S", time.localtime(trade['time'] / 1000))
                trade_color = curses.color_pair(1) if trade['pnl'] > 0 else curses.color_pair(2)
                text = f"{trade_time} {trade['symbol'][:6]} {trade['type'][:4]} ${trade['price']:,.0f} {trade['size']:.2f} ${trade['pnl']:,.2f}"
                self.safe_addstr(y + i, start_x + 2, text, trade_color)
            if not trades:
                self.safe_addstr(y, start_x + 2, "No recent trades", curses.color_pair(3))
        except curses.error:
            self.logger.error("Error drawing recent trades")

    def _draw_trading_actions(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Trading Actions", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            self.safe_addstr(y, start_x + 2, "[B] Buy [S] Sell [C] Close All", curses.color_pair(1))
            y += 1
            self.safe_addstr(y, start_x + 2, "[L] Limit [T] Stop [P] TP [O] SL", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, "[+] Lev Up [-] Lev Down", curses.color_pair(4))
        except curses.error:
            self.logger.error("Error drawing trading actions")

    def _draw_performance_metrics(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            self.safe_addstr(start_y - 1, start_x + 2, "Performance", curses.color_pair(4) | curses.A_BOLD)
            y = start_y
            metrics = self.config.get('metrics', {})
            pos = self.config.get('open_positions', [])
            total_margin = sum(p['size'] * p['entry_price'] / p['leverage'] for p in pos)
            risk_level = "High" if total_margin > self.config['portfolio_value'] * 0.5 else "Low"
            self.safe_addstr(y, start_x + 2, f"Win Rate: {metrics.get('win_rate', 0.0):.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Sharpe: {metrics.get('sharpe_ratio', 0.0):.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Max DD: {metrics.get('max_drawdown', 0.0):.2f}%", curses.color_pair(2))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Daily Ret: {metrics.get('daily_return', 0.0):.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Weekly Ret: {metrics.get('weekly_return', 0.0):.2f}%", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Risk Level: {risk_level}", curses.color_pair(2 if risk_level == "High" else 4))
        except curses.error:
            self.logger.error("Error drawing performance metrics")

    def _draw_system_info(self, start_y: int, start_x: int, end_x: int) -> None:
        try:
            y = start_y
            uptime = self.config.get('uptime', 0)
            uptime_str = f"{uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}"
            self.safe_addstr(y, start_x + 2, f"CPU: {self.config.get('cpu_usage', 0.0):.1f}%", curses.color_pair(3))
            self.safe_addstr(y, start_x + 20, f"Mem: {self.config.get('memory_usage', 0.0):.1f}%", curses.color_pair(4))
            self.safe_addstr(y, start_x + 40, f"Uptime: {uptime_str}", curses.color_pair(5))
            y += 1
            sentiment = self.config.get('sentiment', {})
            funding_rate = sentiment.get('funding_rate', 0.0)
            funding_color = curses.color_pair(1) if funding_rate > 0 else curses.color_pair(2)
            self.safe_addstr(y, start_x + 2, f"Funding: {funding_rate:.4f}%", funding_color)
            self.safe_addstr(y, start_x + 20, f"OI: {sentiment.get('open_interest', 0.0):,.0f}", curses.color_pair(4))
            self.safe_addstr(y, start_x + 40, f"L/S: {sentiment.get('long_short_ratio', 0.0):.2f}", curses.color_pair(4))
            y += 1
            self.safe_addstr(y, start_x + 2, f"Fear/Greed: {sentiment.get('fear_greed_index', 50)}", curses.color_pair(4))
        except curses.error:
            self.logger.error("Error drawing system info")

    def run(self) -> None:
        try:
            self.stdscr = curses.initscr()
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

class ConsoleDashboardManager:
    """Manages the console dashboard."""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        self.dashboard = None
        self.running = False
        self.thread = None
        self.actions = {}

    def start_dashboard(self):
        self.logger.info("Starting console dashboard...")
        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard)
        self.thread.daemon = True
        self.thread.start()

    def stop_dashboard(self):
        self.logger.info("Stopping console dashboard...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def is_running(self):
        return self.running and self.thread and self.thread.is_alive()

    def update_portfolio_value(self, value: float):
        if self.dashboard:
            self.dashboard.config['portfolio_value'] = value

    def update_metrics(self, metrics: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.config['metrics'] = metrics

    def update_strategy_signals(self, signals: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.config['strategy_signals'] = signals

    def update_open_positions(self, positions: List[Dict[str, Any]]):
        if self.dashboard:
            self.dashboard.config['open_positions'] = positions

    def add_trade(self, trade: Dict[str, Any]):
        if self.dashboard:
            if 'recent_trades' not in self.dashboard.config:
                self.dashboard.config['recent_trades'] = deque(maxlen=10)
            self.dashboard.config['recent_trades'].append(trade)

    def update_market_data(self, data: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.config.update(data)

    def register_action(self, key: str, callback: callable):
        self.actions[key] = callback

    def _run_dashboard(self):
        try:
            stdscr = curses.initscr()
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.curs_set(0)
                curses.noecho()
                curses.cbreak()
                stdscr.keypad(True)
                stdscr.nodelay(True)
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_RED, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_CYAN, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
                curses.init_pair(6, 213, -1)
                if not self.dashboard:
                    self.dashboard = ConsoleDashboard(self.config, self.logger)
                self.dashboard.stdscr = stdscr
                while self.running:
                    self.dashboard._draw_frame()
                    key = stdscr.getch()
                    if key == ord('q'):
                        break
                    elif chr(key).lower() in self.actions:
                        self.actions[chr(key).lower()]()
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Curses error during initialization: {e}")
            finally:
                curses.nocbreak()
                stdscr.keypad(False)
                curses.echo()
                curses.endwin()
        except Exception as e:
            self.logger.error(f"Failed to initialize curses: {e}")
        finally:
            self.running = False

if __name__ == "__main__":
    import curses.wrapper
    def main(stdscr):
        logger = logging.getLogger(__name__)
        dashboard = ConsoleDashboard({
            'PRODUCTION_MODE': False,
            'portfolio_value': 10000.0,
            'open_positions': [],
            'recent_trades': [],
            'current_price': 75000.0,
            'spot_price': 74950.0,
            'order_book': {'bids': [['74900', '1']], 'asks': [['75050', '1']]},
            'indicators': {'ema_short': 75010, 'ema_long': 74980, 'rsi': 55, 'macd': {'macd': 30, 'signal': 25}},
            'metrics': {'win_rate': 60.0, 'sharpe_ratio': 1.5, 'max_drawdown': 5.0},
            'funding_rate': 0.01,
            'cpu_usage': 10.0,
            'memory_usage': 20.0,
            'uptime': 0
        }, logger)
        dashboard.stdscr = stdscr
        dashboard.run()
    curses.wrapper(main)