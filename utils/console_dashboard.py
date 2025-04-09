import curses
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import deque

class ConsoleDashboard:
    """Console-based trading dashboard."""
    
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
            header_height = 8
            self._draw_box(0, 0, header_height, max_x-1)
            self._draw_header()
            content_start = header_height + 1
            content_height = max_y - content_start - 3
            half_width = max_x // 2
            self._draw_box(content_start, 0, content_start + content_height, half_width - 1)
            self._draw_open_positions(content_start, half_width)
            self._draw_box(content_start, half_width, content_start + content_height, max_x - 1)
            self._draw_trades_list(content_start, half_width, max_x)
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
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.safe_addstr(8, max_x - len(current_time) - 2, current_time, curses.color_pair(5))
        except curses.error:
            pass
            
    def _draw_elvis_logo(self) -> None:
        try:
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
                if start_y + i < self.stdscr.getmaxyx()[0]:
                    x = (self.stdscr.getmaxyx()[1] - len(line)) // 2
                    color = curses.color_pair(6) if self.animation_frame < 5 else curses.color_pair(5)
                    self.safe_addstr(start_y + i, x, line, color | curses.A_BOLD)
        except curses.error:
            pass
            
    def _draw_open_positions(self, start_y: int, width: int) -> None:
        try:
            self.safe_addstr(start_y, 2, "=== OPEN POSITIONS ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            portfolio_value = self.config.get('portfolio_value', 0.0)
            self.safe_addstr(y, 2, f"Portfolio: ${portfolio_value:,.2f}", curses.color_pair(4))
            y += 2
            self.safe_addstr(y, 2, "Symbol", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, 12, "Size", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, 22, "Entry", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, 32, "PnL", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            for x in range(2, width - 2):
                self.safe_addch(y, x, '=', curses.color_pair(4))
            y += 1
            open_positions = self.config.get('open_positions', [])
            self.logger.debug(f"Displaying {len(open_positions)} open positions")
            if open_positions:
                for i, pos in enumerate(open_positions):
                    if y + i >= start_y + content_height - 1:
                        break
                    pos_color = curses.color_pair(1) if pos['pnl'] > 0 else curses.color_pair(2)
                    self.safe_addstr(y + i, 2, f"{pos['symbol']}", curses.color_pair(4))
                    self.safe_addstr(y + i, 12, f"{pos['size']:.4f}", curses.color_pair(4))
                    self.safe_addstr(y + i, 22, f"${pos['entry_price']:,.0f}", curses.color_pair(4))
                    self.safe_addstr(y + i, 32, f"${pos['pnl']:,.2f}", pos_color)
            else:
                self.safe_addstr(y, 2, "No open positions", curses.color_pair(3))
        except curses.error:
            self.logger.error("Curses error in drawing open positions")
            
    def _draw_trades_list(self, start_y: int, start_x: int, max_x: int) -> None:
        try:
            self.logger.debug("Drawing trades list")
            content_height = self.stdscr.getmaxyx()[0] - start_y - 3
            self.safe_addstr(start_y, start_x + 2, "=== TRADE HISTORY ===", curses.color_pair(4) | curses.A_BOLD | curses.A_REVERSE)
            y = start_y + 1
            self.safe_addstr(y, start_x + 2, "Symbol", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, start_x + 12, "Type", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, start_x + 20, "Size", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, start_x + 30, "Price", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, start_x + 40, "PnL", curses.color_pair(4) | curses.A_BOLD)
            self.safe_addstr(y, start_x + 50, "Time", curses.color_pair(4) | curses.A_BOLD)
            y += 1
            for x in range(start_x + 2, max_x - 2):
                self.safe_addch(y, x, '=', curses.color_pair(4))
            y += 1
            recent_trades = list(self.config.get('recent_trades', []))
            self.logger.debug(f"Displaying {len(recent_trades)} trades")
            if recent_trades:
                recent_trades.sort(key=lambda t: t['time'], reverse=True)
                for i, trade in enumerate(recent_trades[:10]):
                    if y + i >= start_y + content_height - 1:
                        break
                    type_color = curses.color_pair(1) if trade['type'] == 'BUY' else curses.color_pair(2)
                    self.safe_addstr(y + i, start_x + 2, f"{trade['symbol']}", curses.color_pair(4))
                    self.safe_addstr(y + i, start_x + 12, f"{trade['type']}", type_color)
                    self.safe_addstr(y + i, start_x + 20, f"{trade['size']:.4f}", curses.color_pair(4))
                    self.safe_addstr(y + i, start_x + 30, f"${trade['price']:,.0f}", curses.color_pair(4))
                    pnl_color = curses.color_pair(1) if trade['pnl'] > 0 else curses.color_pair(2)
                    self.safe_addstr(y + i, start_x + 40, f"${trade['pnl']:,.2f}", pnl_color)
                    time_str = datetime.fromtimestamp(trade['time'] / 1000).strftime('%H:%M:%S')
                    self.safe_addstr(y + i, start_x + 50, time_str, curses.color_pair(4))
            else:
                self.safe_addstr(y, start_x + 2, "No trades available", curses.color_pair(3))
        except curses.error:
            self.logger.error("Curses error in drawing trades list")
            
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

class ConsoleDashboardManager:
    """Manages the console dashboard."""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config
        self.dashboard = None
        self.running = False
        self.thread = None
        
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
        
    def update_position(self, size: float, entry_price: float, current_price: float):
        if self.dashboard:
            self.dashboard.config.update({
                'position_size': size,
                'entry_price': entry_price,
                'current_price': current_price
            })
            if size != 0 and entry_price != 0:
                pnl = (current_price - entry_price) * size
                pnl_pct = (pnl / (size * entry_price)) * 100
                self.dashboard.config.update({
                    'unrealized_pnl': pnl,
                    'unrealized_pnl_pct': pnl_pct
                })
        
    def update_metrics(self, metrics: Dict[str, Any]):
        if self.dashboard:
            self.dashboard.config.update(metrics)
        
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
        
    def set_model_name(self, model_name: str):
        if self.dashboard:
            self.dashboard.config['ml_model'] = model_name
        
    def _run_dashboard(self):
        try:
            stdscr = curses.initscr()
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.curs_set(0)
                curses.noecho()
                try:
                    curses.cbreak()
                except curses.error as e:
                    self.logger.error(f"Failed to enable cbreak mode: {e}")
                    return
                stdscr.keypad(True)
                stdscr.nodelay(True)
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_RED, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_CYAN, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
                curses.init_pair(6, 213, -1)
                if not self.dashboard:
                    self.dashboard = ConsoleDashboard(self.config)
                self.dashboard.stdscr = stdscr
                while self.running:
                    self.dashboard._draw_frame()
                    key = stdscr.getch()
                    if key == ord('q'):
                        break
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"Curses error during initialization: {e}")
            finally:
                try:
                    curses.nocbreak()
                    stdscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                except Exception as e:
                    self.logger.error(f"Error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize curses: {e}")
        finally:
            self.running = False

if __name__ == "__main__":
    import curses.wrapper
    def main(stdscr):
        dashboard = ConsoleDashboard({
            'PRODUCTION_MODE': False,
            'portfolio_value': 0.0,
            'open_positions': [],
            'recent_trades': [],
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime': 0
        })
        dashboard.stdscr = stdscr
        dashboard.run()
    curses.wrapper(main)