"""
Trading dashboard module for the ELVIS project with inline EMA-RSI trading logic.
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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.console_dashboard import ConsoleDashboard, ConsoleDashboardManager
from trading.execution.binance_executor import BinanceExecutor
from config import API_CONFIG, TRADING_CONFIG
from utils.paper_trade_db import get_open_positions, get_all_trades

class TradingDashboard:
    """Trading dashboard with automated trading functionality."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Starting dashboard initialization")
        
        self.executor = BinanceExecutor(self.logger)
        self.executor.initialize()
        
        self.is_testnet = not TRADING_CONFIG.get('PRODUCTION_MODE', False)
        self.logger.info(f"Running in {'Testnet' if self.is_testnet else 'Production'} mode")
        
        self.config = {
            'PRODUCTION_MODE': not self.is_testnet,
            'portfolio_value': 0.0,
            'position_size': 0.0,
            'entry_price': 0.0,
            'current_price': 0.0,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'market_regime': 'Unknown',
            'regime_confidence': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime': 0,
            'price_history': deque(maxlen=50),
            'volume_history': deque(maxlen=50),
            'recent_trades': deque(maxlen=10),
            'open_positions': []
        }
        
        self.dashboard_manager = ConsoleDashboardManager(self.logger, self.config)
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
        """Run the trading dashboard with trading logic."""
        try:
            self.logger.info("Starting trading dashboard...")
            self.dashboard_manager.start_dashboard()
            
            start_time = time.time()
            try:
                while True:
                    self._update_real_data()
                    self._execute_trading_strategy()
                    self.config['uptime'] = int(time.time() - start_time)
                    time.sleep(1)
                    if not self.dashboard_manager.is_running():
                        self.logger.info("Dashboard UI stopped, continuing trading logic")
            except KeyboardInterrupt:
                self.logger.info("Dashboard stopped by user")
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}")
            finally:
                self.dashboard_manager.stop_dashboard()
                self.logger.info("Trading dashboard stopped")
                
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
            raise
            
    def _update_real_data(self) -> None:
        """Update dashboard with real data from Binance."""
        self.logger.info("Updating real data from Binance")
        self.config['PRODUCTION_MODE'] = not self.is_testnet
        
        total_balance = 0.0
        try:
            self.logger.debug("Fetching account balance")
            account = self.executor.client.futures_account()
            total_balance = float(account['totalWalletBalance'])
            self.logger.debug(f"Account balance fetched: {total_balance}")
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {e}")
        
        symbol = 'BTCUSDT'
        current_price = 0.0
        try:
            self.logger.debug(f"Fetching mark price for {symbol}")
            ticker = self.executor.client.mark_price(symbol)
            current_price = float(ticker['markPrice'])
            if current_price == 0:
                self.logger.warning("Received zero price from Binance, retrying...")
                time.sleep(1)
                ticker = self.executor.client.mark_price(symbol)
                current_price = float(ticker['markPrice'])
                if current_price == 0:
                    self.logger.error("Failed to get valid price from Binance")
        except Exception as e:
            self.logger.error(f"Failed to fetch mark price for {symbol}: {e}")
        
        self.config['price_history'].append(current_price)
        
        open_positions = []
        try:
            if self.is_testnet:  # Paper trading mode - use PostgreSQL
                self.logger.debug("Processing positions from PostgreSQL")
                positions = get_open_positions()
                for pos in positions:
                    # pos structure: (id, symbol, entry_price, quantity, leverage, entry_time)
                    size = pos[3]
                    entry_price = pos[2]
                    leverage = pos[4]
                    # Calculate unrealized PnL
                    pnl = (current_price - entry_price) * size if current_price > 0 else 0
                    pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                    
                    open_positions.append({
                        'symbol': pos[1],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'leverage': leverage,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage
                    })
            else:  # Live trading mode - use Binance API
                self.logger.debug("Processing positions from Binance")
                positions = account['positions']
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        size = float(pos['positionAmt'])
                        entry_price = float(pos['entryPrice'])
                        leverage = float(pos['leverage'])
                        pnl = float(pos['unrealizedProfit'])
                        pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                        
                        open_positions.append({
                            'symbol': pos['symbol'],
                            'size': size,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'leverage': leverage,
                            'pnl': pnl,
                            'pnl_percentage': pnl_percentage
                        })
        except Exception as e:
            self.logger.error(f"Failed to process positions: {e}")
        
        recent_trades = []
        try:
            if self.is_testnet:  # Paper trading mode - use PostgreSQL
                self.logger.debug(f"Fetching recent trades from PostgreSQL")
                trades = get_all_trades(limit=10)
                for trade in trades:
                    # trade structure: (id, timestamp, symbol, side, price, quantity, pnl, fee)
                    recent_trades.append({
                        'symbol': trade[2],
                        'type': trade[3],
                        'size': trade[5],
                        'price': trade[4],
                        'pnl': trade[6],
                        'time': int(trade[1].timestamp() * 1000)  # Convert to milliseconds
                    })
            else:  # Live trading mode - use Binance API
                self.logger.debug(f"Fetching recent trades for {symbol}")
                trades = self.executor.client.get_account_trades({'symbol': symbol, 'limit': 10})
                for trade in trades:
                    recent_trades.append({
                        'symbol': trade['symbol'],
                        'type': trade['side'],
                        'size': float(trade['qty']),
                        'price': float(trade['price']),
                        'pnl': float(trade['realizedPnl']),
                        'time': trade['time']
                    })
        except Exception as e:
            self.logger.warning(f"Failed to fetch trades: {e}")
        
        self.config['portfolio_value'] = total_balance
        self.config['current_price'] = current_price
        self.config['open_positions'] = open_positions
        self.config['recent_trades'] = recent_trades
        self.config['cpu_usage'] = psutil.cpu_percent()
        self.config['memory_usage'] = psutil.virtual_memory().percent
        
        # Log and add messages to dashboard
        mode_msg = f"Mode: {'Production' if self.config['PRODUCTION_MODE'] else 'Testnet'}, Portfolio: ${total_balance:.2f}, Price: ${current_price:.2f}"
        self.logger.info(f"Updated dashboard - {mode_msg}")
        self.dashboard_manager.add_message(mode_msg)
        
        positions_msg = f"Open positions: {len(open_positions)}"
        self.logger.info(positions_msg)
        self.dashboard_manager.add_message(positions_msg)
        
        for pos in open_positions:
            pos_msg = f"Position: {pos['symbol']} - Size: {pos['size']}, PnL: ${pos['pnl']:.2f} ({pos['pnl_percentage']:.2f}%)"
            self.logger.info(pos_msg)
            self.dashboard_manager.add_message(pos_msg)
        
        trades_msg = f"Recent trades: {len(recent_trades)}"
        self.logger.info(trades_msg)
        self.dashboard_manager.add_message(trades_msg)

    def _generate_signals(self, data: pd.DataFrame) -> tuple:
        """Inline EMA-RSI signal generation."""
        if data.empty:
            self.logger.warning("Empty data provided to generate_signals")
            return False, False
        
        try:
            df = data.copy()
            df['ema_short'] = talib.EMA(df['close'].values, timeperiod=self.ema_short_period)
            df['ema_long'] = talib.EMA(df['close'].values, timeperiod=self.ema_long_period)
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            latest = df.iloc[-1]
            buy_signal = (
                latest['ema_short'] > latest['ema_long'] and 
                latest['rsi'] < self.rsi_oversold and 
                (pd.isna(latest['volatility']) or latest['volatility'] > self.volatility_threshold)
            )
            sell_signal = (
                latest['ema_short'] < latest['ema_long'] or 
                latest['rsi'] > self.rsi_overbought
            )
            
            self.logger.info(f"Signal Check: EMA9={latest['ema_short']:.2f}, EMA21={latest['ema_long']:.2f}, RSI={latest['rsi']:.2f}, Volatility={latest['volatility']:.4f if not pd.isna(latest['volatility']) else 'NaN'}, Buy={buy_signal}, Sell={sell_signal}")
            return buy_signal, sell_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return False, False

    def _execute_trading_strategy(self):
        """Execute trading strategy based on inline EMA-RSI signals."""
        current_price = self.config['current_price']
        if current_price <= 0:
            self.logger.debug("Skipping trade execution: invalid current price")
            return
        
        price_data = list(self.config['price_history'])
        if len(price_data) < 2:
            self.logger.debug(f"Insufficient price data for strategy: {len(price_data)} prices")
            return
        
        df = pd.DataFrame({'close': price_data})
        buy_signal, sell_signal = self._generate_signals(df)
        self.logger.debug(f"Signals - Buy: {buy_signal}, Sell: {sell_signal}")
        
        signal = 'BUY' if buy_signal else 'SELL' if sell_signal else None
        self.logger.debug(f"Processed signal: {signal}")
        
        position = next((p for p in self.config['open_positions'] if p['symbol'] == 'BTCUSDT'), None)
        
        if self.config['total_trades'] == 0 and len(price_data) >= 2 and not position:
            signal = 'BUY'
            self.logger.info("Forcing initial BUY for testing")
        
        if signal == 'BUY' and not position:
            buy_msg = f"Signal: BUY at ${current_price:.2f}"
            self.logger.info(buy_msg)
            self.dashboard_manager.add_message(buy_msg)
            try:
                order = self.executor.execute_buy('BTCUSDT', self.trade_size, current_price)
                self.config['total_trades'] += 1
                order_msg = f"Buy order executed: {self.trade_size} BTC at ${current_price:.2f}"
                self.logger.info(order_msg)
                self.dashboard_manager.add_message(order_msg)
                # Manually add trade to recent_trades if API fails
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'BUY',
                    'size': self.trade_size,
                    'price': current_price,
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                })
                # Refresh positions immediately
                self._update_positions()
            except Exception as e:
                error_msg = f"Failed to execute BUY order: {e}"
                self.logger.error(error_msg)
                self.dashboard_manager.add_message(error_msg)
        
        elif signal == 'SELL' and position:
            sell_msg = f"Signal: SELL at ${current_price:.2f}"
            self.logger.info(sell_msg)
            self.dashboard_manager.add_message(sell_msg)
            try:
                order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
                self.config['total_trades'] += 1
                if position['pnl'] > 0:
                    self.config['winning_trades'] += 1
                else:
                    self.config['losing_trades'] += 1
                order_msg = f"Sell order executed: {abs(position['size'])} BTC at ${current_price:.2f} | PnL: ${position['pnl']:.2f}"
                self.logger.info(order_msg)
                self.dashboard_manager.add_message(order_msg)
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'SELL',
                    'size': abs(position['size']),
                    'price': current_price,
                    'pnl': position['pnl'],
                    'time': int(time.time() * 1000)
                })
                self._update_positions()
            except Exception as e:
                error_msg = f"Failed to execute SELL order: {e}"
                self.logger.error(error_msg)
                self.dashboard_manager.add_message(error_msg)
        else:
            self.logger.debug(f"No trade action - Signal: {signal}, Position exists: {bool(position)}")

    def _update_positions(self):
        """Force update open positions after a trade."""
        try:
            self.config['open_positions'] = []
            current_price = self.config['current_price']
            
            if self.is_testnet:  # Paper trading mode - use PostgreSQL
                positions = get_open_positions()
                for pos in positions:
                    # pos structure: (id, symbol, entry_price, quantity, leverage, entry_time)
                    size = pos[3]
                    entry_price = pos[2]
                    leverage = pos[4]
                    pnl = (current_price - entry_price) * size if current_price > 0 else 0
                    pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                    
                    self.config['open_positions'].append({
                        'symbol': pos[1],
                        'size': size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'leverage': leverage,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage
                    })
            else:  # Live trading mode - use Binance API
                account = self.executor.client.futures_account()
                for pos in account['positions']:
                    if float(pos['positionAmt']) != 0:
                        size = float(pos['positionAmt'])
                        entry_price = float(pos['entryPrice'])
                        leverage = float(pos['leverage'])
                        pnl = float(pos['unrealizedProfit'])
                        pnl_percentage = (pnl / (abs(size) * entry_price)) * 100 if size != 0 and entry_price != 0 else 0
                        self.config['open_positions'].append({
                            'symbol': pos['symbol'],
                            'size': size,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'leverage': leverage,
                            'pnl': pnl,
                            'pnl_percentage': pnl_percentage
                        })
                        
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dashboard = TradingDashboard()
    dashboard.run()
