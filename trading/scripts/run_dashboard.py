"""
Script to run the trading dashboard.
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
        
        self.trade_size = 0.002  # Increased to meet Binance notional minimum (100 USDT)
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
            account = self.executor.client.account()
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
            self.logger.debug("Processing positions")
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
            self.logger.warning(f"Failed to fetch trades for {symbol}: {e}")
        
        self.config['portfolio_value'] = total_balance
        self.config['current_price'] = current_price
        self.config['open_positions'] = open_positions
        self.config['recent_trades'] = recent_trades
        self.config['cpu_usage'] = psutil.cpu_percent()
        self.config['memory_usage'] = psutil.virtual_memory().percent
        
        self.logger.info(f"Updated dashboard - Mode: {'Production' if self.config['PRODUCTION_MODE'] else 'Testnet'}, Portfolio: ${total_balance:.2f}, Price: ${current_price:.2f}")
        self.logger.info(f"Open positions: {len(open_positions)}")
        for pos in open_positions:
            self.logger.info(f"Position: {pos['symbol']} - Size: {pos['size']}, PnL: ${pos['pnl']:.2f} ({pos['pnl_percentage']:.2f}%)")
        self.logger.info(f"Recent trades: {len(recent_trades)}")

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
        
        # Ensure notional value meets Binance minimum (100 USDT)
        min_notional = 100
        quantity = max(self.trade_size, min_notional / current_price)
        
        if signal == 'BUY' and not position:
            self.logger.info(f"Signal: BUY at {current_price}")
            try:
                order = self.executor.execute_buy('BTCUSDT', quantity, current_price)
                self.config['total_trades'] += 1
                self.logger.info(f"Buy order executed: {order}")
                self.config['recent_trades'].append({
                    'symbol': 'BTCUSDT',
                    'type': 'BUY',
                    'size': quantity,
                    'price': current_price,
                    'pnl': 0.0,
                    'time': int(time.time() * 1000)
                })
                self._update_positions()
            except Exception as e:
                self.logger.error(f"Failed to execute BUY order: {e}")
        
        elif signal == 'SELL' and position:
            self.logger.info(f"Signal: SELL at {current_price}")
            try:
                order = self.executor.execute_sell('BTCUSDT', abs(position['size']), current_price)
                self.config['total_trades'] += 1
                if position['pnl'] > 0:
                    self.config['winning_trades'] += 1
                else:
                    self.config['losing_trades'] += 1
                self.logger.info(f"Sell order executed: {order}")
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
                self.logger.error(f"Failed to execute SELL order: {e}")
        else:
            self.logger.debug(f"No trade action - Signal: {signal}, Position exists: {bool(position)}")

    def _update_positions(self):
        """Force update open positions after a trade."""
        try:
            account = self.executor.client.account()
            self.config['open_positions'] = []
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
                        'current_price': self.config['current_price'],
                        'leverage': leverage,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage
                    })
            self.logger.info(f"Positions updated: {len(self.config['open_positions'])} open")
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")

if __name__ == "__main__":
    # Setup logging to console and file
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"ELVIS_{time.strftime('%d_%m_%Y_%H_%M_%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # Save to logs/ELVIS_*.log
            logging.StreamHandler()             # Also print to console
        ]
    )
    dashboard = TradingDashboard()
    dashboard.run()