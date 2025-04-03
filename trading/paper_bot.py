"""
Paper trading bot for the ELVIS project.
This module provides a paper trading implementation for testing strategies without real money.
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from utils.price_fetcher import PriceFetcher
from trading.strategies.base_strategy import BaseStrategy
from trading.risk.risk_manager import RiskManager
from trading.performance_monitor import PerformanceMonitor
from config import TRADING_CONFIG
import psutil

class PaperBot:
    def __init__(self, symbol: str, timeframe: str, leverage: int, strategy=None, logger: Optional[logging.Logger] = None, dashboard_manager=None):
        """Initialize the paper trading bot."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.dashboard_manager = dashboard_manager
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Initialize other attributes
        self.position = 0
        self.entry_price = 0.0
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.portfolio_value = TRADING_CONFIG['MIN_CAPITAL_USD']
        self.running = False
        
        self.logger.info("Initializing paper trading bot...")
        self.data_history = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        self.history_lock = threading.Lock()
        self.price_fetcher = PriceFetcher(self.logger, symbol=self.symbol, timeframe='1m')
        self.price_fetcher.start()
        
        if not isinstance(strategy, BaseStrategy):
            self.logger.error(f"Invalid strategy object provided: {type(strategy)}. Must inherit from BaseStrategy.")
            raise TypeError("Strategy must inherit from BaseStrategy")
        
        self.risk_manager = RiskManager(self.logger)
        
        # Use the provided dashboard manager or create a new one
        if dashboard_manager is None:
            from utils.console_dashboard import ConsoleDashboardManager
            self.dashboard_manager = ConsoleDashboardManager(self.logger)
        
        self.logger.info(f"Paper trading bot initialized for {symbol} on {timeframe} timeframe with {leverage}x leverage using {self.strategy.__class__.__name__}")

    def _process_new_candle(self, candle: Dict[str, Any]):
        if not candle or candle.get('close') is None or candle.get('close') <= 0:
            self.logger.debug(f"Received invalid or empty candle: {candle}")
            return

        current_price = candle['close']
        # Use current time since PriceFetcher doesn't provide timestamp
        candle_time = pd.Timestamp.now()

        new_data = pd.DataFrame([{
            'date': candle_time,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        
        with self.history_lock:
            self.data_history = pd.concat([self.data_history, new_data], ignore_index=True)
            self.data_history = self.data_history.iloc[-200:]
            strategy_data = self.data_history.copy()

        if not strategy_data.empty and len(strategy_data) >= 50:
            try:
                signals = self.strategy.generate_signals(strategy_data)
                # Handle different strategy output formats
                if isinstance(signals, tuple):  # For strategies like EmaRsiStrategy
                    buy_signal, sell_signal = signals
                else:  # For EnsembleStrategy
                    buy_signal = signals.get("signal") == "BUY"
                    sell_signal = signals.get("signal") == "SELL"
                
                if buy_signal and self.position == 0:
                    self.execute_buy(current_price, strategy_data)
                elif sell_signal and self.position > 0:
                    self.execute_sell(current_price)
            except Exception as e:
                self.logger.error(f"Error during signal generation or trade execution: {e}", exc_info=True)

        if self.position > 0:
            self.check_exit_conditions(current_price)

        self._update_console_dashboard(strategy_data, candle)

    def run(self):
        """Run the paper trading bot."""
        self.logger.info("Starting paper trading bot...")
        self.running = True
        self.start_time = time.time()
        
        # Start the dashboard in a separate thread
        if self.dashboard_manager:
            self.logger.info("Starting dashboard...")
            dashboard_thread = threading.Thread(target=self.dashboard_manager.start_dashboard)
            dashboard_thread.daemon = True
            dashboard_thread.start()
            time.sleep(1)  # Wait for dashboard to initialize
            
            # Verify dashboard is running
            if not self.dashboard_manager.is_running():
                self.logger.error("Failed to start dashboard")
                self.running = False
                return
        
        # Create mock positions for testing the dashboard
        if TRADING_CONFIG.get('CREATE_MOCK_POSITION', True):  # Default to True
            self.logger.info("Creating mock positions for testing...")
            
            # Create multiple positions with different entry prices and sizes
            current_price = self.price_fetcher.get_current_price()
            if current_price == 0:
                current_price = self.price_fetcher.default_price
            
            # Main position (BTC)
            self.position = 0.01  # 0.01 BTC
            self.entry_price = 75000.0  # $75,000 per BTC
            
            # Define crypto symbols, prices, and sizes for mock positions
            crypto_data = [
                {'symbol': self.symbol, 'entry_price': 75000.0, 'current_price': current_price, 'size': 0.01},
                {'symbol': 'ETH/USDT', 'entry_price': 3800.0, 'current_price': 3850.0, 'size': 0.15},
                {'symbol': 'SOL/USDT', 'entry_price': 150.0, 'current_price': 145.0, 'size': 2.5},
                {'symbol': 'BNB/USDT', 'entry_price': 580.0, 'current_price': 595.0, 'size': 0.5},
                {'symbol': 'ADA/USDT', 'entry_price': 0.45, 'current_price': 0.47, 'size': 1000},
                {'symbol': 'DOT/USDT', 'entry_price': 7.2, 'current_price': 7.0, 'size': 50},
                {'symbol': 'AVAX/USDT', 'entry_price': 35.0, 'current_price': 36.5, 'size': 10},
                {'symbol': 'MATIC/USDT', 'entry_price': 0.65, 'current_price': 0.63, 'size': 500}
            ]
            
            # Create positions based on MOCK_POSITIONS_COUNT config
            positions_count = min(TRADING_CONFIG.get('MOCK_POSITIONS_COUNT', 5), len(crypto_data))
            
            # Generate open positions
            open_positions = []
            for i in range(positions_count):
                crypto = crypto_data[i]
                pnl = (crypto['current_price'] - crypto['entry_price']) * crypto['size']
                pnl_pct = ((crypto['current_price'] / crypto['entry_price']) - 1) * 100 if crypto['entry_price'] > 0 else 0.0
                
                open_positions.append({
                    'symbol': crypto['symbol'],
                    'size': crypto['size'],
                    'entry_price': crypto['entry_price'],
                    'current_price': crypto['current_price'],
                    'leverage': self.leverage,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
            
            # Update portfolio value based on all positions
            self.portfolio_value = TRADING_CONFIG['MIN_CAPITAL_USD'] - sum(pos['size'] * pos['entry_price'] for pos in open_positions)
            
            # Update the dashboard with the mock positions
            if self.dashboard_manager:
                self.dashboard_manager.update_open_positions(open_positions)
            
            # Add mock trades to the performance monitor
            self.logger.info("Adding mock trades for testing...")
            
            # Generate a series of mock trades over the past 24 hours
            trades_count = TRADING_CONFIG.get('MOCK_TRADES_COUNT', 50)
            self.logger.info(f"Generating {trades_count} mock trades...")
            
            # Generate trades for BTC
            for i in range(trades_count):
                # Calculate time (spread trades over 24 hours)
                minutes_ago = int((24 * 60) * (i / trades_count))
                
                # Alternate buy and sell trades
                side = 'buy' if i % 2 == 0 else 'sell'
                
                # Generate realistic price movements
                base_price = 74000.0 + (i * 20)  # Gradually increasing price
                price_noise = np.random.normal(0, 200)  # Random noise
                price = base_price + price_noise
                
                # Vary quantity slightly
                quantity = 0.01 * (1 + np.random.uniform(-0.2, 0.2))
                
                # Calculate PnL for sell trades
                pnl = 0.0
                
                if side == 'sell' and i > 0:
                    prev_price = 74000.0 + ((i-1) * 20) + np.random.normal(0, 200)
                    pnl = (price - prev_price) * quantity
                
                # Add the trade
                self.performance_monitor.add_trade({
                    'timestamp': (datetime.now() - timedelta(minutes=minutes_ago)).isoformat(),
                    'symbol': self.symbol,
                    'side': side,
                    'price': price,
                    'quantity': quantity,
                    'pnl': pnl if side == 'sell' else 0.0
                })
            
            # Add a few trades for other symbols
            for symbol, entry_price in [('ETH/USDT', 3800.0), ('SOL/USDT', 150.0)]:
                # Buy trade
                self.performance_monitor.add_trade({
                    'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                    'symbol': symbol,
                    'side': 'buy',
                    'price': entry_price,
                    'quantity': 0.5 if symbol == 'ETH/USDT' else 2.0,
                    'pnl': 0.0
                })
                
                # Sell trade
                exit_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
                quantity = 0.5 if symbol == 'ETH/USDT' else 2.0
                pnl = (exit_price - entry_price) * quantity
                
                self.performance_monitor.add_trade({
                    'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                    'symbol': symbol,
                    'side': 'sell',
                    'price': exit_price,
                    'quantity': quantity,
                    'pnl': pnl
                })
            
            # Add the current open position as the most recent buy
            self.performance_monitor.add_trade({
                'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'symbol': self.symbol,
                'side': 'buy',
                'price': self.entry_price,
                'quantity': self.position,
                'pnl': 0.0
            })

        try:
            last_update = time.time()
            while self.running and not self.stop_event.is_set():
                # Get current candle
                candle = self.price_fetcher.get_current_candle()
                if candle:
                    self.logger.debug(f"Processing candle: {candle}")
                    self._process_new_candle(candle)
                
                # Update dashboard every second
                current_time = time.time()
                if current_time - last_update >= 1.0:
                    if self.dashboard_manager and self.dashboard_manager.is_running():
                        self._update_console_dashboard(self.data_history, candle if candle else {})
                    last_update = current_time
                
                time.sleep(0.1)  # Small sleep to prevent CPU overuse
                
                if self.dashboard_manager and not self.dashboard_manager.is_running():
                    self.logger.info("Dashboard closed, stopping paper bot.")
                    self.running = False
                    
        except KeyboardInterrupt:
            self.logger.info("Paper trading bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            self.logger.error(f"Unhandled error in paper trading bot run loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.logger.info("Stopping paper trading bot...")
        self.running = False
        self.stop_event.set()
        if self.price_fetcher:
            self.price_fetcher.stop()
        if self.dashboard_manager:
            self.dashboard_manager.stop_dashboard()

    def execute_buy(self, price: float, data: pd.DataFrame):
        if not self.risk_manager.check_trade_limits():
            self.logger.warning("Trade limits reached. Skipping buy signal.")
            return
        
        volatility = 0.02  # Default
        if not data.empty and len(data) > 14:
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                recent_data = data.tail(14)
                if recent_data['close'].mean() != 0:
                    volatility = (recent_data['high'].mean() - recent_data['low'].mean()) / recent_data['close'].mean()

        position_size = self.risk_manager.calculate_position_size(self.portfolio_value, price, volatility)
        entry_price = price * 1.001
        self.position = position_size
        self.entry_price = entry_price
        self.portfolio_value -= position_size * entry_price
        
        stop_loss = self.entry_price * (1 - TRADING_CONFIG.get('STOP_LOSS_PCT', 0.01))
        take_profit = self.entry_price * (1 + TRADING_CONFIG.get('TAKE_PROFIT_PCT', 0.03))
        if len(data) >= 20:
            try:
                sl = self.strategy.calculate_stop_loss(data, entry_price) if hasattr(self.strategy, 'calculate_stop_loss') else None
                tp = self.strategy.calculate_take_profit(data, entry_price) if hasattr(self.strategy, 'calculate_take_profit') else None
                if sl is not None: stop_loss = sl
                if tp is not None: take_profit = tp
            except Exception as e:
                self.logger.error(f"Error calculating SL/TP: {e}", exc_info=True)

        self.logger.info(f"BUY: {position_size} {self.symbol} at {entry_price} (Portfolio: ${self.portfolio_value:.2f})")
        self.performance_monitor.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': 'buy',
            'price': entry_price,
            'quantity': position_size,
            'pnl': 0.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })

    def execute_sell(self, price: float):
        exit_price = price * 0.999
        pnl = (exit_price - self.entry_price) * self.position
        self.portfolio_value += self.position * exit_price
        self.logger.info(f"SELL: {self.position} {self.symbol} at {exit_price} (PnL: ${pnl:.2f} | Portfolio: ${self.portfolio_value:.2f})")
        self.performance_monitor.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': 'sell',
            'price': exit_price,
            'quantity': self.position,
            'pnl': pnl
        })
        self.risk_manager.update_trade_stats(pnl)
        self.position = 0.0
        self.entry_price = 0.0

    def check_exit_conditions(self, current_price: float):
        if not self.performance_monitor.trades:
            return
        last_trade = self.performance_monitor.trades[-1]
        if 'stop_loss' in last_trade and last_trade['stop_loss'] is not None and current_price <= last_trade['stop_loss']:
            self.logger.info(f"Stop loss triggered at {current_price} (SL: {last_trade['stop_loss']})")
            self.execute_sell(current_price)
        elif 'take_profit' in last_trade and last_trade['take_profit'] is not None and current_price >= last_trade['take_profit']:
            self.logger.info(f"Take profit triggered at {current_price} (TP: {last_trade['take_profit']})")
            self.execute_sell(current_price)

    def _update_console_dashboard(self, history_data: pd.DataFrame, latest_candle: Dict[str, Any]):
        """Update the console dashboard with current metrics."""
        try:
            current_price = latest_candle['close']
            unrealized_pnl = (current_price - self.entry_price) * self.position if self.position != 0 else 0.0
            unrealized_pnl_pct = (unrealized_pnl / (self.entry_price * self.position)) * 100 if self.position != 0 else 0.0
            
            # Get performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()
            
            # Update dashboard metrics
            metrics = {
                'portfolio_value': self.portfolio_value,
                'position': self.position,
                'entry_price': self.entry_price,
                'current_price': current_price,
                'unrealized_pnl': round(unrealized_pnl, 2),
                'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                'total_trades': len(self.performance_monitor.trades),
                'winning_trades': sum(1 for t in self.performance_monitor.trades if t.get('pnl', 0) > 0),
                'losing_trades': sum(1 for t in self.performance_monitor.trades if t.get('pnl', 0) < 0),
                'win_rate': performance_metrics['win_rate'] * 100,
                'profit_factor': performance_metrics['profit_factor'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': performance_metrics['max_drawdown'],
                'market_regime': 'Bullish' if current_price > self.entry_price else 'Bearish' if current_price < self.entry_price else 'Neutral',
                'regime_confidence': 0.75,  # Placeholder for actual confidence calculation
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'uptime': int(time.time() - self.start_time),
                'api_calls': self.price_fetcher.api_calls
            }
            self.dashboard_manager.update_metrics(metrics)
            
            # Update strategy signals
            strategy_signals = {
                self.strategy.__class__.__name__: {
                    'buy': '✓' if self.position > 0 else '✗',
                    'sell': '✓' if self.position == 0 else '✗'
                }
            }
            self.dashboard_manager.update_strategy_signals(strategy_signals)
            
            # Set the model name
            model_name = "None"
            if hasattr(self.strategy, 'model') and self.strategy.model is not None:
                model_name = self.strategy.model.__class__.__name__
            self.dashboard_manager.set_model_name(model_name)
            
            # Update open positions
            if self.position > 0:
                open_positions = [{
                    'symbol': self.symbol,
                    'size': self.position,
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'leverage': self.leverage,
                    'pnl': unrealized_pnl,
                    'pnl_pct': unrealized_pnl_pct
                }]
                self.dashboard_manager.update_open_positions(open_positions)
            
            # Add current state as a trade
            self.dashboard_manager.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'hold' if self.position == 0 else 'buy',
                'price': current_price,
                'quantity': self.position,
                'pnl': 0.0
            })
            
            # Update market data
            if not history_data.empty:
                chart_data = history_data.tail(60)
                prices = [{'time': row['date'].strftime('%H:%M'), 'price': row['close']} for _, row in chart_data.iterrows()]
                market_data = {'prices': prices}
                self.dashboard_manager.update_market_data(market_data)
                
        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}", exc_info=True)
