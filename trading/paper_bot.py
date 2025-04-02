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
from core.metrics.performance_monitor import PerformanceMonitor
from config import TRADING_CONFIG

class PaperBot:
    def __init__(self, symbol: str, timeframe: str, leverage: int, strategy=None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ELVIS.PaperBot")
        self.stop_event = threading.Event()
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.position = 0.0
        self.entry_price = 0.0
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
        self.strategy = strategy
        
        self.risk_manager = RiskManager(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        # Move import inside __init__ to avoid circular import
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
        self.logger.info("Starting paper trading bot run loop...")
        self.running = True
        
        dashboard_thread = threading.Thread(target=self.dashboard_manager.start_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(1)
        
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
            
            # Additional mock positions for display
            open_positions = [
                {
                    'symbol': self.symbol,
                    'size': self.position,
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'leverage': self.leverage,
                    'pnl': (current_price - self.entry_price) * self.position,
                    'pnl_pct': ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0.0
                },
                {
                    'symbol': 'ETH/USDT',
                    'size': 0.15,
                    'entry_price': 3800.0,
                    'current_price': 3850.0,
                    'leverage': self.leverage,
                    'pnl': (3850.0 - 3800.0) * 0.15,
                    'pnl_pct': ((3850.0 / 3800.0) - 1) * 100
                },
                {
                    'symbol': 'SOL/USDT',
                    'size': 2.5,
                    'entry_price': 150.0,
                    'current_price': 145.0,
                    'leverage': self.leverage,
                    'pnl': (145.0 - 150.0) * 2.5,
                    'pnl_pct': ((145.0 / 150.0) - 1) * 100
                }
            ]
            
            # Update portfolio value based on all positions
            self.portfolio_value = TRADING_CONFIG['MIN_CAPITAL_USD'] - sum(pos['size'] * pos['entry_price'] for pos in open_positions)
            
            # Update the dashboard with the mock positions
            self.dashboard_manager.update_open_positions(open_positions)
            
            # Add mock trades to the performance monitor
            self.logger.info("Adding mock trades for testing...")
            
            # Generate a series of mock trades over the past 24 hours
            for i in range(20):
                hours_ago = 24 - i
                
                # Alternate buy and sell trades
                side = 'buy' if i % 2 == 0 else 'sell'
                
                # Generate realistic price movements
                base_price = 74000.0 + (i * 100)  # Gradually increasing price
                price_noise = np.random.normal(0, 200)  # Random noise
                price = base_price + price_noise
                
                # Calculate PnL for sell trades
                pnl = 0.0
                quantity = 0.01
                
                if side == 'sell' and i > 0:
                    prev_price = 74000.0 + ((i-1) * 100) + np.random.normal(0, 200)
                    pnl = (price - prev_price) * quantity
                
                # Add the trade
                self.performance_monitor.add_trade({
                    'timestamp': (datetime.now() - timedelta(hours=hours_ago)).isoformat(),
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
            while self.running and not self.stop_event.is_set():
                candle = self.price_fetcher.get_current_candle()
                if candle:
                    self.logger.debug(f"Processing candle: {candle}")
                    self._process_new_candle(candle)
                else:
                    time.sleep(0.1)
                if not self.dashboard_manager.is_running():
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
        if not self.dashboard_manager:
            return
        try:
            # Get the current price from the price fetcher directly to ensure it's up-to-date
            current_price = self.price_fetcher.get_current_price()
            
            # If the price is still 0, use the default price from the price fetcher
            if current_price == 0:
                current_price = self.price_fetcher.default_price
            
            total_value = self.portfolio_value + (self.position * current_price if self.position > 0 else 0)
            self.dashboard_manager.update_portfolio_value(total_value)
            
            # Update the position with the current price
            if self.position > 0:
                self.dashboard_manager.update_position(self.position, self.entry_price, current_price)
            
            metrics = {
                'portfolio_value': round(total_value, 2),
                'position_size': round(self.position, 8),
                'entry_price': round(self.entry_price, 2) if self.position > 0 else 0.0,
                'current_price': round(current_price, 2),
                'unrealized_pnl': round((current_price - self.entry_price) * self.position if self.position > 0 else 0.0, 2),
                'unrealized_pnl_pct': round(((current_price / self.entry_price) - 1) * 100 if self.position > 0 and self.entry_price != 0 else 0.0, 2)
            }
            metrics.update(self.performance_monitor.get_metrics())
            self.dashboard_manager.update_metrics(metrics)
            strategy_signals = {self.strategy.__class__.__name__: {'buy': '?', 'sell': '?'}}
            self.dashboard_manager.update_strategy_signals(strategy_signals)
            
            # Set the model name in the dashboard
            model_name = "None"
            if hasattr(self.strategy, 'model') and self.strategy.model is not None:
                model_name = self.strategy.model.__class__.__name__
            self.dashboard_manager.set_model_name(model_name)
            
            # Update open positions if we have a position
            if self.position > 0:
                open_positions = [{
                    'symbol': self.symbol,
                    'size': self.position,
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'leverage': self.leverage,
                    'pnl': (current_price - self.entry_price) * self.position,
                    'pnl_pct': ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0.0
                }]
                self.dashboard_manager.update_open_positions(open_positions)
            self.dashboard_manager.add_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'hold' if self.position == 0 else 'buy',
                'price': current_price,
                'quantity': self.position,
                'pnl': 0.0
            })
            if not history_data.empty:
                chart_data = history_data.tail(60)
                prices = [{'time': row['date'].strftime('%H:%M'), 'price': row['close']} for _, row in chart_data.iterrows()]
                market_data = {'prices': prices}
                self.dashboard_manager.update_market_data(market_data)
        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}", exc_info=True)
