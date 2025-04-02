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

from core.data.processors.binance_processor import BinanceProcessor
from trading.strategies.technical_strategy import TechnicalStrategy
from trading.risk.risk_manager import RiskManager
from core.metrics.performance_monitor import PerformanceMonitor
from utils.console_dashboard import ConsoleDashboardManager
from config import TRADING_CONFIG, TECH_INDICATORS

class PaperBot:
    """
    Paper trading bot for testing strategies without real money.
    """
    
    def __init__(self, symbol: str, timeframe: str, leverage: int, strategy=None, logger: Optional[logging.Logger] = None):
        """
        Initialize the paper trading bot.
        
        Args:
            symbol (str): The trading symbol.
            timeframe (str): The trading timeframe.
            leverage (int): The leverage to use.
            strategy (BaseStrategy, optional): The trading strategy to use.
            logger (logging.Logger, optional): The logger to use.
        """
        # Set up logger
        self.logger = logger or logging.getLogger("ELVIS.PaperBot")
        
        # Trading parameters
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = TRADING_CONFIG['MIN_CAPITAL_USD']
        self.running = False
        
        # Initialize components
        self.logger.info("Initializing paper trading bot...")
        
        # Data processor
        self.processor = BinanceProcessor(
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
            end_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time_interval=timeframe,
            logger=self.logger
        )
        
        # Strategy
        if strategy is None:
            from trading.strategies import TechnicalStrategy
            self.strategy = TechnicalStrategy(self.logger)
        else:
            self.strategy = strategy
        
        # Risk manager
        self.risk_manager = RiskManager(self.logger)
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Dashboard
        self.dashboard_manager = ConsoleDashboardManager(self.logger)
        self.dashboard_manager.start_dashboard()
        
        self.logger.info(f"Paper trading bot initialized for {symbol} on {timeframe} timeframe with {leverage}x leverage")
    
    def run(self):
        """
        Run the paper trading bot.
        """
        self.logger.info("Starting paper trading bot...")
        self.running = True
        
        try:
            while self.running:
                # Get latest data
                self.logger.info("Fetching latest market data...")
                try:
                    data = self.processor.download_data([self.symbol])
                    
                    if data.empty:
                        self.logger.warning("No data received. Generating mock data...")
                        data = self._generate_mock_data()
                    
                    # Add technical indicators
                    data = self.processor.add_technical_indicator(TECH_INDICATORS)
                    
                    if data.empty:
                        self.logger.warning("No data after adding indicators. Generating mock data with indicators...")
                        data = self._generate_mock_data_with_indicators()
                except Exception as e:
                    self.logger.error(f"Error fetching data: {e}")
                    self.logger.warning("Generating mock data with indicators...")
                    data = self._generate_mock_data_with_indicators()
                
                # Generate signals
                self.logger.info("Generating trading signals...")
                buy_signal, sell_signal = self.strategy.generate_signals(data.tail(50))
                
                # Check if data is empty
                if data.empty or 'close' not in data.columns:
                    self.logger.warning("No valid price data available. Skipping trading logic.")
                else:
                    # Current price
                    current_price = data['close'].iloc[-1]
                    
                    # Execute trades
                    if buy_signal and self.position == 0:
                        self.execute_buy(current_price, data.tail(20))
                    elif sell_signal and self.position > 0:
                        self.execute_sell(current_price)
                    
                # Check for stop loss or take profit
                if self.position > 0:
                    self.check_exit_conditions(current_price)
                
                # Update dashboard
                self._update_console_dashboard(data, current_price)
                
                # Sleep
                self.logger.info(f"Sleeping for {TRADING_CONFIG['SLEEP_INTERVAL']} seconds...")
                time.sleep(TRADING_CONFIG['SLEEP_INTERVAL'])
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading bot stopped by user")
        except Exception as e:
            self.logger.error(f"Paper trading bot error: {e}")
        finally:
            self.running = False
            self.dashboard_manager.stop_dashboard()
            self.logger.info("Paper trading bot stopped")
    
    def execute_buy(self, price: float, data: pd.DataFrame):
        """
        Execute a buy order.
        
        Args:
            price (float): The current price.
            data (pd.DataFrame): The market data.
        """
        # Check if we can trade
        if not self.risk_manager.check_trade_limits():
            self.logger.warning("Trade limits reached. Skipping buy signal.")
            return
        
        # Calculate volatility (using ATR if available, otherwise use a simple estimate)
        volatility = 0.0
        if 'atr' in data.columns:
            volatility = data['atr'].iloc[-1]
        else:
            # Simple volatility estimate based on high-low range
            volatility = (data['high'].mean() - data['low'].mean()) / data['close'].mean()
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.portfolio_value,
            price,
            volatility
        )
        
        # Calculate entry price (with slippage)
        entry_price = price * 1.001
        
        # Update position
        self.position = position_size
        self.entry_price = entry_price
        self.portfolio_value -= position_size * entry_price
        
        # Calculate stop loss and take profit
        stop_loss = self.strategy.calculate_stop_loss(data, entry_price)
        take_profit = self.strategy.calculate_take_profit(data, entry_price)
        
        # Log trade
        self.logger.info(f"BUY: {position_size} {self.symbol} at {entry_price} (Portfolio: ${self.portfolio_value:.2f})")
        self.logger.info(f"Stop Loss: {stop_loss} | Take Profit: {take_profit}")
        
        # Record trade
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
        """
        Execute a sell order.
        
        Args:
            price (float): The current price.
        """
        # Calculate exit price (with slippage)
        exit_price = price * 0.999
        
        # Calculate PnL
        pnl = (exit_price - self.entry_price) * self.position
        
        # Update portfolio value
        self.portfolio_value += self.position * exit_price
        
        # Log trade
        self.logger.info(f"SELL: {self.position} {self.symbol} at {exit_price} (PnL: ${pnl:.2f} | Portfolio: ${self.portfolio_value:.2f})")
        
        # Record trade
        self.performance_monitor.add_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': 'sell',
            'price': exit_price,
            'quantity': self.position,
            'pnl': pnl
        })
        
        # Update risk manager
        self.risk_manager.update_trade_stats(pnl)
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
    
    def check_exit_conditions(self, current_price: float):
        """
        Check if we should exit the position based on stop loss or take profit.
        
        Args:
            current_price (float): The current price.
        """
        # Get the last trade
        if not self.performance_monitor.trades:
            return
        
        last_trade = self.performance_monitor.trades[-1]
        
        # Check stop loss
        if 'stop_loss' in last_trade and current_price <= last_trade['stop_loss']:
            self.logger.info(f"Stop loss triggered at {current_price}")
            self.execute_sell(current_price)
        
        # Check take profit
        elif 'take_profit' in last_trade and current_price >= last_trade['take_profit']:
            self.logger.info(f"Take profit triggered at {current_price}")
            self.execute_sell(current_price)
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """
        Generate mock data for testing.
        
        Returns:
            pd.DataFrame: The generated mock data.
        """
        self.logger.info("Generating mock data for testing")
        
        # Create a date range
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Generate dates based on timeframe
        timeframe_minutes = 60  # Default to 1h
        if self.timeframe == '1m':
            timeframe_minutes = 1
        elif self.timeframe == '5m':
            timeframe_minutes = 5
        elif self.timeframe == '15m':
            timeframe_minutes = 15
        elif self.timeframe == '30m':
            timeframe_minutes = 30
        elif self.timeframe == '1h':
            timeframe_minutes = 60
        elif self.timeframe == '4h':
            timeframe_minutes = 240
        elif self.timeframe == '1d':
            timeframe_minutes = 1440
            
        # Generate dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(minutes=timeframe_minutes)
        
        # Limit to 100 dates
        if len(dates) > 100:
            dates = dates[:100]
        
        # Create mock data
        mock_data = []
        base_price = 75655.0  # Current price for BTC (as of April 2025)
        current_price = base_price
        
        for date in dates:
            # Add some randomness to the price
            price_change = current_price * np.random.normal(0, 0.01)
            
            # Calculate OHLCV values
            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = abs(np.random.normal(1000, 500))
            
            # Update current price for next candle
            current_price = close_price
            
            # Add to mock data
            mock_data.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'ticker': self.symbol
            })
        
        # Create DataFrame
        df = pd.DataFrame(mock_data)
        self.logger.info(f"Generated mock data with {len(mock_data)} candles")
        return df
    
    def _generate_mock_data_with_indicators(self) -> pd.DataFrame:
        """
        Generate mock data with technical indicators for testing.
        
        Returns:
            pd.DataFrame: The generated mock data with indicators.
        """
        self.logger.info("Generating mock data with indicators for testing")
        
        # Create a date range
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Generate dates based on timeframe
        timeframe_minutes = 60  # Default to 1h
        if self.timeframe == '1m':
            timeframe_minutes = 1
        elif self.timeframe == '5m':
            timeframe_minutes = 5
        elif self.timeframe == '15m':
            timeframe_minutes = 15
        elif self.timeframe == '30m':
            timeframe_minutes = 30
        elif self.timeframe == '1h':
            timeframe_minutes = 60
        elif self.timeframe == '4h':
            timeframe_minutes = 240
        elif self.timeframe == '1d':
            timeframe_minutes = 1440
            
        # Generate dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(minutes=timeframe_minutes)
        
        # Limit to 100 dates
        if len(dates) > 100:
            dates = dates[:100]
        
        # Create mock data
        mock_data = []
        base_price = 75655.0  # Current price for BTC (as of April 2025)
        current_price = base_price
        
        for date in dates:
            # Add some randomness to the price
            price_change = current_price * np.random.normal(0, 0.01)
            
            # Calculate OHLCV values
            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = abs(np.random.normal(1000, 500))
            
            # Update current price for next candle
            current_price = close_price
            
            # Add mock indicators
            rsi = np.random.uniform(30, 70)
            macd = np.random.normal(0, 10)
            dx = np.random.uniform(20, 30)
            obv = volume * np.random.uniform(0.8, 1.2)
            
            # Add to mock data
            mock_data.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'ticker': self.symbol,
                'rsi': rsi,
                'macd': macd,
                'dx': dx,
                'obv': obv
            })
        
        # Create DataFrame
        df = pd.DataFrame(mock_data)
        self.logger.info(f"Generated mock data with {len(mock_data)} candles and indicators")
        return df
        
    def _update_console_dashboard(self, data: pd.DataFrame, current_price: float) -> None:
        """
        Update the console dashboard with the latest data.
        
        Args:
            data (pd.DataFrame): The market data.
            current_price (float): The current price.
        """
        try:
            # Update portfolio value
            total_value = self.portfolio_value
            if self.position > 0:
                total_value += self.position * current_price
            
            self.dashboard_manager.update_portfolio_value(total_value)
            
            # Update position information
            self.dashboard_manager.update_position(
                self.position,
                self.entry_price if self.position > 0 else 0.0,
                current_price
            )
            
            # Update metrics
            metrics = {
                'portfolio_value': total_value,
                'position_size': self.position,
                'entry_price': self.entry_price if self.position > 0 else 0.0,
                'current_price': current_price,
                'unrealized_pnl': (current_price - self.entry_price) * self.position if self.position > 0 else 0.0,
                'unrealized_pnl_pct': ((current_price / self.entry_price) - 1) * 100 if self.position > 0 else 0.0
            }
            
            # Add performance metrics
            if hasattr(self.performance_monitor, 'get_metrics'):
                performance_metrics = self.performance_monitor.get_metrics()
                metrics.update(performance_metrics)
            
            self.dashboard_manager.update_metrics(metrics)
            
            # Update strategy signals
            if hasattr(self.strategy, 'generate_signals'):
                buy_signal, sell_signal = self.strategy.generate_signals(data.tail(50))
                
                strategy_signals = {
                    self.strategy.__class__.__name__: {
                        'buy': buy_signal,
                        'sell': sell_signal
                    }
                }
                
                self.dashboard_manager.update_strategy_signals(strategy_signals)
            
            # Update market data
            if not data.empty and 'close' in data.columns:
                # Get the last 20 candles
                recent_data = data.tail(20)
                
                # Format for dashboard
                prices = []
                for _, row in recent_data.iterrows():
                    prices.append({
                        'time': row['date'].strftime('%H:%M:%S') if isinstance(row['date'], datetime) else str(row['date']),
                        'price': row['close']
                    })
                
                market_data = {
                    'prices': prices
                }
                
                self.dashboard_manager.update_market_data(market_data)
            
        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}")
