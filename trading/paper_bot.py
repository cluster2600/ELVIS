"""
Paper trading bot for the ELVIS project.
This module provides a paper trading implementation for testing strategies without real money.
"""

import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from core.data.processors.binance_processor import BinanceProcessor
from trading.strategies.technical_strategy import TechnicalStrategy
from trading.risk.risk_manager import RiskManager
from core.metrics.performance_monitor import PerformanceMonitor
from config import TRADING_CONFIG, TECH_INDICATORS

class PaperBot:
    """
    Paper trading bot for testing strategies without real money.
    """
    
    def __init__(self, symbol: str, timeframe: str, leverage: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the paper trading bot.
        
        Args:
            symbol (str): The trading symbol.
            timeframe (str): The trading timeframe.
            leverage (int): The leverage to use.
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
        self.strategy = TechnicalStrategy(self.logger)
        
        # Risk manager
        self.risk_manager = RiskManager(self.logger)
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor(self.logger)
        
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
                data = self.processor.download_data([self.symbol])
                
                if data.empty:
                    self.logger.warning("No data received. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Add technical indicators
                data = self.processor.add_technical_indicator(TECH_INDICATORS)
                
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
                
                # Sleep
                self.logger.info(f"Sleeping for {TRADING_CONFIG['SLEEP_INTERVAL']} seconds...")
                time.sleep(TRADING_CONFIG['SLEEP_INTERVAL'])
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading bot stopped by user")
        except Exception as e:
            self.logger.error(f"Paper trading bot error: {e}")
        finally:
            self.running = False
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
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            data,
            price,
            self.portfolio_value
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
