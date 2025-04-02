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
import threading # For stopping mechanism

# Removed BinanceProcessor import, added PriceFetcher
# from core.data.processors.binance_processor import BinanceProcessor
from utils.price_fetcher import PriceFetcher
from trading.strategies.base_strategy import BaseStrategy # Use base class for type hint
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
        self.stop_event = threading.Event() # For graceful shutdown

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

        # Data history (maintain state for strategy)
        self.data_history = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        self.history_lock = threading.Lock() # Protect access to data_history

        # Price Fetcher for real-time data (using 1m for dashboard responsiveness)
        self.price_fetcher = PriceFetcher(self.logger, symbol=self.symbol, timeframe='1m') # Use 1m for dashboard
        self.price_fetcher.start()

        # Strategy
        if not isinstance(strategy, BaseStrategy):
             self.logger.error(f"Invalid strategy object provided: {type(strategy)}. Must inherit from BaseStrategy.")
             raise TypeError("Strategy must inherit from BaseStrategy")
        self.strategy = strategy

        # Risk manager
        self.risk_manager = RiskManager(self.logger)
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Dashboard - Start it within the run method to ensure curses setup is correct
        self.dashboard_manager = ConsoleDashboardManager(self.logger)

        self.logger.info(f"Paper trading bot initialized for {symbol} on {timeframe} timeframe with {leverage}x leverage using {self.strategy.__class__.__name__}")

    def _process_new_candle(self, candle: Dict[str, Any]):
        """Processes a new candle received from the PriceFetcher."""
        if not candle or candle.get('close') is None or candle.get('close') <= 0:
            self.logger.debug("Received invalid or empty candle, skipping.")
            return

        current_price = candle['close']
        # Corrected key from 'time' to 'T' (Kline close time from Binance stream)
        # Add error handling in case 'T' is also missing or invalid
        try:
            candle_time = pd.to_datetime(candle['T'], unit='ms')
        except KeyError:
            self.logger.error("Candle data missing 'T' (timestamp) key. Using current time.")
            candle_time = pd.Timestamp.now() # Fallback to current time
        except Exception as time_err:
            self.logger.error(f"Error converting candle timestamp '{candle.get('T')}': {time_err}. Using current time.")
            candle_time = pd.Timestamp.now() # Fallback

        # --- Update Data History ---
        new_data = pd.DataFrame([{
            'date': candle_time,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        
        with self.history_lock:
            # Append new candle and keep a reasonable history size (e.g., 200 candles)
            self.data_history = pd.concat([self.data_history, new_data], ignore_index=True)
            self.data_history = self.data_history.iloc[-200:] # Keep last 200 candles
            
            # Make a copy for strategy processing to avoid race conditions if needed
            strategy_data = self.data_history.copy()

        # --- Strategy Execution ---
        if not strategy_data.empty:
            # Add indicators (assuming strategy or a helper handles this)
            # TODO: Need a mechanism to add indicators required by the specific strategy
            # For now, assume strategy handles data with OHLCV
            # Example: strategy_data = self.strategy.preprocess_data(strategy_data)

            self.logger.debug(f"Generating signals with {len(strategy_data)} candles...")
            try:
                # Ensure enough data for the strategy (e.g., 50 candles)
                if len(strategy_data) >= 50:
                    buy_signal, sell_signal = self.strategy.generate_signals(strategy_data)

                    # Execute trades based on signals
                    if buy_signal and self.position == 0:
                        self.execute_buy(current_price, strategy_data) # Pass historical data
                    elif sell_signal and self.position > 0:
                        self.execute_sell(current_price)
                else:
                     self.logger.debug(f"Not enough data ({len(strategy_data)}/50) for signal generation.")

            except Exception as e:
                self.logger.error(f"Error during signal generation or trade execution: {e}", exc_info=True)

        # --- Check Exit Conditions ---
        if self.position > 0:
            self.check_exit_conditions(current_price)

        # --- Update Dashboard ---
        # Pass historical data and the latest candle
        self._update_console_dashboard(strategy_data, candle)


    def run(self):
        """
        Run the paper trading bot, driven by real-time data from PriceFetcher.
        """
        self.logger.info("Starting paper trading bot run loop...")
        self.running = True
        
        # Start the dashboard within the run loop context
        dashboard_thread = threading.Thread(target=self.dashboard_manager.start_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(1) # Give dashboard time to initialize

        try:
            while self.running and not self.stop_event.is_set():
                # Get the latest candle from the fetcher
                candle = self.price_fetcher.get_current_candle()
                
                if candle:
                    self._process_new_candle(candle)
                else:
                    # No new candle yet, wait briefly
                    time.sleep(0.1) # Short sleep to prevent busy-waiting

                # Check if dashboard is still running
                if not self.dashboard_manager.is_running():
                    self.logger.info("Dashboard closed, stopping paper bot.")
                    self.running = False

        except KeyboardInterrupt:
            self.logger.info("Paper trading bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            self.logger.error(f"Unhandled error in paper trading bot run loop: {e}", exc_info=True)
        finally:
            self.stop() # Ensure cleanup happens

    def stop(self):
        """Stops the bot and cleans up resources."""
        if not self.running:
            return
        self.logger.info("Stopping paper trading bot...")
        self.running = False
        self.stop_event.set() # Signal threads to stop
        
        if self.price_fetcher:
            self.price_fetcher.stop()
            self.logger.info("Price fetcher stopped.")
            
        if self.dashboard_manager:
            self.dashboard_manager.stop_dashboard()
            self.logger.info("Dashboard manager stopped.")
            
        # Wait briefly for threads to exit if needed (optional)
        # dashboard_thread.join(timeout=2) 
        
        self.logger.info("Paper trading bot fully stopped.")
    
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
        # Ensure data has enough rows and required columns
        if not data.empty and len(data) > 1:
            if 'atr' in data.columns and not pd.isna(data['atr'].iloc[-1]):
                 volatility = data['atr'].iloc[-1]
            elif all(col in data.columns for col in ['high', 'low', 'close']):
                 # Simple volatility estimate based on recent high-low range mean
                 recent_data = data.tail(14) # Use recent period (e.g., 14)
                 if not recent_data.empty and recent_data['close'].mean() != 0:
                     volatility = (recent_data['high'].mean() - recent_data['low'].mean()) / recent_data['close'].mean()
                 else:
                     volatility = 0.02 # Default fallback volatility (e.g., 2%)
            else:
                 volatility = 0.02 # Default fallback volatility

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
        
        # Calculate stop loss and take profit using the strategy
        # Ensure data has enough history for the strategy's calculation
        stop_loss = self.entry_price * (1 - TRADING_CONFIG.get('STOP_LOSS_PCT', 0.01)) # Default SL
        take_profit = self.entry_price * (1 + TRADING_CONFIG.get('TAKE_PROFIT_PCT', 0.03)) # Default TP
        try:
            if len(data) >= 20: # Example: Ensure enough data for strategy calculation
                 sl = self.strategy.calculate_stop_loss(data, entry_price)
                 tp = self.strategy.calculate_take_profit(data, entry_price)
                 if sl is not None: stop_loss = sl
                 if tp is not None: take_profit = tp
            else:
                 self.logger.warning("Not enough data for strategy-based SL/TP, using defaults.")
        except Exception as e:
            self.logger.error(f"Error calculating strategy SL/TP: {e}. Using defaults.", exc_info=True)


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
        
        # Check stop loss (use the SL calculated during buy)
        if 'stop_loss' in last_trade and last_trade['stop_loss'] is not None and current_price <= last_trade['stop_loss']:
            self.logger.info(f"Stop loss triggered at {current_price} (SL: {last_trade['stop_loss']})")
            self.execute_sell(current_price) # Sell at current market price

        # Check take profit (use the TP calculated during buy)
        elif 'take_profit' in last_trade and last_trade['take_profit'] is not None and current_price >= last_trade['take_profit']:
            self.logger.info(f"Take profit triggered at {current_price} (TP: {last_trade['take_profit']})")
            self.execute_sell(current_price) # Sell at current market price

    # --- Mock Data Generation (Keep for potential testing, but not used in main run loop) ---
    def _generate_mock_data(self) -> pd.DataFrame:
        """
        Generate mock OHLCV data for testing purposes.
        
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
        Generate mock OHLCV data with mock indicators for testing purposes.

        Returns:
            pd.DataFrame: The generated mock data with indicators.
        """
        self.logger.debug("Generating mock data with indicators for testing") # Changed to debug level

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
        self.logger.debug(f"Generated mock data with {len(mock_data)} candles and indicators")
        return df

    # --- Dashboard Update ---
    def _update_console_dashboard(self, history_data: pd.DataFrame, latest_candle: Dict[str, Any]) -> None:
        """
        Update the console dashboard with the latest data.

        Args:
            history_data (pd.DataFrame): Historical market data for context/charts.
            latest_candle (Dict[str, Any]): The most recent candle data from PriceFetcher.
        """
        if not self.dashboard_manager or not self.dashboard_manager.is_running():
             return # Don't update if dashboard isn't running

        try:
            current_price = latest_candle['close']

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
            
            # Update metrics for dashboard display
            metrics = {
                'portfolio_value': round(total_value, 2),
                'position_size': round(self.position, 8),
                'entry_price': round(self.entry_price, 2) if self.position > 0 else 0.0,
                'current_price': round(current_price, 2),
                'unrealized_pnl': round((current_price - self.entry_price) * self.position if self.position > 0 else 0.0, 2),
                'unrealized_pnl_pct': round(((current_price / self.entry_price) - 1) * 100 if self.position > 0 and self.entry_price != 0 else 0.0, 2)
            }

            # Add performance metrics from monitor
            performance_metrics = self.performance_monitor.get_metrics()
            metrics.update(performance_metrics) # Overwrites if keys clash, which is fine

            self.dashboard_manager.update_metrics(metrics)

            # Update strategy signals (get latest signals if possible)
            # Note: Signals might be slightly delayed if calculated only on new candles
            # We can potentially store the last signals generated in _process_new_candle
            # For now, just display placeholder or last known signals
            # Re-generating signals here might be too slow for dashboard update
            strategy_signals = { self.strategy.__class__.__name__: {'buy': '?', 'sell': '?'} } # Placeholder
            # TODO: Store last signals in self attributes if needed for dashboard
            self.dashboard_manager.update_strategy_signals(strategy_signals)

            # Update candle display
            self.dashboard_manager.update_candle(latest_candle)

            # Update market data chart (using historical data)
            if not history_data.empty:
                 # Format recent history for dashboard chart (e.g., last 60 points)
                 chart_data = history_data.tail(60)
                 prices = []
                 for _, row in chart_data.iterrows():
                     # Ensure 'date' is datetime object before formatting
                     time_str = row['date'].strftime('%H:%M') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                     prices.append({
                         'time': time_str,
                         'price': row['close']
                     })
                 market_data = {'prices': prices}
                 self.dashboard_manager.update_market_data(market_data)

        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}", exc_info=True)
