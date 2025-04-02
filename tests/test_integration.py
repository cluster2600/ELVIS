"""
Integration tests for the ELVIS project.
This module provides tests for the entire trading system.
"""

import unittest
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.data.processors.binance_processor import BinanceProcessor
from trading.strategies.technical_strategy import TechnicalStrategy
from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading.strategies.trend_following_strategy import TrendFollowingStrategy
from trading.execution.binance_executor import BinanceExecutor
from trading.risk.risk_manager import RiskManager
from core.models.random_forest_model import RandomForestModel
from core.models.neural_network_model import NeuralNetworkModel
from core.models.ensemble_model import EnsembleModel
from core.metrics.performance_monitor import PerformanceMonitor

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the ELVIS project.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Set up test data
        self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
    
    def test_data_processing_to_strategy(self):
        """
        Test data processing to strategy integration.
        """
        # Set up processor
        processor = BinanceProcessor(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.timeframe,
            logger=self.logger
        )
        
        # Download data
        data = processor.download_data([self.symbol])
        
        # Check data
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
        # Clean data
        processor.data = data
        cleaned_data = processor.clean_data()
        
        # Check cleaned data
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertFalse(cleaned_data.empty)
        
        # Add technical indicators
        data_with_indicators = processor.add_technical_indicator([
            'rsi', 'macd', 'bbands', 'sma', 'adx', 'obv', 'atr'
        ])
        
        # Check data with indicators
        self.assertIsInstance(data_with_indicators, pd.DataFrame)
        self.assertFalse(data_with_indicators.empty)
        self.assertTrue('rsi' in data_with_indicators.columns)
        self.assertTrue('macd' in data_with_indicators.columns)
        self.assertTrue('upperband' in data_with_indicators.columns)
        self.assertTrue('middleband' in data_with_indicators.columns)
        self.assertTrue('lowerband' in data_with_indicators.columns)
        self.assertTrue('sma_20' in data_with_indicators.columns)
        self.assertTrue('adx' in data_with_indicators.columns)
        self.assertTrue('obv' in data_with_indicators.columns)
        self.assertTrue('atr' in data_with_indicators.columns)
        
        # Set up strategies
        technical_strategy = TechnicalStrategy(logger=self.logger)
        mean_reversion_strategy = MeanReversionStrategy(logger=self.logger)
        trend_following_strategy = TrendFollowingStrategy(logger=self.logger)
        
        # Generate signals
        tech_buy, tech_sell = technical_strategy.generate_signals(data_with_indicators)
        mr_buy, mr_sell = mean_reversion_strategy.generate_signals(data_with_indicators)
        tf_buy, tf_sell = trend_following_strategy.generate_signals(data_with_indicators)
        
        # Check signals
        self.assertIsInstance(tech_buy, bool)
        self.assertIsInstance(tech_sell, bool)
        self.assertIsInstance(mr_buy, bool)
        self.assertIsInstance(mr_sell, bool)
        self.assertIsInstance(tf_buy, bool)
        self.assertIsInstance(tf_sell, bool)
    
    def test_model_training_and_prediction(self):
        """
        Test model training and prediction integration.
        """
        # Create test data
        X_train = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        X_test = pd.DataFrame({
            'feature1': np.random.random(20),
            'feature2': np.random.random(20),
            'feature3': np.random.random(20)
        })
        
        # Set up models
        rf_model = RandomForestModel(logger=self.logger)
        nn_model = NeuralNetworkModel(
            logger=self.logger,
            input_shape=(10, 3),  # Smaller for testing
            lstm_units=[32, 16],
            dense_units=[8],
            epochs=1  # Just for testing
        )
        
        # Train Random Forest model
        rf_model.train(X_train, y_train)
        
        # Make predictions with Random Forest model
        rf_predictions = rf_model.predict(X_test)
        
        # Check predictions
        self.assertIsInstance(rf_predictions, np.ndarray)
        self.assertEqual(len(rf_predictions), len(X_test))
        
        # Set up ensemble model
        ensemble_model = EnsembleModel(
            logger=self.logger,
            models=[
                ('random_forest', rf_model)
            ]
        )
        
        # Make predictions with ensemble model
        ensemble_predictions = ensemble_model.predict(X_test)
        
        # Check predictions
        self.assertIsInstance(ensemble_predictions, np.ndarray)
        self.assertEqual(len(ensemble_predictions), len(X_test))
    
    def test_risk_management(self):
        """
        Test risk management integration.
        """
        # Set up risk manager
        risk_manager = RiskManager(logger=self.logger)
        
        # Check capital
        sufficient_capital = risk_manager.check_capital(10000.0)
        insufficient_capital = risk_manager.check_capital(100.0)
        
        # Check results
        self.assertTrue(sufficient_capital)
        self.assertFalse(insufficient_capital)
        
        # Check trade limits
        trade_limits = risk_manager.check_trade_limits()
        
        # Check result
        self.assertTrue(trade_limits)
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(10000.0, 36000.0, 1000.0)
        
        # Check result
        self.assertGreater(position_size, 0.0)
        
        # Calculate leverage
        leverage = risk_manager.calculate_leverage(1000.0, 0.8)
        
        # Check result
        self.assertGreaterEqual(leverage, risk_manager.min_leverage)
        self.assertLessEqual(leverage, risk_manager.max_leverage)
        
        # Calculate stop loss
        stop_loss = risk_manager.calculate_stop_loss(36000.0, 1000.0)
        
        # Check result
        self.assertLess(stop_loss, 36000.0)
        
        # Calculate take profit
        take_profit = risk_manager.calculate_take_profit(36000.0, 1000.0)
        
        # Check result
        self.assertGreater(take_profit, 36000.0)
    
    def test_performance_monitoring(self):
        """
        Test performance monitoring integration.
        """
        # Set up performance monitor
        performance_monitor = PerformanceMonitor(logger=self.logger)
        
        # Add trades
        for i in range(10):
            trade = {
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'symbol': self.symbol,
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 36000.0 + i * 100.0,
                'quantity': 0.1,
                'pnl': 100.0 if i % 3 == 0 else -50.0
            }
            performance_monitor.add_trade(trade)
        
        # Calculate metrics
        metrics = performance_monitor.calculate_metrics()
        
        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_trades', metrics)
        self.assertIn('winning_trades', metrics)
        self.assertIn('losing_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_pnl', metrics)
        
        # Generate plots
        equity_curve_path = performance_monitor.plot_equity_curve()
        daily_returns_path = performance_monitor.plot_daily_returns()
        win_loss_path = performance_monitor.plot_win_loss_distribution()
        
        # Generate report
        report_path = performance_monitor.generate_report()
        
        # Check report
        self.assertTrue(os.path.exists(report_path))
    
    def test_end_to_end_workflow(self):
        """
        Test end-to-end workflow integration.
        """
        # Set up components
        processor = BinanceProcessor(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.timeframe,
            logger=self.logger
        )
        
        strategy = TechnicalStrategy(logger=self.logger)
        risk_manager = RiskManager(logger=self.logger)
        performance_monitor = PerformanceMonitor(logger=self.logger)
        
        # Download and process data
        data = processor.download_data([self.symbol])
        processor.data = data
        cleaned_data = processor.clean_data()
        data_with_indicators = processor.add_technical_indicator([
            'rsi', 'macd', 'bbands', 'sma', 'adx', 'obv', 'atr'
        ])
        
        # Generate signals
        buy_signal, sell_signal = strategy.generate_signals(data_with_indicators)
        
        # Simulate trading decision
        if buy_signal and risk_manager.check_trade_limits() and risk_manager.check_capital(10000.0):
            # Calculate position size
            current_price = data_with_indicators.iloc[-1]['close']
            position_size = risk_manager.calculate_position_size(10000.0, current_price, data_with_indicators.iloc[-1]['atr'])
            
            # Calculate stop loss and take profit
            stop_loss = risk_manager.calculate_stop_loss(current_price, data_with_indicators.iloc[-1]['atr'])
            take_profit = risk_manager.calculate_take_profit(current_price, data_with_indicators.iloc[-1]['atr'])
            
            # Simulate trade execution
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'buy',
                'price': current_price,
                'quantity': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pnl': 100.0  # Simulated PnL
            }
            
            # Update performance monitor
            performance_monitor.add_trade(trade)
            
            # Update risk manager
            risk_manager.update_trade_stats(trade['pnl'])
        
        # Calculate metrics
        metrics = performance_monitor.calculate_metrics()
        
        # Check metrics
        self.assertIsInstance(metrics, dict)

if __name__ == '__main__':
    unittest.main()
