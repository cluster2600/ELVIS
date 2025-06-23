"""
Integration tests for the ELVIS project.
This module provides tests for the entire trading system.
"""

import unittest
import logging
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from core.data.processors.binance_processor import BinanceProcessor
from trading.strategies.technical_strategy import TechnicalStrategy
from trading.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading.strategies.trend_following_strategy import TrendFollowingStrategy
from trading.execution.binance_executor import BinanceExecutor
from trading.risk_management import RiskManager
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
        self.start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
        self.end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
    
    def test_data_processing_to_strategy(self):
        """
        Test data processing to strategy integration.
        """
        # Set up processor
        processor = BinanceProcessor(
            data_source='binance',
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
        self.assertIsInstance(tech_buy, (bool, np.bool_))
        self.assertIsInstance(tech_sell, (bool, np.bool_))
        self.assertIsInstance(mr_buy, (bool, np.bool_))
        self.assertIsInstance(mr_sell, (bool, np.bool_))
        self.assertIsInstance(tf_buy, (bool, np.bool_))
        self.assertIsInstance(tf_sell, (bool, np.bool_))
    
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
    
    
    def test_performance_monitoring(self):
        """
        Test performance monitoring integration.
        """
        # Set up performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Add trades
        for i in range(10):
            pnl = 100.0 if i % 3 == 0 else -50.0
            performance_monitor.add_return(pnl)
        
        # Calculate metrics
        sharpe = performance_monitor.calculate_rolling_sharpe()
        drawdown = performance_monitor.calculate_rolling_drawdown()
        
        # Check metrics
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(drawdown, float)
        
    
    def test_end_to_end_workflow(self):
        """
        Test end-to-end workflow integration.
        """
        # Set up components
        processor = BinanceProcessor(
            data_source='binance',
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.timeframe,
            logger=self.logger
        )
        
        strategy = TechnicalStrategy(logger=self.logger)
        risk_manager = RiskManager(
            executor=MagicMock(spec=BinanceExecutor),
            logger=self.logger
        )
        performance_monitor = PerformanceMonitor()
        
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
        if buy_signal:
            # Simulate trade execution
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': 'buy',
                'price': data_with_indicators.iloc[-1]['close'],
                'quantity': 0.1,
                'pnl': 100.0  # Simulated PnL
            }
            
            # Update performance monitor
            performance_monitor.add_return(trade['pnl'])
            
            # Update risk manager
            risk_manager.add_position(self.symbol, trade)
        
        # Calculate metrics
        sharpe = performance_monitor.calculate_rolling_sharpe()
        drawdown = performance_monitor.calculate_rolling_drawdown()
        
        # Check metrics
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(drawdown, float)

if __name__ == '__main__':
    unittest.main()
