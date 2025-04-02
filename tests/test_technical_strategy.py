"""
Unit tests for the TechnicalStrategy class.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock

from trading.strategies.technical_strategy import TechnicalStrategy

class TestTechnicalStrategy(unittest.TestCase):
    """
    Test cases for the TechnicalStrategy class.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Set up strategy
        self.strategy = TechnicalStrategy(
            logger=self.logger,
            rsi_overbought=70,
            rsi_oversold=30,
            dx_threshold=25,
            macd_threshold=0,
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
            position_size_pct=0.5
        )
        
        # Set up test data
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5, freq='H'),
            'open': [35000.0, 35500.0, 36000.0, 36500.0, 37000.0],
            'high': [36000.0, 36500.0, 37000.0, 37500.0, 38000.0],
            'low': [34000.0, 34500.0, 35000.0, 35500.0, 36000.0],
            'close': [35500.0, 36000.0, 36500.0, 37000.0, 37500.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0],
            'rsi': [40.0, 45.0, 50.0, 55.0, 60.0],
            'macd': [-5.0, -2.0, 1.0, 3.0, 5.0],
            'dx': [20.0, 22.0, 26.0, 28.0, 30.0],
            'obv': [1000.0, 1200.0, 1400.0, 1600.0, 1800.0],
            'atr': [100.0, 110.0, 120.0, 130.0, 140.0]
        })
    
    def test_init(self):
        """
        Test the initialization of the TechnicalStrategy.
        """
        # Check attributes
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.dx_threshold, 25)
        self.assertEqual(self.strategy.macd_threshold, 0)
        self.assertEqual(self.strategy.stop_loss_pct, 0.01)
        self.assertEqual(self.strategy.take_profit_pct, 0.03)
        self.assertEqual(self.strategy.position_size_pct, 0.5)
    
    def test_generate_signals_buy(self):
        """
        Test the generate_signals method for a buy signal.
        """
        # Set up data for a buy signal
        # DX > threshold, MACD > threshold, RSI < overbought, OBV increasing
        data = self.test_data.copy()
        data.loc[4, 'dx'] = 30.0       # > threshold (25)
        data.loc[4, 'macd'] = 5.0      # > threshold (0)
        data.loc[4, 'rsi'] = 60.0      # < overbought (70)
        data.loc[4, 'obv'] = 1800.0    # > previous (1600.0)
        
        # Call method
        buy_signal, sell_signal = self.strategy.generate_signals(data)
        
        # Check result
        self.assertTrue(buy_signal)
        self.assertFalse(sell_signal)
    
    def test_generate_signals_sell(self):
        """
        Test the generate_signals method for a sell signal.
        """
        # Set up data for a sell signal
        # RSI > overbought
        data = self.test_data.copy()
        data.loc[4, 'rsi'] = 75.0      # > overbought (70)
        
        # Call method
        buy_signal, sell_signal = self.strategy.generate_signals(data)
        
        # Check result
        self.assertFalse(buy_signal)
        self.assertTrue(sell_signal)
        
        # Set up data for another sell signal
        # DX < threshold and MACD < threshold
        data = self.test_data.copy()
        data.loc[4, 'dx'] = 20.0       # < threshold (25)
        data.loc[4, 'macd'] = -5.0     # < threshold (0)
        
        # Call method
        buy_signal, sell_signal = self.strategy.generate_signals(data)
        
        # Check result
        self.assertFalse(buy_signal)
        self.assertTrue(sell_signal)
    
    def test_calculate_position_size(self):
        """
        Test the calculate_position_size method.
        """
        # Set up parameters
        current_price = 36000.0
        available_capital = 10000.0
        
        # Call method
        position_size = self.strategy.calculate_position_size(self.test_data, current_price, available_capital)
        
        # Check result
        expected_size = (available_capital * self.strategy.position_size_pct) / current_price
        self.assertEqual(position_size, expected_size)
        
        # Test minimum quantity
        available_capital = 10.0  # Very small capital
        position_size = self.strategy.calculate_position_size(self.test_data, current_price, available_capital)
        self.assertEqual(position_size, 0.001)  # Minimum quantity
    
    def test_calculate_stop_loss(self):
        """
        Test the calculate_stop_loss method.
        """
        # Set up parameters
        entry_price = 36000.0
        
        # Call method
        stop_loss = self.strategy.calculate_stop_loss(self.test_data, entry_price)
        
        # Check result
        # Using ATR from the data
        atr = self.test_data.iloc[-1]['atr']
        expected_stop_loss = entry_price - (2 * atr)
        self.assertEqual(stop_loss, expected_stop_loss)
        
        # Test without ATR
        data_without_atr = self.test_data.drop(columns=['atr'])
        stop_loss = self.strategy.calculate_stop_loss(data_without_atr, entry_price)
        expected_stop_loss = entry_price * (1 - self.strategy.stop_loss_pct)
        self.assertEqual(stop_loss, expected_stop_loss)
    
    def test_calculate_take_profit(self):
        """
        Test the calculate_take_profit method.
        """
        # Set up parameters
        entry_price = 36000.0
        
        # Call method
        take_profit = self.strategy.calculate_take_profit(self.test_data, entry_price)
        
        # Check result
        # Using ATR from the data
        atr = self.test_data.iloc[-1]['atr']
        expected_take_profit = entry_price + (3 * atr)
        self.assertEqual(take_profit, expected_take_profit)
        
        # Test without ATR
        data_without_atr = self.test_data.drop(columns=['atr'])
        take_profit = self.strategy.calculate_take_profit(data_without_atr, entry_price)
        expected_take_profit = entry_price * (1 + self.strategy.take_profit_pct)
        self.assertEqual(take_profit, expected_take_profit)

if __name__ == '__main__':
    unittest.main()
