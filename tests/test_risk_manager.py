"""
Unit tests for the RiskManager class.
"""

import unittest
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from trading.risk.risk_manager import RiskManager
from config import TRADING_CONFIG

class TestRiskManager(unittest.TestCase):
    """
    Test cases for the RiskManager class.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Set up risk manager
        self.risk_manager = RiskManager(
            logger=self.logger,
            max_position_size_pct=0.5,
            max_leverage=TRADING_CONFIG['LEVERAGE_MAX'],
            min_leverage=TRADING_CONFIG['LEVERAGE_MIN'],
            stop_loss_pct=0.01,
            take_profit_pct=0.03,
            max_trades_per_day=5,
            daily_profit_target_usd=1000.0,
            daily_loss_limit_usd=-500.0,
            min_capital_usd=1000.0,
            cooldown_period=3600.0
        )
    
    def test_init(self):
        """
        Test the initialization of the RiskManager.
        """
        # Check attributes
        self.assertEqual(self.risk_manager.logger, self.logger)
        self.assertEqual(self.risk_manager.max_position_size_pct, 0.5)
        self.assertEqual(self.risk_manager.max_leverage, TRADING_CONFIG['LEVERAGE_MAX'])
        self.assertEqual(self.risk_manager.min_leverage, TRADING_CONFIG['LEVERAGE_MIN'])
        self.assertEqual(self.risk_manager.stop_loss_pct, 0.01)
        self.assertEqual(self.risk_manager.take_profit_pct, 0.03)
        self.assertEqual(self.risk_manager.max_trades_per_day, 5)
        self.assertEqual(self.risk_manager.daily_profit_target_usd, 1000.0)
        self.assertEqual(self.risk_manager.daily_loss_limit_usd, -500.0)
        self.assertEqual(self.risk_manager.min_capital_usd, 1000.0)
        self.assertEqual(self.risk_manager.cooldown_period, 3600.0)
        
        # Check trade tracking
        self.assertEqual(self.risk_manager.trades_today, 0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertIsNone(self.risk_manager.last_trade_time)
    
    def test_check_capital_sufficient(self):
        """
        Test the check_capital method with sufficient capital.
        """
        # Call method
        result = self.risk_manager.check_capital(2000.0)
        
        # Check result
        self.assertTrue(result)
    
    def test_check_capital_insufficient(self):
        """
        Test the check_capital method with insufficient capital.
        """
        # Call method
        result = self.risk_manager.check_capital(500.0)
        
        # Check result
        self.assertFalse(result)
    
    def test_check_trade_limits_not_reached(self):
        """
        Test the check_trade_limits method when limits are not reached.
        """
        # Call method
        result = self.risk_manager.check_trade_limits()
        
        # Check result
        self.assertTrue(result)
    
    def test_check_trade_limits_max_trades_reached(self):
        """
        Test the check_trade_limits method when max trades are reached.
        """
        # Set up
        self.risk_manager.trades_today = 5
        
        # Call method
        result = self.risk_manager.check_trade_limits()
        
        # Check result
        self.assertFalse(result)
    
    def test_check_trade_limits_profit_target_reached(self):
        """
        Test the check_trade_limits method when profit target is reached.
        """
        # Set up
        self.risk_manager.daily_pnl = 1500.0
        
        # Call method
        result = self.risk_manager.check_trade_limits()
        
        # Check result
        self.assertFalse(result)
    
    def test_check_trade_limits_loss_limit_reached(self):
        """
        Test the check_trade_limits method when loss limit is reached.
        """
        # Set up
        self.risk_manager.daily_pnl = -600.0
        
        # Call method
        result = self.risk_manager.check_trade_limits()
        
        # Check result
        self.assertFalse(result)
    
    def test_check_trade_limits_cooldown_not_elapsed(self):
        """
        Test the check_trade_limits method when cooldown period has not elapsed.
        """
        # Set up
        self.risk_manager.last_trade_time = datetime.now() - timedelta(seconds=1800)  # 30 minutes ago
        
        # Call method
        result = self.risk_manager.check_trade_limits()
        
        # Check result
        self.assertFalse(result)
    
    def test_calculate_position_size(self):
        """
        Test the calculate_position_size method.
        """
        # Set up parameters
        available_capital = 10000.0
        current_price = 36000.0
        volatility = 1000.0  # ATR
        
        # Call method
        position_size = self.risk_manager.calculate_position_size(available_capital, current_price, volatility)
        
        # Check result
        # Expected calculation:
        # volatility_factor = 1.0 / (1.0 + 1000.0 / 36000.0) = 0.9729
        # position_value = 10000.0 * 0.5 * 0.9729 = 4864.5
        # quantity = 4864.5 / 36000.0 = 0.1351
        expected_volatility_factor = 1.0 / (1.0 + 1000.0 / 36000.0)
        expected_position_value = available_capital * 0.5 * expected_volatility_factor
        expected_quantity = expected_position_value / current_price
        
        self.assertAlmostEqual(position_size, expected_quantity, places=4)
    
    def test_calculate_leverage(self):
        """
        Test the calculate_leverage method.
        """
        # Set up parameters
        volatility = 1000.0  # ATR
        signal_strength = 0.8
        
        # Call method
        leverage = self.risk_manager.calculate_leverage(volatility, signal_strength)
        
        # Check result
        # Expected calculation:
        # base_leverage = 75 + (125 - 75) * 0.8 = 115
        # volatility_factor = 1.0 / (1.0 + 1000.0) = 0.001
        # leverage = 115 * 0.001 = 0.115 -> 75 (min)
        self.assertEqual(leverage, TRADING_CONFIG['LEVERAGE_MIN'])
        
        # Test with lower volatility
        volatility = 0.1
        leverage = self.risk_manager.calculate_leverage(volatility, signal_strength)
        
        # Expected calculation:
        # base_leverage = 75 + (125 - 75) * 0.8 = 115
        # volatility_factor = 1.0 / (1.0 + 0.1) = 0.9091
        # leverage = 115 * 0.9091 = 104.55 -> 104
        expected_base_leverage = TRADING_CONFIG['LEVERAGE_MIN'] + (TRADING_CONFIG['LEVERAGE_MAX'] - TRADING_CONFIG['LEVERAGE_MIN']) * signal_strength
        expected_volatility_factor = 1.0 / (1.0 + 0.1)
        expected_leverage = int(expected_base_leverage * expected_volatility_factor)
        
        self.assertEqual(leverage, expected_leverage)
    
    def test_calculate_stop_loss(self):
        """
        Test the calculate_stop_loss method.
        """
        # Set up parameters
        entry_price = 36000.0
        volatility = 1000.0  # ATR
        
        # Call method
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, volatility)
        
        # Check result
        # Expected calculation:
        # stop_loss = 36000.0 - (2 * 1000.0) = 34000.0
        # min_stop_loss = 36000.0 * (1 - 0.01) = 35640.0
        # max(34000.0, 35640.0) = 35640.0
        expected_stop_loss = max(entry_price - (2 * volatility), entry_price * (1 - self.risk_manager.stop_loss_pct))
        
        self.assertEqual(stop_loss, expected_stop_loss)
    
    def test_calculate_take_profit(self):
        """
        Test the calculate_take_profit method.
        """
        # Set up parameters
        entry_price = 36000.0
        volatility = 1000.0  # ATR
        
        # Call method
        take_profit = self.risk_manager.calculate_take_profit(entry_price, volatility)
        
        # Check result
        # Expected calculation:
        # take_profit = 36000.0 + (3 * 1000.0) = 39000.0
        # min_take_profit = 36000.0 * (1 + 0.03) = 37080.0
        # max(39000.0, 37080.0) = 39000.0
        expected_take_profit = max(entry_price + (3 * volatility), entry_price * (1 + self.risk_manager.take_profit_pct))
        
        self.assertEqual(take_profit, expected_take_profit)
    
    def test_update_trade_stats(self):
        """
        Test the update_trade_stats method.
        """
        # Call method
        self.risk_manager.update_trade_stats(100.0)
        
        # Check result
        self.assertEqual(self.risk_manager.trades_today, 1)
        self.assertEqual(self.risk_manager.daily_pnl, 100.0)
        self.assertIsNotNone(self.risk_manager.last_trade_time)
        
        # Call again
        self.risk_manager.update_trade_stats(-50.0)
        
        # Check result
        self.assertEqual(self.risk_manager.trades_today, 2)
        self.assertEqual(self.risk_manager.daily_pnl, 50.0)
    
    def test_reset_daily_stats(self):
        """
        Test the reset_daily_stats method.
        """
        # Set up
        self.risk_manager.trades_today = 3
        self.risk_manager.daily_pnl = 500.0
        
        # Call method
        self.risk_manager.reset_daily_stats()
        
        # Check result
        self.assertEqual(self.risk_manager.trades_today, 0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
    
    def test_check_new_day_no_last_trade(self):
        """
        Test the check_new_day method with no last trade.
        """
        # Call method
        result = self.risk_manager.check_new_day()
        
        # Check result
        self.assertFalse(result)
    
    def test_check_new_day_same_day(self):
        """
        Test the check_new_day method on the same day.
        """
        # Set up
        self.risk_manager.last_trade_time = datetime.now() - timedelta(hours=2)
        
        # Call method
        result = self.risk_manager.check_new_day()
        
        # Check result
        self.assertFalse(result)
    
    def test_check_new_day_new_day(self):
        """
        Test the check_new_day method on a new day.
        """
        # Set up
        self.risk_manager.last_trade_time = datetime.now() - timedelta(days=1)
        self.risk_manager.trades_today = 3
        self.risk_manager.daily_pnl = 500.0
        
        # Call method
        result = self.risk_manager.check_new_day()
        
        # Check result
        self.assertTrue(result)
        self.assertEqual(self.risk_manager.trades_today, 0)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)

if __name__ == '__main__':
    unittest.main()
