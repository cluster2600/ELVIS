"""
Unit tests for the RiskManager class.
"""

import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from config import TRADING_CONFIG
from trading.risk_management import RiskManager


class TestRiskManager(unittest.TestCase):
    """
    Test cases for the RiskManager class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

        # Set up risk manager
        self.risk_manager = RiskManager(executor=MagicMock(), logger=self.logger)

    def test_init(self):
        """
        Test the initialization of the RiskManager.
        """
        # Check attributes
        self.assertEqual(self.risk_manager.logger, self.logger)
        self.assertIsNotNone(self.risk_manager.executor)

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
        # Set up (reach the configured max trades per day)
        self.risk_manager.trades_today = self.risk_manager.max_trades_per_day

        # Call method
        result = self.risk_manager.check_trade_limits()

        # Check result
        self.assertFalse(result)

    def test_check_trade_limits_profit_target_reached(self):
        """
        Test the check_trade_limits method when profit target is reached.
        """
        # Set up (reach the configured daily profit target)
        self.risk_manager.daily_pnl = self.risk_manager.daily_profit_target_usd

        # Call method
        result = self.risk_manager.check_trade_limits()

        # Check result
        self.assertFalse(result)

    def test_check_trade_limits_loss_limit_reached(self):
        """
        Test the check_trade_limits method when loss limit is reached.
        """
        # Set up (reach the configured daily loss limit)
        self.risk_manager.daily_pnl = self.risk_manager.daily_loss_limit_usd

        # Call method
        result = self.risk_manager.check_trade_limits()

        # Check result
        self.assertFalse(result)

    @unittest.skip(
        "Cooldown enforcement was intentionally removed from check_trade_limits "
        "(cooldown_period = 0, 'COOLDOWN COMPLETELY REMOVED'); recent trades no "
        "longer block trading, so this assertion no longer reflects behavior."
    )
    def test_check_trade_limits_cooldown_not_elapsed(self):
        """
        Test the check_trade_limits method when cooldown period has not elapsed.
        """
        # Set up
        self.risk_manager.last_trade_time = datetime.now() - timedelta(
            seconds=1800
        )  # 30 minutes ago

        # Call method
        result = self.risk_manager.check_trade_limits()

        # Check result
        self.assertFalse(result)

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
        # Current implementation applies the dynamic stop-loss percentage
        # directly to the entry price (volatility is not used here).
        expected_stop_loss = entry_price * (1 - self.risk_manager.dynamic_stop_loss_pct)

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
        # Current implementation takes the larger of a volatility-based target
        # and the dynamic take-profit percentage applied to the entry price.
        expected_take_profit = max(
            entry_price + (3 * volatility),
            entry_price * (1 + self.risk_manager.dynamic_take_profit_pct),
        )

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
        current_time = datetime(2026, 8, 12, 0, 5)
        self.risk_manager.last_trade_time = current_time - timedelta(minutes=2)

        # Call method
        with patch("trading.risk_management.datetime") as clock:
            clock.now.return_value = current_time
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


if __name__ == "__main__":
    unittest.main()
