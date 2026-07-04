"""
Unit tests for the BinanceExecutor class.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from trading.execution.binance_executor import BinanceExecutor


class TestBinanceExecutor(unittest.TestCase):
    """
    Test cases for the BinanceExecutor class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

        # Set up executor
        self.executor = BinanceExecutor(logger=self.logger)

        # Mock client
        self.executor.client = MagicMock()

        # Test symbol
        self.symbol = "BTCUSDT"

    def test_initialize(self):
        """
        Test the initialize method.

        For a live spot executor (given explicit valid API keys) initialize()
        constructs a binance ``Client``.  API_CONFIG is now an object whose
        credentials are read as attributes, so the executor is given explicit
        keys instead of patching a dict.
        """
        executor = BinanceExecutor(
            logger=self.logger, api_key="realkey", api_secret="realsecret"
        )
        with patch("trading.execution.binance_executor.Client") as mock_client:
            self.assertTrue(executor.initialize())
            mock_client.assert_called_once()

    def test_get_balance(self):
        """
        Test the get_balance method.
        """
        self.executor.client.get_account.return_value = {
            "balances": [
                {"asset": "USDT", "free": "1000.0"},
                {"asset": "BTC", "free": "0.1"},
            ]
        }
        balance = self.executor.get_balance()
        self.assertEqual(balance["USDT"], 1000.0)
        self.assertEqual(balance["BTC"], 0.1)

    def test_get_position_paper(self):
        """
        In paper / spot mode (use_futures=False) get_position returns an empty
        dict without touching the client.
        """
        self.assertEqual(self.executor.get_position("BTCUSDT"), {})

    def test_get_position_futures(self):
        """
        For a futures executor get_position returns the first entry from the
        client's get_position_risk response.
        """
        executor = BinanceExecutor(logger=self.logger, use_futures=True)
        executor.client = MagicMock()
        executor.client.get_position_risk.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.1"}
        ]
        position = executor.get_position("BTCUSDT")
        self.assertEqual(position["positionAmt"], "0.1")
        executor.client.get_position_risk.assert_called_once_with(symbol="BTCUSDT")

    def test_get_current_price(self):
        """
        Test the get_current_price method.
        """
        self.executor.client.get_symbol_ticker.return_value = {"price": "50000.0"}
        price = self.executor.get_current_price("BTCUSDT")
        self.assertEqual(price, 50000.0)

    def test_set_leverage_paper(self):
        """
        In paper / spot mode set_leverage is a no-op that does not call the
        client.
        """
        self.executor.set_leverage("BTCUSDT", 10)
        self.executor.client.change_leverage.assert_not_called()

    def test_set_leverage_futures(self):
        """
        For a futures executor set_leverage forwards to the client's
        change_leverage method.
        """
        executor = BinanceExecutor(logger=self.logger, use_futures=True)
        executor.client = MagicMock()
        executor.set_leverage("BTCUSDT", 10)
        executor.client.change_leverage.assert_called_once_with(
            symbol="BTCUSDT", leverage=10
        )

    def test_execute_buy(self):
        """
        The executor runs in paper / mock mode: execute_buy returns a mock
        FILLED order and never calls the live create_order API.
        """
        order = self.executor.execute_buy("BTCUSDT", 0.1, 50000.0)
        self.executor.client.create_order.assert_not_called()
        self.assertEqual(order["status"], "FILLED")
        self.assertEqual(order["side"], "BUY")
        self.assertTrue(order["orderId"].startswith("MOCK_"))

    def test_execute_sell(self):
        """
        execute_sell returns a mock FILLED order in paper mode without calling
        the live create_order API.
        """
        order = self.executor.execute_sell("BTCUSDT", 0.1, 50000.0)
        self.executor.client.create_order.assert_not_called()
        self.assertEqual(order["status"], "FILLED")
        self.assertEqual(order["side"], "SELL")
        self.assertTrue(order["orderId"].startswith("MOCK_"))

    def test_execute_stop_loss(self):
        """
        With no open position, execute_stop_loss reports NO_POSITION and does
        not call the live create_order API.
        """
        with patch(
            "trading.execution.binance_executor.get_open_positions", return_value=[]
        ):
            result = self.executor.execute_stop_loss("BTCUSDT", 0.1, 49000.0)
        self.executor.client.create_order.assert_not_called()
        self.assertEqual(result["status"], "NO_POSITION")

    def test_execute_take_profit(self):
        """
        With no open position, execute_take_profit reports NO_POSITION and does
        not call the live create_order API.
        """
        with patch(
            "trading.execution.binance_executor.get_open_positions", return_value=[]
        ):
            result = self.executor.execute_take_profit("BTCUSDT", 0.1, 51000.0)
        self.executor.client.create_order.assert_not_called()
        self.assertEqual(result["status"], "NO_POSITION")

    def test_cancel_order(self):
        """
        cancel_order is now a paper-mode stub taking a single order_id and
        returning True; it does not call the live client.
        """
        self.assertTrue(self.executor.cancel_order("12345"))
        self.executor.client.cancel_order.assert_not_called()

    def test_get_order_status(self):
        """
        get_order_status is now a paper-mode stub taking a single order_id and
        returning an empty dict; it does not call the live client.
        """
        self.assertEqual(self.executor.get_order_status("12345"), {})
        self.executor.client.get_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
