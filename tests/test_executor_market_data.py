"""
Unit tests for the market-data helpers on BinanceExecutor.

These exercise get_funding_rate() and get_order_book() in paper mode
(no live client), so they run without any Binance connectivity.
"""

import logging
import unittest

from trading.execution.binance_executor import BinanceExecutor


class TestExecutorMarketData(unittest.TestCase):
    """Paper-mode coverage for the executor market-data methods."""

    def setUp(self):
        self.logger = logging.getLogger("test_market_data")
        self.logger.setLevel(logging.INFO)
        # Default executor: paper mode, client is None (no live connection).
        self.executor = BinanceExecutor(logger=self.logger)
        self.symbol = "BTCUSDT"

    def test_get_funding_rate_paper(self):
        """Paper mode returns a zero-rate mock structure without a client."""
        self.assertIsNone(self.executor.client)
        result = self.executor.get_funding_rate(self.symbol)
        self.assertEqual(result["symbol"], self.symbol)
        self.assertEqual(result["fundingRate"], 0.0)
        self.assertIn("ts", result)
        self.assertIsInstance(result["ts"], int)

    def test_get_order_book_paper(self):
        """Paper mode returns an empty book structure without a client."""
        self.assertIsNone(self.executor.client)
        result = self.executor.get_order_book(self.symbol)
        self.assertEqual(result["symbol"], self.symbol)
        self.assertEqual(result["bids"], [])
        self.assertEqual(result["asks"], [])
        self.assertIn("timestamp", result)
        self.assertIsInstance(result["timestamp"], int)

    def test_get_order_book_paper_custom_limit(self):
        """The limit argument is accepted and the empty-book shape is stable."""
        result = self.executor.get_order_book(self.symbol, limit=5)
        self.assertEqual(result["symbol"], self.symbol)
        self.assertEqual(result["bids"], [])
        self.assertEqual(result["asks"], [])


if __name__ == "__main__":
    unittest.main()
