"""
Unit tests for the market-data helpers on BinanceExecutor.

These exercise get_funding_rate() and get_order_book() in paper mode
(no live client). The public-depth fetch is mocked, so they run without
any Binance connectivity.
"""

import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from trading.execution.binance_executor import BinanceExecutor


class TestExecutorMarketData(unittest.TestCase):
    """Paper-mode coverage for the executor market-data methods."""

    def setUp(self):
        self.logger = logging.getLogger("test_market_data")
        self.logger.setLevel(logging.INFO)
        # Default executor: paper mode, client is None (no live connection).
        self.executor = BinanceExecutor(logger=self.logger)
        self.symbol = "BTCUSDT"
        # Force the offline paper book by default so tests never depend on
        # the network; the public-fetch tests re-enable it with a mock.
        self._prev_flag = os.environ.get("ELVIS_PAPER_PUBLIC_BOOK")
        os.environ["ELVIS_PAPER_PUBLIC_BOOK"] = "0"

    def tearDown(self):
        if self._prev_flag is None:
            os.environ.pop("ELVIS_PAPER_PUBLIC_BOOK", None)
        else:
            os.environ["ELVIS_PAPER_PUBLIC_BOOK"] = self._prev_flag

    def test_get_funding_rate_paper(self):
        """Paper mode returns a zero-rate mock structure without a client."""
        self.assertIsNone(self.executor.client)
        result = self.executor.get_funding_rate(self.symbol)
        self.assertEqual(result["symbol"], self.symbol)
        self.assertEqual(result["fundingRate"], 0.0)
        self.assertIn("ts", result)
        self.assertIsInstance(result["ts"], int)

    def test_get_order_book_paper_offline(self):
        """With ELVIS_PAPER_PUBLIC_BOOK=0 paper mode returns an empty book."""
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

    def test_get_order_book_paper_public_fetch(self):
        """Paper mode fetches the real public depth when the flag is on."""
        os.environ["ELVIS_PAPER_PUBLIC_BOOK"] = "1"
        fake = MagicMock()
        fake.json.return_value = {
            "bids": [["117000.00", "1.5"], ["116999.00", "0.2"]],
            "asks": [["117001.00", "0.9"]],
        }
        fake.raise_for_status.return_value = None
        with patch("requests.get", return_value=fake) as mock_get:
            result = self.executor.get_order_book(self.symbol, limit=10)
        self.assertEqual(result["symbol"], self.symbol)
        self.assertEqual(len(result["bids"]), 2)
        self.assertEqual(result["asks"], [["117001.00", "0.9"]])
        self.assertIsInstance(result["timestamp"], int)
        url = mock_get.call_args[0][0]
        params = mock_get.call_args[1]["params"]
        self.assertIn("api.binance.com/api/v3/depth", url)
        self.assertEqual(params["symbol"], self.symbol)
        self.assertEqual(params["limit"], 10)

    def test_get_order_book_paper_public_fetch_failure_falls_back(self):
        """A network failure degrades to the empty book, never raises."""
        os.environ["ELVIS_PAPER_PUBLIC_BOOK"] = "1"
        with patch("requests.get", side_effect=OSError("network down")):
            result = self.executor.get_order_book(self.symbol)
        self.assertEqual(result["bids"], [])
        self.assertEqual(result["asks"], [])
        self.assertEqual(result["symbol"], self.symbol)


if __name__ == "__main__":
    unittest.main()
