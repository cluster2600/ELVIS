"""
Unit tests for the BinanceProcessor class.
"""

import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from core.data.processors.binance_processor import BinanceProcessor


class TestBinanceProcessor(unittest.TestCase):
    """
    Test cases for the BinanceProcessor class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Set up logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

        # Set up dates
        self.start_date = (datetime.now() - timedelta(days=7)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Set up processor
        self.processor = BinanceProcessor(
            data_source="binance",
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval="1h",
            logger=self.logger,
        )

        # Mock client
        self.processor.exchange = MagicMock()

    def test_init(self):
        """
        Test the initialization of the BinanceProcessor.
        """
        # Create processor
        processor = BinanceProcessor(
            data_source="binance",
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval="1h",
            logger=self.logger,
        )

        # Check attributes
        self.assertEqual(processor.data_source, "binance")
        self.assertEqual(processor.start_date, self.start_date)
        self.assertEqual(processor.end_date, self.end_date)
        self.assertEqual(processor.time_interval, "1h")
        self.assertEqual(processor.logger, self.logger)

    def test_download_data(self):
        """
        Test the download_data method.
        """
        # Set up mock
        mock_ohlcv = [
            [1625097600000, "35000.0", "36000.0", "34000.0", "35500.0", "100.0"]
        ]
        self.processor.exchange.fetch_ohlcv.return_value = mock_ohlcv

        # Call method
        result = self.processor.download_data(["BTC/USDT"])

        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            list(result.columns),
            ["date", "open", "high", "low", "close", "volume", "ticker"],
        )

        # Check exchange call
        self.processor.exchange.fetch_ohlcv.assert_called_once()

    def test_clean_data(self):
        """
        Test the clean_data method.
        """
        # Set up data
        self.processor.data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    pd.date_range(start=self.start_date, periods=5, freq="h")
                ),
                "open": [35000.0, 35500.0, 36000.0, 36500.0, 37000.0],
                "high": [36000.0, 36500.0, 37000.0, 37500.0, 38000.0],
                "low": [34000.0, 34500.0, 35000.0, 35500.0, 36000.0],
                "close": [35500.0, 36000.0, 36500.0, 37000.0, 37500.0],
                "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

        # Add duplicates
        duplicate = self.processor.data.iloc[0].copy()
        self.processor.data = pd.concat(
            [self.processor.data, pd.DataFrame([duplicate])], ignore_index=True
        )

        # Add NaN values
        self.processor.data.loc[2, "close"] = np.nan

        # Call method
        result = self.processor.clean_data()

        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # Duplicates removed
        self.assertFalse(result.isnull().any().any())  # NaN values filled

    def test_add_technical_indicator(self):
        """
        Test the add_technical_indicator method.
        """
        # Set up data
        self.processor.data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    pd.date_range(start=self.start_date, periods=40, freq="h")
                ),
                "open": np.random.uniform(35000.0, 37000.0, 40),
                "high": np.random.uniform(36000.0, 38000.0, 40),
                "low": np.random.uniform(34000.0, 36000.0, 40),
                "close": np.random.uniform(35000.0, 37000.0, 40),
                "volume": np.random.uniform(100.0, 500.0, 40),
                "ticker": "BTC/USDT",
            }
        )

        # Call method
        result = self.processor.add_technical_indicator(["rsi", "macd", "bbands"])

        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("rsi" in result.columns)
        self.assertTrue("macd" in result.columns)
        self.assertTrue("macdsignal" in result.columns)
        self.assertTrue("macdhist" in result.columns)
        self.assertTrue("upperband" in result.columns)
        self.assertTrue("middleband" in result.columns)
        self.assertTrue("lowerband" in result.columns)

    def test_df_to_array(self):
        """
        Test the df_to_array method.
        """
        # Set up data
        self.processor.data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    pd.date_range(start=self.start_date, periods=20, freq="h")
                ),
                "open": np.random.uniform(35000.0, 37000.0, 20),
                "high": np.random.uniform(36000.0, 38000.0, 20),
                "low": np.random.uniform(34000.0, 36000.0, 20),
                "close": np.random.uniform(35000.0, 37000.0, 20),
                "volume": np.random.uniform(100.0, 500.0, 20),
                "ticker": "BTC/USDT",
                "rsi": np.random.uniform(30.0, 70.0, 20),
                "macd": np.random.uniform(-10.0, 10.0, 20),
            }
        )

        # Call method
        data, price_array, tech_array, time_array = self.processor.df_to_array(
            ["rsi", "macd"], False
        )

        # Check result
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(price_array, np.ndarray)
        self.assertIsInstance(tech_array, np.ndarray)
        self.assertIsInstance(time_array, pd.arrays.DatetimeArray)

        self.assertEqual(price_array.shape, (20, 1))
        self.assertEqual(tech_array.shape, (20, 2))
        self.assertEqual(len(time_array), 20)


if __name__ == "__main__":
    unittest.main()
