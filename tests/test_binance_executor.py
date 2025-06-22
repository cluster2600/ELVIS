"""
Unit tests for the BinanceExecutor class.
"""

import unittest
import logging
from unittest.mock import patch, MagicMock

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
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Set up executor
        self.executor = BinanceExecutor(
            logger=self.logger
        )
        
        # Mock client
        self.executor.client = MagicMock()
        
        # Test symbol
        self.symbol = 'BTCUSDT'
    
    def test_initialize(self):
        """
        Test the initialize method.
        """
        with patch('trading.execution.binance_executor.API_CONFIG', {'API_KEY': 'test', 'API_SECRET': 'test'}):
            with patch('trading.execution.binance_executor.Client') as mock_client:
                self.executor.initialize()
                mock_client.assert_called_once()

    def test_get_balance(self):
        """
        Test the get_balance method.
        """
        self.executor.client.get_account.return_value = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0'},
                {'asset': 'BTC', 'free': '0.1'}
            ]
        }
        balance = self.executor.get_balance()
        self.assertEqual(balance['USDT'], 1000.0)
        self.assertEqual(balance['BTC'], 0.1)

    def test_get_position(self):
        """
        Test the get_position method.
        """
        self.executor.client.get_account.return_value = {
            'positions': [
                {'symbol': 'BTCUSDT', 'positionAmt': '0.1'}
            ]
        }
        position = self.executor.get_position('BTCUSDT')
        self.assertEqual(position['positionAmt'], '0.1')

    def test_get_current_price(self):
        """
        Test the get_current_price method.
        """
        self.executor.client.get_symbol_ticker.return_value = {'price': '50000.0'}
        price = self.executor.get_current_price('BTCUSDT')
        self.assertEqual(price, 50000.0)

    def test_set_leverage(self):
        """
        Test the set_leverage method.
        """
        self.executor.set_leverage('BTCUSDT', 10)
        self.executor.client.change_leverage.assert_called_once_with(symbol='BTCUSDT', leverage=10)

    def test_execute_buy(self):
        """
        Test the execute_buy method.
        """
        self.executor.execute_buy('BTCUSDT', 0.1, 50000.0)
        self.executor.client.create_order.assert_called_once()

    def test_execute_sell(self):
        """
        Test the execute_sell method.
        """
        self.executor.execute_sell('BTCUSDT', 0.1, 50000.0)
        self.executor.client.create_order.assert_called_once()

    def test_execute_stop_loss(self):
        """
        Test the execute_stop_loss method.
        """
        self.executor.execute_stop_loss('BTCUSDT', 0.1, 49000.0)
        self.executor.client.create_order.assert_called_once()

    def test_execute_take_profit(self):
        """
        Test the execute_take_profit method.
        """
        self.executor.execute_take_profit('BTCUSDT', 0.1, 51000.0)
        self.executor.client.create_order.assert_called_once()

    def test_cancel_order(self):
        """
        Test the cancel_order method.
        """
        self.executor.cancel_order('BTCUSDT', '12345')
        self.executor.client.cancel_order.assert_called_once_with(symbol='BTCUSDT', orderId='12345')

    def test_get_order_status(self):
        """
        Test the get_order_status method.
        """
        self.executor.get_order_status('BTCUSDT', '12345')
        self.executor.client.get_order.assert_called_once_with(symbol='BTCUSDT', orderId='12345')

if __name__ == '__main__':
    unittest.main()
