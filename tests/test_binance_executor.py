"""
Unit tests for the BinanceExecutor class.
"""

import unittest
import logging
import json
from unittest.mock import patch, MagicMock, call

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
        
        # Mock exchange
        self.executor.exchange = MagicMock()
        
        # Test symbol
        self.symbol = 'BTC/USDT'
    
    @patch('ccxt.binance')
    def test_init(self, mock_binance):
        """
        Test the initialization of the BinanceExecutor.
        """
        # Set up mock
        mock_binance.return_value = MagicMock()
        
        # Create executor
        executor = BinanceExecutor(
            logger=self.logger
        )
        
        # Check attributes
        self.assertEqual(executor.logger, self.logger)
        self.assertIsNone(executor.exchange)
        self.assertEqual(executor.orders, {})
        self.assertEqual(executor.positions, {})
    
    def test_initialize(self):
        """
        Test the initialize method.
        """
        # Reset exchange to None
        self.executor.exchange = None
        
        # Mock ccxt.binance
        with patch('ccxt.binance') as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange
            
            # Call method
            self.executor.initialize()
            
            # Check exchange initialization
            self.assertEqual(self.executor.exchange, mock_exchange)
            mock_binance.assert_called_once()
    
    def test_get_balance(self):
        """
        Test the get_balance method.
        """
        # Set up mock
        mock_balance = {
            'USDT': {'free': 1000.0, 'used': 500.0, 'total': 1500.0},
            'BTC': {'free': 0.1, 'used': 0.05, 'total': 0.15}
        }
        self.executor.exchange.fetch_balance.return_value = mock_balance
        
        # Call method
        result = self.executor.get_balance()
        
        # Check result
        self.assertEqual(result['USDT'], 1500.0)
        self.assertEqual(result['BTC'], 0.15)
        
        # Check exchange call
        self.executor.exchange.fetch_balance.assert_called_once_with({'type': 'future'})
    
    def test_get_position(self):
        """
        Test the get_position method.
        """
        # Set up mock
        mock_position = [{
            'symbol': self.symbol,
            'contracts': 0.1,
            'notional': 3600.0,
            'leverage': 10,
            'entryPrice': 36000.0,
            'unrealizedPnl': 100.0,
            'side': 'long'
        }]
        self.executor.exchange.fetch_positions.return_value = mock_position
        
        # Call method
        result = self.executor.get_position(self.symbol)
        
        # Check result
        self.assertEqual(result['symbol'], self.symbol)
        self.assertEqual(result['contracts'], 0.1)
        self.assertEqual(result['notional'], 3600.0)
        self.assertEqual(result['leverage'], 10)
        self.assertEqual(result['entryPrice'], 36000.0)
        self.assertEqual(result['unrealizedPnl'], 100.0)
        self.assertEqual(result['side'], 'long')
        
        # Check exchange call
        self.executor.exchange.fetch_positions.assert_called_once_with([self.symbol])
    
    def test_get_current_price(self):
        """
        Test the get_current_price method.
        """
        # Set up mock
        mock_ticker = {'last': 36000.0}
        self.executor.exchange.fetch_ticker.return_value = mock_ticker
        
        # Call method
        result = self.executor.get_current_price(self.symbol)
        
        # Check result
        self.assertEqual(result, 36000.0)
        
        # Check exchange call
        self.executor.exchange.fetch_ticker.assert_called_once_with(self.symbol)
    
    def test_set_leverage(self):
        """
        Test the set_leverage method.
        """
        # Call method
        self.executor.set_leverage(self.symbol, 10)
        
        # Check exchange call
        self.executor.exchange.set_leverage.assert_called_once_with(10, self.symbol)
        
        # Check current leverage
        self.assertEqual(self.executor.current_leverage, 10)
    
    def test_adjust_quantity(self):
        """
        Test the adjust_quantity method.
        """
        # Set up mock
        mock_market = {
            'info': {
                'filters': [
                    {
                        'filterType': 'LOT_SIZE',
                        'stepSize': '0.001',
                        'minQty': '0.001'
                    },
                    {
                        'filterType': 'MIN_NOTIONAL',
                        'notional': '10'
                    }
                ]
            }
        }
        self.executor.exchange.market.return_value = mock_market
        
        # Call method
        result = self.executor.adjust_quantity(self.symbol, 0.1234, 36000.0)
        
        # Check result
        self.assertEqual(result, 0.123)  # Rounded to step size
        
        # Check exchange call
        self.executor.exchange.market.assert_called_once_with(self.symbol)
    
    def test_execute_buy(self):
        """
        Test the execute_buy method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'buy',
            'price': 36000.0,
            'amount': 0.1,
            'status': 'open'
        }
        self.executor.exchange.create_limit_buy_order.return_value = mock_order
        self.executor.adjust_quantity = MagicMock(return_value=0.1)
        
        # Call method
        result = self.executor.execute_buy(self.symbol, 0.1, 36000.0)
        
        # Check result
        self.assertEqual(result, mock_order)
        
        # Check exchange call
        self.executor.exchange.create_limit_buy_order.assert_called_once_with(
            self.symbol, 0.1, 36000.0, params={"reduceOnly": False}
        )
        
        # Check order storage
        self.assertEqual(self.executor.orders['123456'], mock_order)
    
    def test_execute_sell(self):
        """
        Test the execute_sell method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'sell',
            'price': 36000.0,
            'amount': 0.1,
            'status': 'open'
        }
        self.executor.exchange.create_limit_sell_order.return_value = mock_order
        self.executor.adjust_quantity = MagicMock(return_value=0.1)
        
        # Call method
        result = self.executor.execute_sell(self.symbol, 0.1, 36000.0)
        
        # Check result
        self.assertEqual(result, mock_order)
        
        # Check exchange call
        self.executor.exchange.create_limit_sell_order.assert_called_once_with(
            self.symbol, 0.1, 36000.0, params={"reduceOnly": True}
        )
        
        # Check order storage
        self.assertEqual(self.executor.orders['123456'], mock_order)
    
    def test_execute_stop_loss(self):
        """
        Test the execute_stop_loss method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'sell',
            'price': None,
            'stopPrice': 35000.0,
            'amount': 0.1,
            'status': 'open'
        }
        self.executor.exchange.create_order.return_value = mock_order
        self.executor.adjust_quantity = MagicMock(return_value=0.1)
        
        # Call method
        result = self.executor.execute_stop_loss(self.symbol, 0.1, 35000.0)
        
        # Check result
        self.assertEqual(result, mock_order)
        
        # Check exchange call
        self.executor.exchange.create_order.assert_called_once_with(
            self.symbol, 'stop', 'sell', 0.1, None, 
            params={'stopPrice': 35000.0, 'reduceOnly': True}
        )
        
        # Check order storage
        self.assertEqual(self.executor.orders['123456'], mock_order)
    
    def test_execute_take_profit(self):
        """
        Test the execute_take_profit method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'sell',
            'price': 37000.0,
            'amount': 0.1,
            'status': 'open'
        }
        self.executor.exchange.create_order.return_value = mock_order
        self.executor.adjust_quantity = MagicMock(return_value=0.1)
        
        # Call method
        result = self.executor.execute_take_profit(self.symbol, 0.1, 37000.0)
        
        # Check result
        self.assertEqual(result, mock_order)
        
        # Check exchange call
        self.executor.exchange.create_order.assert_called_once_with(
            self.symbol, 'limit', 'sell', 0.1, 37000.0, 
            params={'reduceOnly': True}
        )
        
        # Check order storage
        self.assertEqual(self.executor.orders['123456'], mock_order)
    
    def test_cancel_order(self):
        """
        Test the cancel_order method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'buy',
            'price': 36000.0,
            'amount': 0.1,
            'status': 'open'
        }
        self.executor.orders = {'123456': mock_order}
        
        # Call method
        result = self.executor.cancel_order('123456')
        
        # Check result
        self.assertTrue(result)
        
        # Check exchange call
        self.executor.exchange.cancel_order.assert_called_once_with('123456', self.symbol)
        
        # Check order removal
        self.assertNotIn('123456', self.executor.orders)
    
    def test_get_order_status(self):
        """
        Test the get_order_status method.
        """
        # Set up mocks
        mock_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'buy',
            'price': 36000.0,
            'amount': 0.1,
            'status': 'open'
        }
        mock_updated_order = {
            'id': '123456',
            'symbol': self.symbol,
            'side': 'buy',
            'price': 36000.0,
            'amount': 0.1,
            'status': 'closed'
        }
        self.executor.orders = {'123456': mock_order}
        self.executor.exchange.fetch_order.return_value = mock_updated_order
        
        # Call method
        result = self.executor.get_order_status('123456')
        
        # Check result
        self.assertEqual(result, mock_updated_order)
        
        # Check exchange call
        self.executor.exchange.fetch_order.assert_called_once_with('123456', self.symbol)
        
        # Check order update
        self.assertEqual(self.executor.orders['123456'], mock_updated_order)

if __name__ == '__main__':
    unittest.main()
