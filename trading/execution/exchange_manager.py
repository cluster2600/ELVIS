"""
Exchange Manager
Unified interface for managing multiple cryptocurrency exchanges
Provides exchange routing, arbitrage detection, and multi-exchange trading capabilities
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .base_executor import BaseExecutor
from .binance_executor import BinanceExecutor
from .kraken_executor import KrakenExecutor
from .coinbase_executor import CoinbaseExecutor
from utils.logger_config import get_logger


class ExchangeManager:
    """
    Manages multiple exchange connections and provides unified trading interface
    Supports arbitrage detection, best price routing, and multi-exchange strategies
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize Exchange Manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        
        # Exchange instances
        self.exchanges: Dict[str, BaseExecutor] = {}
        self.exchange_configs = {}
        
        # Price cache for arbitrage detection
        self.price_cache = {}
        self.price_cache_ttl = 10  # seconds
        self.price_lock = threading.Lock()
        
        # Exchange health monitoring
        self.exchange_health = {}
        self.health_check_interval = 60  # seconds
        
        # Arbitrage settings
        self.min_arbitrage_profit = 0.005  # 0.5% minimum profit
        self.max_trade_amount = 1.0  # BTC equivalent
        
        self.logger.info("Exchange Manager initialized")
    
    def add_exchange(self, name: str, executor_class, config: Dict[str, Any]) -> bool:
        """
        Add an exchange to the manager
        
        Args:
            name: Exchange name (binance, kraken, coinbase)
            executor_class: Executor class
            config: Exchange configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Store config
            self.exchange_configs[name] = config
            
            # Initialize executor
            executor = executor_class(
                logger=self.logger,
                **config
            )
            
            # Test connection
            if executor.initialize():
                self.exchanges[name] = executor
                self.exchange_health[name] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'error_count': 0,
                    'last_error': None
                }
                self.logger.info(f"✓ Added exchange: {name}")
                return True
            else:
                self.logger.error(f"✗ Failed to initialize exchange: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding exchange {name}: {e}")
            return False
    
    def remove_exchange(self, name: str) -> bool:
        """Remove an exchange from the manager"""
        try:
            if name in self.exchanges:
                del self.exchanges[name]
                del self.exchange_configs[name]
                if name in self.exchange_health:
                    del self.exchange_health[name]
                self.logger.info(f"Removed exchange: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing exchange {name}: {e}")
            return False
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return [name for name, health in self.exchange_health.items() 
                if health['status'] == 'healthy']
    
    def get_exchange(self, name: str) -> Optional[BaseExecutor]:
        """Get specific exchange executor"""
        return self.exchanges.get(name)
    
    def check_exchange_health(self, name: str) -> Dict[str, Any]:
        """Check health of a specific exchange"""
        try:
            if name not in self.exchanges:
                return {'status': 'not_found', 'message': 'Exchange not configured'}
            
            executor = self.exchanges[name]
            
            # Check if this is paper trading mode
            if hasattr(executor, 'is_testnet') and executor.is_testnet and getattr(executor, 'client', None) is None:
                # For paper trading, just mark as healthy since no real API connection needed
                self.exchange_health[name] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'error_count': 0,
                    'last_error': None,
                    'test_price': 97000.0,  # Mock price for paper trading
                    'mode': 'paper_trading'
                }
                return self.exchange_health[name]
            
            # Try to get current price as health check for live trading
            price = executor.get_current_price('BTCUSDT')
            
            if price > 0:
                self.exchange_health[name] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'error_count': 0,
                    'last_error': None,
                    'test_price': price
                }
                return self.exchange_health[name]
            else:
                raise Exception("Invalid price response")
                
        except Exception as e:
            self.exchange_health[name] = {
                'status': 'unhealthy',
                'last_check': datetime.now(),
                'error_count': self.exchange_health.get(name, {}).get('error_count', 0) + 1,
                'last_error': str(e)
            }
            self.logger.warning(f"Exchange {name} health check failed: {e}")
            return self.exchange_health[name]
    
    def check_all_exchanges_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all exchanges"""
        health_results = {}
        
        for name in self.exchanges.keys():
            health_results[name] = self.check_exchange_health(name)
        
        return health_results
    
    def get_prices_all_exchanges(self, symbol: str) -> Dict[str, float]:
        """Get current price from all available exchanges"""
        prices = {}
        
        def get_price(exchange_name: str, executor: BaseExecutor) -> Tuple[str, float]:
            try:
                if self.exchange_health.get(exchange_name, {}).get('status') != 'healthy':
                    return exchange_name, 0.0
                
                # Handle paper trading mode
                if hasattr(executor, 'is_testnet') and executor.is_testnet and getattr(executor, 'client', None) is None:
                    # Return mock price for paper trading
                    mock_price = executor._get_mock_price(symbol) if hasattr(executor, '_get_mock_price') else 97000.0
                    return exchange_name, mock_price
                
                price = executor.get_current_price(symbol)
                return exchange_name, price
            except Exception as e:
                self.logger.warning(f"Error getting price from {exchange_name}: {e}")
                return exchange_name, 0.0
        
        # Use ThreadPoolExecutor for concurrent price fetching
        max_workers = max(1, len(self.exchanges))  # Ensure at least 1 worker
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_exchange = {
                executor.submit(get_price, name, ex): name 
                for name, ex in self.exchanges.items()
            }
            
            for future in as_completed(future_to_exchange):
                exchange_name, price = future.result()
                if price > 0:
                    prices[exchange_name] = price
        
        # Update price cache
        with self.price_lock:
            self.price_cache[symbol] = {
                'prices': prices,
                'timestamp': datetime.now()
            }
        
        return prices
    
    def get_best_price(self, symbol: str, side: str) -> Tuple[str, float]:
        """
        Get best price across all exchanges
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            
        Returns:
            Tuple of (exchange_name, best_price)
        """
        prices = self.get_prices_all_exchanges(symbol)
        
        if not prices:
            return None, 0.0
        
        if side.lower() == 'buy':
            # For buying, we want the lowest ask price
            best_exchange = min(prices.keys(), key=lambda x: prices[x])
            best_price = prices[best_exchange]
        else:
            # For selling, we want the highest bid price
            best_exchange = max(prices.keys(), key=lambda x: prices[x])
            best_price = prices[best_exchange]
        
        return best_exchange, best_price
    
    def detect_arbitrage_opportunities(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect arbitrage opportunities across exchanges
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of arbitrage opportunities
        """
        prices = self.get_prices_all_exchanges(symbol)
        
        if len(prices) < 2:
            return []
        
        opportunities = []
        
        # Find price differences
        for buy_exchange in prices:
            for sell_exchange in prices:
                if buy_exchange == sell_exchange:
                    continue
                
                buy_price = prices[buy_exchange]
                sell_price = prices[sell_exchange]
                
                # Calculate potential profit
                if sell_price > buy_price:
                    profit_pct = (sell_price - buy_price) / buy_price
                    
                    if profit_pct >= self.min_arbitrage_profit:
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct,
                            'profit_abs': sell_price - buy_price,
                            'timestamp': datetime.now().isoformat()
                        })
        
        # Sort by profit percentage (descending)
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        
        return opportunities
    
    def execute_arbitrage(self, opportunity: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """
        Execute arbitrage trade
        
        Args:
            opportunity: Arbitrage opportunity from detect_arbitrage_opportunities
            amount: Amount to trade
            
        Returns:
            Trade execution result
        """
        try:
            symbol = opportunity['symbol']
            buy_exchange = opportunity['buy_exchange']
            sell_exchange = opportunity['sell_exchange']
            
            # Limit trade amount
            trade_amount = min(amount, self.max_trade_amount)
            
            # Get executors
            buy_executor = self.exchanges[buy_exchange]
            sell_executor = self.exchanges[sell_exchange]
            
            # Execute trades simultaneously
            results = []
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both orders
                buy_future = executor.submit(
                    buy_executor.execute_buy, symbol, trade_amount
                )
                sell_future = executor.submit(
                    sell_executor.execute_sell, symbol, trade_amount
                )
                
                # Get results
                buy_result = buy_future.result()
                sell_result = sell_future.result()
            
            arbitrage_result = {
                'symbol': symbol,
                'amount': trade_amount,
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'buy_result': buy_result,
                'sell_result': sell_result,
                'success': 'error' not in buy_result and 'error' not in sell_result,
                'timestamp': datetime.now().isoformat()
            }
            
            if arbitrage_result['success']:
                expected_profit = opportunity['profit_abs'] * trade_amount
                arbitrage_result['expected_profit'] = expected_profit
                self.logger.info(f"Arbitrage executed: {symbol} - Expected profit: ${expected_profit:.2f}")
            else:
                self.logger.error(f"Arbitrage execution failed: {arbitrage_result}")
            
            return arbitrage_result
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {e}")
            return {'error': str(e), 'success': False}
    
    def execute_smart_order(self, symbol: str, side: str, quantity: float, 
                          order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute order on the exchange with best price
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type ('market' or 'limit')
            
        Returns:
            Order execution result
        """
        try:
            # Find best exchange
            best_exchange, best_price = self.get_best_price(symbol, side)
            
            if not best_exchange:
                return {'error': 'No available exchanges', 'success': False}
            
            executor = self.exchanges[best_exchange]
            
            # Execute order
            if side.lower() == 'buy':
                if order_type == 'market':
                    result = executor.execute_buy(symbol, quantity)
                else:
                    result = executor.execute_buy(symbol, quantity, best_price)
            else:
                if order_type == 'market':
                    result = executor.execute_sell(symbol, quantity)
                else:
                    result = executor.execute_sell(symbol, quantity, best_price)
            
            result['selected_exchange'] = best_exchange
            result['selected_price'] = best_price
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing smart order: {e}")
            return {'error': str(e), 'success': False}
    
    def get_consolidated_balance(self) -> Dict[str, Dict[str, float]]:
        """Get consolidated balance across all exchanges"""
        consolidated = {}
        
        for exchange_name, executor in self.exchanges.items():
            try:
                if self.exchange_health.get(exchange_name, {}).get('status') != 'healthy':
                    continue
                
                balance = executor.get_balance()
                
                for currency, amounts in balance.items():
                    if currency not in consolidated:
                        consolidated[currency] = {
                            'total_free': 0.0,
                            'total_locked': 0.0,
                            'total_balance': 0.0,
                            'exchanges': {}
                        }
                    
                    consolidated[currency]['total_free'] += amounts['free']
                    consolidated[currency]['total_locked'] += amounts['locked']
                    consolidated[currency]['total_balance'] += amounts['total']
                    consolidated[currency]['exchanges'][exchange_name] = amounts
                    
            except Exception as e:
                self.logger.warning(f"Error getting balance from {exchange_name}: {e}")
        
        return consolidated
    
    def get_market_summary(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market summary across all exchanges"""
        summary = {}
        
        for symbol in symbols:
            prices = self.get_prices_all_exchanges(symbol)
            
            if prices:
                summary[symbol] = {
                    'prices': prices,
                    'min_price': min(prices.values()),
                    'max_price': max(prices.values()),
                    'avg_price': sum(prices.values()) / len(prices),
                    'price_spread': max(prices.values()) - min(prices.values()),
                    'spread_pct': (max(prices.values()) - min(prices.values())) / min(prices.values()) * 100,
                    'arbitrage_opportunities': self.detect_arbitrage_opportunities(symbol)
                }
        
        return summary
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        def monitor():
            while True:
                try:
                    self.check_all_exchanges_health()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Error in health monitoring: {e}")
                    time.sleep(10)
        
        monitoring_thread = threading.Thread(target=monitor, daemon=True)
        monitoring_thread.start()
        self.logger.info("Exchange health monitoring started")
    
    def get_exchange_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured exchanges"""
        info = {}
        
        for name, executor in self.exchanges.items():
            try:
                info[name] = {
                    'exchange_info': executor.get_exchange_info(),
                    'health': self.exchange_health.get(name, {}),
                    'config': {k: v for k, v in self.exchange_configs.get(name, {}).items() 
                             if k not in ['api_key', 'api_secret', 'passphrase']}  # Hide sensitive data
                }
            except Exception as e:
                info[name] = {'error': str(e)}
        
        return info