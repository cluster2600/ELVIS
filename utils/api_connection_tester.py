"""
API Connection Tester for ELVIS Trading Bot
Tests connectivity and health of all external APIs and services
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import requests
from dataclasses import dataclass
from enum import Enum

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"
    UNKNOWN = "unknown"

@dataclass
class APIStatus:
    name: str
    status: ConnectionStatus
    response_time: float
    last_checked: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class APIConnectionTester:
    """Tests and monitors API connections for the trading bot"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.api_statuses: Dict[str, APIStatus] = {}
        self.test_interval = 30  # Test every 30 seconds
        self.running = False
        self._test_thread = None
        
        # Initialize all API statuses
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API status tracking"""
        apis = [
            "binance_spot", "binance_futures", "binance_testnet",
            "postgres", "redis", "vault", "telegram", "prometheus"
        ]
        
        for api in apis:
            self.api_statuses[api] = APIStatus(
                name=api,
                status=ConnectionStatus.UNKNOWN,
                response_time=0.0,
                last_checked=datetime.now()
            )
    
    def test_binance_spot(self) -> APIStatus:
        """Test Binance Spot API connection"""
        start_time = time.time()
        try:
            from config.config import API_CONFIG
            
            # Test public endpoint first (no auth required)
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=5
            )
            
            if response.status_code == 200:
                # Test server time (public endpoint)
                response = requests.get(
                    "https://api.binance.com/api/v3/time",
                    timeout=5
                )
                
                if response.status_code == 200:
                    server_time = response.json()
                    response_time = time.time() - start_time
                    
                    return APIStatus(
                        name="binance_spot",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "server_time": server_time.get("serverTime"),
                            "endpoint": "api.binance.com"
                        }
                    )
            
            return APIStatus(
                name="binance_spot",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
                
        except Exception as e:
            return APIStatus(
                name="binance_spot",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_binance_futures(self) -> APIStatus:
        """Test Binance Futures API connection"""
        start_time = time.time()
        try:
            # Test public futures endpoint
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/ping",
                timeout=5
            )
            
            if response.status_code == 200:
                # Test exchange info
                response = requests.get(
                    "https://fapi.binance.com/fapi/v1/exchangeInfo",
                    timeout=5
                )
                
                if response.status_code == 200:
                    exchange_info = response.json()
                    response_time = time.time() - start_time
                    
                    return APIStatus(
                        name="binance_futures",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "symbols_count": len(exchange_info.get("symbols", [])),
                            "endpoint": "fapi.binance.com"
                        }
                    )
            
            return APIStatus(
                name="binance_futures",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
                
        except Exception as e:
            return APIStatus(
                name="binance_futures",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_binance_testnet(self) -> APIStatus:
        """Test Binance Testnet API connection"""
        start_time = time.time()
        try:
            # Test testnet futures endpoint
            response = requests.get(
                "https://testnet.binancefuture.com/fapi/v1/ping",
                timeout=5
            )
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                return APIStatus(
                    name="binance_testnet",
                    status=ConnectionStatus.CONNECTED,
                    response_time=response_time,
                    last_checked=datetime.now(),
                    details={"endpoint": "testnet.binancefuture.com"}
                )
            
            return APIStatus(
                name="binance_testnet",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
                
        except Exception as e:
            return APIStatus(
                name="binance_testnet",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_postgres(self) -> APIStatus:
        """Test PostgreSQL database connection"""
        start_time = time.time()
        try:
            from utils.paper_trade_db import get_conn
            
            conn = get_conn()
            if conn:
                # Test a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                
                response_time = time.time() - start_time
                conn.close()
                
                return APIStatus(
                    name="postgres",
                    status=ConnectionStatus.CONNECTED,
                    response_time=response_time,
                    last_checked=datetime.now(),
                    details={"version": version[:50]}  # Truncate version string
                )
            
            return APIStatus(
                name="postgres",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message="Failed to establish connection"
            )
                
        except Exception as e:
            return APIStatus(
                name="postgres",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_redis(self) -> APIStatus:
        """Test Redis connection"""
        start_time = time.time()
        try:
            from utils.redis_cache import RedisCache
            
            cache = RedisCache()
            if cache._is_connected():
                # Test a simple operation
                test_key = "api_test_key"
                cache.set(test_key, "test_value", ttl=10)
                result = cache.get(test_key)
                
                response_time = time.time() - start_time
                
                if result == "test_value":
                    return APIStatus(
                        name="redis",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "host": cache.host,
                            "port": cache.port,
                            "db": cache.db
                        }
                    )
            
            return APIStatus(
                name="redis",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message="Redis not connected"
            )
                
        except Exception as e:
            return APIStatus(
                name="redis",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_vault(self) -> APIStatus:
        """Test HashiCorp Vault connection"""
        start_time = time.time()
        try:
            # Ensure Vault environment is set
            import os
            if not os.getenv('VAULT_ADDR'):
                os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'
            if not os.getenv('VAULT_TOKEN'):
                os.environ['VAULT_TOKEN'] = 'trading-bot-token'
            
            # Test Vault directly with hvac client
            try:
                import hvac
                client = hvac.Client(
                    url=os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200'),
                    token=os.getenv('VAULT_TOKEN', 'trading-bot-token')
                )
                
                # Test authentication and health
                if client.is_authenticated():
                    # Try a simple operation
                    client.sys.list_auth_methods()
                    response_time = time.time() - start_time
                    
                    return APIStatus(
                        name="vault",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "url": os.getenv('VAULT_ADDR'),
                            "authenticated": True
                        }
                    )
                else:
                    response_time = time.time() - start_time
                    return APIStatus(
                        name="vault",
                        status=ConnectionStatus.ERROR,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        error_message="Not authenticated"
                    )
                    
            except ImportError:
                # Fallback to secrets manager if hvac not available
                from utils.secrets_manager import get_enhanced_secrets_manager
                secrets_manager = get_enhanced_secrets_manager()
                vault_status = secrets_manager.get_vault_status()
                response_time = time.time() - start_time
                
                if vault_status.get('healthy'):
                    return APIStatus(
                        name="vault",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "url": vault_status.get('url'),
                            "connected": vault_status.get('connected')
                        }
                    )
                elif vault_status.get('enabled'):
                    return APIStatus(
                    name="vault",
                    status=ConnectionStatus.ERROR,
                    response_time=response_time,
                    last_checked=datetime.now(),
                    error_message=vault_status.get('error', 'Vault unhealthy')
                )
            else:
                return APIStatus(
                    name="vault",
                    status=ConnectionStatus.DISCONNECTED,
                    response_time=response_time,
                    last_checked=datetime.now(),
                    error_message="Vault disabled - using fallback"
                )
                
        except Exception as e:
            return APIStatus(
                name="vault",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_telegram(self) -> APIStatus:
        """Test Telegram Bot API connection"""
        start_time = time.time()
        try:
            from utils.secrets_manager import get_enhanced_secrets_manager
            
            secrets_manager = get_enhanced_secrets_manager()
            telegram_creds = secrets_manager.get_telegram_credentials()
            bot_token = telegram_creds.get('bot_token')
            
            if not bot_token or 'your_' in bot_token:
                return APIStatus(
                    name="telegram",
                    status=ConnectionStatus.DISCONNECTED,
                    response_time=time.time() - start_time,
                    last_checked=datetime.now(),
                    error_message="No bot token configured"
                )
            
            # Test Telegram API
            response = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getMe",
                timeout=5
            )
            
            if response.status_code == 200:
                bot_info = response.json()
                response_time = time.time() - start_time
                
                if bot_info.get('ok'):
                    return APIStatus(
                        name="telegram",
                        status=ConnectionStatus.CONNECTED,
                        response_time=response_time,
                        last_checked=datetime.now(),
                        details={
                            "bot_name": bot_info['result'].get('username'),
                            "bot_id": bot_info['result'].get('id')
                        }
                    )
            
            return APIStatus(
                name="telegram",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
                
        except Exception as e:
            return APIStatus(
                name="telegram",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_prometheus(self) -> APIStatus:
        """Test Prometheus Pushgateway connection"""
        start_time = time.time()
        try:
            import os
            pushgateway_url = os.getenv('PROMETHEUS_PUSHGATEWAY_URL', 'http://localhost:9091')
            
            # Test Prometheus pushgateway
            response = requests.get(
                f"{pushgateway_url}/metrics",
                timeout=5
            )
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                return APIStatus(
                    name="prometheus",
                    status=ConnectionStatus.CONNECTED,
                    response_time=response_time,
                    last_checked=datetime.now(),
                    details={"url": pushgateway_url}
                )
            
            return APIStatus(
                name="prometheus",
                status=ConnectionStatus.ERROR,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
                
        except Exception as e:
            return APIStatus(
                name="prometheus",
                status=ConnectionStatus.DISCONNECTED,
                response_time=time.time() - start_time,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def test_all_apis(self) -> Dict[str, APIStatus]:
        """Test all APIs and return status dictionary"""
        test_methods = {
            "binance_spot": self.test_binance_spot,
            "binance_futures": self.test_binance_futures,
            "binance_testnet": self.test_binance_testnet,
            "postgres": self.test_postgres,
            "redis": self.test_redis,
            "vault": self.test_vault,
            "telegram": self.test_telegram,
            "prometheus": self.test_prometheus
        }
        
        for api_name, test_method in test_methods.items():
            try:
                # Mark as testing if status exists
                if api_name in self.api_statuses and self.api_statuses[api_name]:
                    self.api_statuses[api_name].status = ConnectionStatus.TESTING
                
                # Run test
                result = test_method()
                
                # Ensure result is valid before storing
                if result and hasattr(result, 'name') and hasattr(result, 'status'):
                    self.api_statuses[api_name] = result
                else:
                    self.logger.warning(f"Invalid result for {api_name}: {result}")
                    self.api_statuses[api_name] = APIStatus(
                        name=api_name,
                        status=ConnectionStatus.ERROR,
                        response_time=0.0,
                        last_checked=datetime.now(),
                        error_message="Invalid test result"
                    )
                
            except Exception as e:
                self.logger.error(f"Test failed for {api_name}: {e}")
                self.api_statuses[api_name] = APIStatus(
                    name=api_name,
                    status=ConnectionStatus.ERROR,
                    response_time=0.0,
                    last_checked=datetime.now(),
                    error_message=f"Test failed: {str(e)}"
                )
        
        return self.api_statuses.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        if not self.api_statuses:
            return {
                "overall_status": "unknown",
                "health_percentage": 0,
                "total_apis": 0,
                "connected": 0,
                "error": 0,
                "disconnected": 0,
                "critical_services": []
            }
        
        total_apis = len(self.api_statuses)
        valid_statuses = [status for status in self.api_statuses.values() if status and hasattr(status, 'status')]
        
        connected = sum(1 for status in valid_statuses 
                       if status.status == ConnectionStatus.CONNECTED)
        error = sum(1 for status in valid_statuses 
                   if status.status == ConnectionStatus.ERROR)
        disconnected = sum(1 for status in valid_statuses 
                          if status.status == ConnectionStatus.DISCONNECTED)
        
        health_percentage = (connected / total_apis) * 100 if total_apis > 0 else 0
        
        # Determine overall status
        if connected == total_apis:
            overall_status = "healthy"
        elif connected >= total_apis * 0.8:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "total_apis": total_apis,
            "connected": connected,
            "error": error,
            "disconnected": disconnected,
            "critical_services": [
                name for name, status in self.api_statuses.items()
                if status and hasattr(status, 'status') and status.status != ConnectionStatus.CONNECTED and name in ["binance_spot", "binance_futures", "postgres"]
            ]
        }
    
    def start_monitoring(self):
        """Start background monitoring of APIs"""
        if self.running:
            return
        
        self.running = True
        self._test_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._test_thread.start()
        self.logger.info("Started API connection monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self._test_thread and self._test_thread.is_alive():
            self._test_thread.join(timeout=5)
        self.logger.info("Stopped API connection monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self.test_all_apis()
                time.sleep(self.test_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Shorter sleep on error


# Global instance
_api_tester = None

def get_api_tester(logger: logging.Logger = None) -> APIConnectionTester:
    """Get or create global API connection tester"""
    global _api_tester
    if _api_tester is None:
        _api_tester = APIConnectionTester(logger)
    return _api_tester