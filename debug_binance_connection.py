#!/usr/bin/env python3
"""
Debug Binance API connection issues
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment
os.environ['VAULT_ENABLED'] = 'false'

def test_config_loading():
    """Test configuration loading"""
    logger.info("🔧 Testing Configuration Loading")
    logger.info("-" * 50)
    
    try:
        # Test 1: Direct config import
        logger.info("1. Testing direct config import...")
        from config.config import API_CONFIG, TRADING_CONFIG
        
        logger.info(f"✅ API_CONFIG type: {type(API_CONFIG)}")
        logger.info(f"✅ TRADING_CONFIG type: {type(TRADING_CONFIG)}")
        
        # Test 2: Check if API_CONFIG is a class or dict
        if hasattr(API_CONFIG, '__dict__'):
            logger.info("📋 API_CONFIG attributes:")
            for attr in dir(API_CONFIG):
                if not attr.startswith('_'):
                    try:
                        value = getattr(API_CONFIG, attr)
                        # Mask sensitive data
                        if 'key' in attr.lower() or 'secret' in attr.lower():
                            display_value = f"{str(value)[:10]}..." if value else "None"
                        else:
                            display_value = value
                        logger.info(f"   {attr}: {display_value}")
                    except Exception as e:
                        logger.error(f"   {attr}: Error accessing - {e}")
        else:
            logger.info("📋 API_CONFIG is a dict:")
            for key, value in API_CONFIG.items():
                if 'key' in key.lower() or 'secret' in key.lower():
                    display_value = f"{str(value)[:10]}..." if value else "None"
                else:
                    display_value = value
                logger.info(f"   {key}: {display_value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_binance_imports():
    """Test Binance library imports"""
    logger.info("\n📦 Testing Binance Library Imports")
    logger.info("-" * 50)
    
    try:
        # Test futures import
        logger.info("1. Testing futures import...")
        try:
            from binance.um_futures import UMFutures
            from binance.error import ClientError
            logger.info("✅ Binance futures library available")
            futures_available = True
        except ImportError as e:
            logger.warning(f"⚠️ Futures library not available: {e}")
            futures_available = False
        
        # Test spot import
        logger.info("2. Testing spot import...")
        try:
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            logger.info("✅ Binance spot library available")
            spot_available = True
        except ImportError as e:
            logger.error(f"❌ Spot library not available: {e}")
            spot_available = False
        
        return futures_available, spot_available
        
    except Exception as e:
        logger.error(f"❌ Import testing failed: {e}")
        return False, False

def test_api_credentials():
    """Test API credentials access"""
    logger.info("\n🔑 Testing API Credentials Access")
    logger.info("-" * 50)
    
    try:
        from config.config import API_CONFIG
        
        # List of expected credential attributes
        expected_credentials = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET', 
            'BINANCE_FUTURES_TESTNET_API_KEY',
            'BINANCE_FUTURES_TESTNET_API_SECRET'
        ]
        
        credentials_status = {}
        
        for cred in expected_credentials:
            try:
                if hasattr(API_CONFIG, cred):
                    value = getattr(API_CONFIG, cred)
                    has_value = value is not None and value != '' and 'your_' not in str(value)
                    credentials_status[cred] = has_value
                    status = "✅ Valid" if has_value else "⚠️ Placeholder/Empty"
                    logger.info(f"   {cred}: {status}")
                else:
                    credentials_status[cred] = False
                    logger.error(f"   {cred}: ❌ Missing")
            except Exception as e:
                credentials_status[cred] = False
                logger.error(f"   {cred}: ❌ Error accessing - {e}")
        
        return credentials_status
        
    except Exception as e:
        logger.error(f"❌ Credentials testing failed: {e}")
        return {}

def test_binance_client_initialization():
    """Test Binance client initialization"""
    logger.info("\n🔌 Testing Binance Client Initialization")
    logger.info("-" * 50)
    
    try:
        from config.config import API_CONFIG
        
        # Test 1: Futures testnet client (most common for testing)
        logger.info("1. Testing Futures Testnet Client...")
        
        try:
            from binance.um_futures import UMFutures
            
            # Get credentials
            if hasattr(API_CONFIG, 'BINANCE_FUTURES_TESTNET_API_KEY'):
                api_key = API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY
                api_secret = API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET
            else:
                api_key = getattr(API_CONFIG, 'BINANCE_FUTURES_TESTNET_API_KEY', None)
                api_secret = getattr(API_CONFIG, 'BINANCE_FUTURES_TESTNET_API_SECRET', None)
            
            logger.info(f"   API Key available: {api_key is not None and 'your_' not in str(api_key)}")
            logger.info(f"   API Secret available: {api_secret is not None and 'your_' not in str(api_secret)}")
            
            if api_key and api_secret and 'your_' not in api_key:
                # Try to initialize client
                client = UMFutures(
                    key=api_key, 
                    secret=api_secret, 
                    base_url="https://testnet.binancefuture.com"
                )
                
                # Test connection with a simple call
                try:
                    server_time = client.time()
                    logger.info(f"✅ Futures testnet connection successful!")
                    logger.info(f"   Server time: {server_time}")
                    return True
                except Exception as e:
                    logger.error(f"❌ Futures testnet connection failed: {e}")
                    return False
            else:
                logger.warning("⚠️ No valid futures testnet credentials found")
                return False
                
        except ImportError:
            logger.warning("⚠️ Futures library not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Client initialization testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_executor_initialization():
    """Test executor initialization with detailed debugging"""
    logger.info("\n🤖 Testing Executor Initialization")
    logger.info("-" * 50)
    
    try:
        # Test the basic BinanceExecutor first
        logger.info("1. Testing basic BinanceExecutor...")
        from trading.execution.binance_executor import BinanceExecutor
        
        executor = BinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True
        )
        
        logger.info("   ✅ Executor created successfully")
        
        # Try to initialize it
        logger.info("2. Testing executor initialization...")
        try:
            executor.initialize()
            logger.info("   ✅ Executor initialized successfully")
            
            # Test balance retrieval
            logger.info("3. Testing balance retrieval...")
            balance = executor.get_balance()
            logger.info(f"   ✅ Balance retrieved: {balance}")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Executor initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"❌ Executor testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_environment_variables():
    """Test environment variables"""
    logger.info("\n🌍 Testing Environment Variables")
    logger.info("-" * 50)
    
    binance_env_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'BINANCE_FUTURES_TESTNET_API_KEY',
        'BINANCE_FUTURES_TESTNET_API_SECRET'
    ]
    
    found_vars = {}
    for var in binance_env_vars:
        value = os.getenv(var)
        found_vars[var] = value is not None
        status = "✅ Set" if value else "❌ Not Set"
        logger.info(f"   {var}: {status}")
    
    return found_vars

def create_test_credentials():
    """Create test credentials for demo purposes"""
    logger.info("\n⚙️ Setting up Test Credentials")
    logger.info("-" * 50)
    
    # For paper trading, we can use dummy credentials
    test_credentials = {
        'BINANCE_FUTURES_TESTNET_API_KEY': 'test_api_key_for_paper_trading',
        'BINANCE_FUTURES_TESTNET_API_SECRET': 'test_api_secret_for_paper_trading'
    }
    
    logger.info("Setting up dummy credentials for paper trading...")
    for key, value in test_credentials.items():
        os.environ[key] = value
        logger.info(f"   ✅ Set {key}")
    
    return test_credentials

def main():
    """Main diagnostic function"""
    logger.info("🔍 BINANCE CONNECTION DIAGNOSTIC")
    logger.info("=" * 60)
    
    # Test 1: Configuration loading
    config_ok = test_config_loading()
    
    # Test 2: Binance imports
    futures_ok, spot_ok = test_binance_imports()
    
    # Test 3: API credentials
    credentials = test_api_credentials()
    
    # Test 4: Environment variables
    env_vars = test_environment_variables()
    
    # Test 5: Client initialization
    client_ok = test_binance_client_initialization()
    
    # If client failed, try with test credentials
    if not client_ok:
        logger.info("\n🔧 Trying with test credentials for paper trading...")
        create_test_credentials()
        client_ok = test_binance_client_initialization()
    
    # Test 6: Executor initialization
    executor_ok = test_executor_initialization()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 DIAGNOSTIC SUMMARY:")
    logger.info(f"   Configuration Loading: {'✅' if config_ok else '❌'}")
    logger.info(f"   Futures Library: {'✅' if futures_ok else '❌'}")
    logger.info(f"   Spot Library: {'✅' if spot_ok else '❌'}")
    logger.info(f"   API Credentials: {'✅' if any(credentials.values()) else '❌'}")
    logger.info(f"   Client Connection: {'✅' if client_ok else '❌'}")
    logger.info(f"   Executor Initialization: {'✅' if executor_ok else '❌'}")
    
    if not client_ok:
        logger.info("\n💡 RECOMMENDATIONS:")
        logger.info("   1. For paper trading: Credentials are optional (bot will use mock data)")
        logger.info("   2. For live trading: Get API keys from Binance")
        logger.info("   3. For testnet: Get testnet API keys from https://testnet.binancefuture.com/")
        logger.info("   4. Set credentials in environment variables or config file")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()