#!/usr/bin/env python3
"""
Quick fix script to resolve Vault authentication and test Binance connection
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_vault_environment():
    """Set up Vault environment variables"""
    os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'
    
    # Try common dev tokens (use environment variable VAULT_TOKEN for production)
    dev_tokens = ['myroot', 'root']  # Common development tokens only
    
    for token in dev_tokens:
        os.environ['VAULT_TOKEN'] = token
        try:
            import hvac
            client = hvac.Client(url='http://127.0.0.1:8200', token=token)
            if client.is_authenticated():
                logger.info(f"✅ Successfully authenticated with Vault using token: {token[:10]}...")
                return True
        except Exception as e:
            logger.debug(f"Token {token[:10]}... failed: {e}")
            continue
    
    logger.warning("⚠️ Could not authenticate with Vault, will use fallback storage")
    return False

def test_binance_connection():
    """Test Binance connection with fallback credentials"""
    try:
        from trading.execution.binance_executor import BinanceExecutor
        from config.config import API_CONFIG
        
        # Try to initialize Binance in testnet mode
        executor = BinanceExecutor(
            api_key=API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY,
            api_secret=API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET,
            testnet=True
        )
        
        # Test connection
        account_info = executor.client.futures_account()
        logger.info("✅ Binance Futures Testnet connection successful")
        logger.info(f"Account balance: {account_info.get('totalWalletBalance', 'N/A')} USDT")
        return True
        
    except Exception as e:
        logger.error(f"❌ Binance connection failed: {e}")
        return False

def main():
    """Main fix function"""
    logger.info("🔧 Starting bot diagnostics and fixes...")
    
    # Fix Vault authentication
    vault_ok = fix_vault_environment()
    
    # Test Binance connection
    binance_ok = test_binance_connection()
    
    # Summary
    logger.info("="*50)
    logger.info("DIAGNOSIS SUMMARY:")
    logger.info(f"Vault Authentication: {'✅ OK' if vault_ok else '⚠️ FALLBACK'}")
    logger.info(f"Binance Connection: {'✅ OK' if binance_ok else '❌ FAILED'}")
    logger.info("="*50)
    
    if binance_ok:
        logger.info("🎉 Bot should be able to trade now!")
    else:
        logger.error("🚨 Trading will not work until Binance API credentials are fixed")
        logger.info("💡 Check your API keys in Vault or environment variables")

if __name__ == "__main__":
    main()