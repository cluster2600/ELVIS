#!/usr/bin/env python3
"""
Setup script to configure trading environment with proper credentials
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

def setup_vault_environment():
    """Set up Vault environment variables"""
    os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'
    # Disable Vault for now to use local storage
    os.environ['VAULT_ENABLED'] = 'false'
    logger.info("✅ Vault disabled, using local encrypted storage")

def setup_binance_testnet_credentials():
    """Set up Binance testnet credentials"""
    # These are example testnet credentials - replace with actual ones
    testnet_credentials = {
        'BINANCE_FUTURES_TESTNET_API_KEY': 'your_testnet_api_key_here',
        'BINANCE_FUTURES_TESTNET_API_SECRET': 'your_testnet_api_secret_here'
    }
    
    # Check if credentials are already set
    from config.config import API_CONFIG
    
    current_key = API_CONFIG.BINANCE_FUTURES_TESTNET_API_KEY
    current_secret = API_CONFIG.BINANCE_FUTURES_TESTNET_API_SECRET
    
    if current_key and current_key != 'your_futures_testnet_api_key_here':
        logger.info("✅ Binance testnet API key already configured")
        return True
    else:
        logger.warning("⚠️ Binance testnet API credentials not configured")
        logger.info("🔧 To fix this:")
        logger.info("1. Get testnet API credentials from https://testnet.binancefuture.com/")
        logger.info("2. Set environment variables:")
        logger.info("   export BINANCE_FUTURES_TESTNET_API_KEY='your_key'")
        logger.info("   export BINANCE_FUTURES_TESTNET_API_SECRET='your_secret'")
        logger.info("3. Or store them in Vault once authentication is fixed")
        return False

def check_database_connection():
    """Check database connection"""
    try:
        from utils.paper_trade_db import PaperTradeDB
        db = PaperTradeDB()
        
        # Test connection by getting account info
        account = db.get_account()
        logger.info(f"✅ Database connection OK - Account balance: ${account.get('balance', 0):.2f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🔧 Setting up trading environment...")
    
    # Setup Vault (disabled for now)
    setup_vault_environment()
    
    # Check Binance credentials
    binance_ok = setup_binance_testnet_credentials()
    
    # Check database
    db_ok = check_database_connection()
    
    # Summary
    logger.info("="*60)
    logger.info("ENVIRONMENT SETUP SUMMARY:")
    logger.info(f"Vault: ✅ DISABLED (using local storage)")
    logger.info(f"Binance API: {'✅ CONFIGURED' if binance_ok else '⚠️ NEEDS SETUP'}")
    logger.info(f"Database: {'✅ OK' if db_ok else '❌ FAILED'}")
    logger.info("="*60)
    
    if binance_ok and db_ok:
        logger.info("🎉 Environment ready for trading!")
        logger.info("💡 You can now run the bot with: python main.py")
    else:
        logger.warning("⚠️ Some components need attention before trading")

if __name__ == "__main__":
    main()