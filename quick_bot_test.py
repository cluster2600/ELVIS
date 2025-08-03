#!/usr/bin/env python3
"""
Quick test to verify bot components are working
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

# Set environment for testing
os.environ['VAULT_ENABLED'] = 'false'
os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'

def test_bot_components():
    """Test key bot components"""
    results = {}
    
    # Test 1: Configuration loading
    try:
        from config.config import TRADING_CONFIG, API_CONFIG
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"Trading mode: {TRADING_CONFIG['DEFAULT_MODE']}")
        results['config'] = True
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        results['config'] = False
    
    # Test 2: Database connection
    try:
        from utils.paper_trade_db import get_conn, get_open_positions
        conn = get_conn()
        positions = get_open_positions()
        conn.close()
        logger.info(f"✅ Database OK - {len(positions)} open positions")
        results['database'] = True
    except Exception as e:
        logger.error(f"❌ Database failed: {e}")
        results['database'] = False
    
    # Test 3: Price fetcher
    try:
        from utils.price_fetcher import get_realtime_data
        data = get_realtime_data('BTCUSDT', interval='5m', limit=10)
        if data is not None and len(data) > 0:
            logger.info(f"✅ Price fetcher OK - Got {len(data)} candles")
            logger.info(f"Latest BTC price: ${data.iloc[-1]['close']:.2f}")
            results['price_fetcher'] = True
        else:
            logger.warning("⚠️ Price fetcher returned no data")
            results['price_fetcher'] = False
    except Exception as e:
        logger.error(f"❌ Price fetcher failed: {e}")
        results['price_fetcher'] = False
    
    # Test 4: Trading strategies
    try:
        from trading.strategies.ensemble_strategy import EnsembleStrategy
        strategy = EnsembleStrategy()
        logger.info("✅ Ensemble strategy loaded")
        results['strategy'] = True
    except Exception as e:
        logger.error(f"❌ Strategy loading failed: {e}")
        results['strategy'] = False
    
    return results

def main():
    """Main test function"""
    logger.info("🔧 Testing bot components...")
    
    results = test_bot_components()
    
    # Summary
    logger.info("="*60)
    logger.info("BOT COMPONENT TEST SUMMARY:")
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"{component.capitalize()}: {status_icon}")
    
    all_good = all(results.values())
    if all_good:
        logger.info("🎉 All components working! Bot should be ready to trade.")
    else:
        logger.warning("⚠️ Some components need fixing before trading")
    logger.info("="*60)

if __name__ == "__main__":
    main()