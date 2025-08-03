#!/usr/bin/env python3
"""
Comprehensive test of all bot components with proper initialization
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment for testing
os.environ['VAULT_ENABLED'] = 'false'
os.environ['VAULT_ADDR'] = 'http://127.0.0.1:8200'

def test_bot_components():
    """Test key bot components with proper initialization"""
    results = {}
    
    # Test 1: Configuration loading
    try:
        from config.config import TRADING_CONFIG, API_CONFIG
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"Trading mode: {TRADING_CONFIG['DEFAULT_MODE']}")
        logger.info(f"Symbol: {TRADING_CONFIG['SYMBOL']}")
        logger.info(f"Max position size: {TRADING_CONFIG['MAX_POSITION_SIZE']}")
        results['config'] = True
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        results['config'] = False
    
    # Test 2: Database connection
    try:
        from utils.paper_trade_db import get_conn, get_open_positions, get_all_trades
        conn = get_conn()
        positions = get_open_positions()
        trades = get_all_trades(limit=5)
        conn.close()
        logger.info(f"✅ Database OK - {len(positions)} open positions, {len(trades)} recent trades")
        results['database'] = True
    except Exception as e:
        logger.error(f"❌ Database failed: {e}")
        results['database'] = False
    
    # Test 3: Price fetcher (using correct method)
    try:
        from utils.price_fetcher import PriceFetcher
        price_fetcher = PriceFetcher(logger=logger, symbols=['BTCUSDT'])
        # Use the correct method that returns a DataFrame
        df = price_fetcher.get_historical_klines('BTCUSDT', '5m', 10)
        if df is not None and not df.empty:
            logger.info(f"✅ Price fetcher OK - Got {len(df)} candles")
            latest_price = df.iloc[-1]['close']
            logger.info(f"Latest BTC price: ${latest_price:.2f}")
            logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            results['price_fetcher'] = True
        else:
            logger.warning("⚠️ Price fetcher returned empty DataFrame")
            results['price_fetcher'] = False
    except Exception as e:
        logger.error(f"❌ Price fetcher failed: {e}")
        results['price_fetcher'] = False
    
    # Test 4: Trading strategies (properly initialized)
    try:
        from trading.strategies.ensemble_strategy import EnsembleStrategy
        strategy = EnsembleStrategy(logger=logger, symbols=['BTCUSDT'])
        logger.info("✅ Ensemble strategy loaded successfully")
        results['strategy'] = True
    except Exception as e:
        logger.error(f"❌ Strategy loading failed: {e}")
        results['strategy'] = False
    
    # Test 5: Check if models exist
    try:
        model_paths = [
            "models/model_rf_tf",
            "models/NNModel.mlpackage", 
            "training/models/trade_learned_model.pkl"
        ]
        existing_models = []
        for path in model_paths:
            if os.path.exists(path):
                existing_models.append(path)
        
        logger.info(f"✅ Found {len(existing_models)}/{len(model_paths)} model files")
        for model in existing_models:
            logger.info(f"  - {model}")
        results['models'] = len(existing_models) > 0
    except Exception as e:
        logger.error(f"❌ Model check failed: {e}")
        results['models'] = False
    
    return results

def main():
    """Main test function"""
    logger.info("🔧 Running comprehensive bot component test...")
    
    results = test_bot_components()
    
    # Summary
    logger.info("="*70)
    logger.info("🤖 BOT COMPONENT TEST SUMMARY:")
    logger.info("-" * 70)
    
    component_names = {
        'config': 'Configuration Loading',
        'database': 'Database Connection', 
        'price_fetcher': 'Price Data Fetcher',
        'strategy': 'Trading Strategy',
        'models': 'ML Models'
    }
    
    for component, status in results.items():
        status_icon = "✅ PASS" if status else "❌ FAIL"
        name = component_names.get(component, component.capitalize())
        logger.info(f"{name:.<25} {status_icon}")
    
    logger.info("-" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        logger.info(f"🎉 ALL TESTS PASSED ({passed}/{total})! Bot is ready to trade.")
        logger.info("💡 You can start the bot with: python main.py")
    elif passed >= 3:  # Core components working
        logger.info(f"⚠️ MOSTLY WORKING ({passed}/{total}) - Bot should function with limitations")
        logger.info("💡 Some advanced features may not work, but basic trading should be OK")
    else:
        logger.error(f"🚨 CRITICAL ISSUES ({passed}/{total}) - Bot needs fixes before trading")
        logger.info("💡 Please fix the failing components before running the bot")
    
    logger.info("="*70)

if __name__ == "__main__":
    main()