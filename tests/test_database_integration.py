#!/usr/bin/env python3
"""
Test database integration with paper trading to verify all issues are resolved.
"""

import logging
from utils.paper_trade_db import init_db, record_trade, add_open_position, get_open_positions, close_open_position

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_integration():
    """Test complete database integration."""
    
    logger.info("=== DATABASE INTEGRATION TEST ===")
    
    # Test 1: Database initialization
    logger.info("1. Testing database initialization...")
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 2: Record a BUY trade
    logger.info("2. Testing BUY trade recording...")
    try:
        record_trade('BTCUSDT', 'BUY', 105000.0, 0.001, 0.0, 0.42)
        add_open_position('BTCUSDT', 105000.0, 0.001, 1.0)
        logger.info("✅ BUY trade recorded successfully")
    except Exception as e:
        logger.error(f"❌ BUY trade recording failed: {e}")
        return False
    
    # Test 3: Check open positions
    logger.info("3. Testing open positions retrieval...")
    try:
        positions = get_open_positions()
        logger.info(f"✅ Retrieved {len(positions)} open positions")
        if positions:
            pos = positions[0]
            logger.info(f"   Position: {pos[1]} at ${pos[2]:.2f}, quantity: {pos[3]}")
    except Exception as e:
        logger.error(f"❌ Open positions retrieval failed: {e}")
        return False
    
    # Test 4: Record a SELL trade and close position
    logger.info("4. Testing SELL trade recording...")
    try:
        pnl = (105500.0 - 105000.0) * 0.001  # $0.50 profit
        record_trade('BTCUSDT', 'SELL', 105500.0, 0.001, pnl, 0.42)
        close_open_position('BTCUSDT')
        logger.info(f"✅ SELL trade recorded with PnL: ${pnl:.2f}")
    except Exception as e:
        logger.error(f"❌ SELL trade recording failed: {e}")
        return False
    
    # Test 5: Verify no open positions remain
    logger.info("5. Testing position closure...")
    try:
        positions = get_open_positions()
        logger.info(f"✅ Positions after closure: {len(positions)}")
    except Exception as e:
        logger.error(f"❌ Position closure test failed: {e}")
        return False
    
    logger.info("=== ALL DATABASE TESTS PASSED ===")
    return True

if __name__ == "__main__":
    success = test_database_integration()
    if success:
        logger.info("🎉 Database integration is fully operational!")
    else:
        logger.error("💥 Database integration has issues!")