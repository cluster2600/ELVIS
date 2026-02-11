#!/usr/bin/env python3
"""
Test database connection and basic functionality
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

# Disable Vault for testing
os.environ['VAULT_ENABLED'] = 'false'

def test_database():
    """Test database connection and functionality"""
    try:
        from utils.paper_trade_db import get_conn, get_all_trades, get_open_positions
        
        # Test connection
        conn = get_conn()
        logger.info("✅ Database connection successful")
        
        # Test basic queries
        trades = get_all_trades(limit=5)
        logger.info(f"✅ Retrieved {len(trades)} recent trades")
        
        positions = get_open_positions()
        logger.info(f"✅ Retrieved {len(positions)} open positions")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🔧 Testing database connection...")
    
    db_ok = test_database()
    
    logger.info("="*50)
    if db_ok:
        logger.info("✅ Database is working properly!")
    else:
        logger.error("❌ Database needs attention")
        logger.info("💡 Make sure PostgreSQL is running and credentials are correct")

if __name__ == "__main__":
    main()