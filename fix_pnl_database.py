#!/usr/bin/env python3
"""
Fix the PnL database by removing erroneous BNBBTC trades with incorrect prices
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

# Set environment
os.environ['VAULT_ENABLED'] = 'false'

def fix_pnl_database():
    """Fix the database by removing erroneous BNBBTC trades"""
    logger.info("🔧 Fixing PnL Database")
    logger.info("=" * 50)
    
    try:
        from utils.paper_trade_db import get_conn
        
        conn = get_conn()
        if not conn:
            logger.error("Cannot connect to database")
            return False
        
        c = conn.cursor()
        
        # First, let's see what we have
        logger.info("1. Analyzing current database state...")
        c.execute("SELECT COUNT(*) FROM trades")
        total_trades = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM trades WHERE symbol = 'BNBBTC'")
        bnbbtc_trades = c.fetchone()[0]
        
        c.execute("SELECT SUM(pnl) FROM trades WHERE side != 'TEST'")
        total_pnl = c.fetchone()[0] or 0
        
        logger.info(f"   Total trades: {total_trades}")
        logger.info(f"   BNBBTC trades: {bnbbtc_trades}")
        logger.info(f"   Total PnL: ${total_pnl:.2f}")
        
        # Check for problematic BNBBTC trades (price should be ~0.00665, not $113,000+)
        logger.info("2. Identifying problematic BNBBTC trades...")
        c.execute("""
            SELECT COUNT(*) FROM trades 
            WHERE symbol = 'BNBBTC' AND price > 1000
        """)
        bad_bnbbtc_trades = c.fetchone()[0]
        
        if bad_bnbbtc_trades > 0:
            logger.warning(f"   🚨 Found {bad_bnbbtc_trades} BNBBTC trades with incorrect high prices")
            
            # Show some examples
            c.execute("""
                SELECT id, price, quantity, pnl FROM trades 
                WHERE symbol = 'BNBBTC' AND price > 1000 
                LIMIT 5
            """)
            examples = c.fetchall()
            
            logger.info("   Examples of problematic trades:")
            for trade in examples:
                logger.info(f"     ID {trade[0]}: Price=${trade[1]:.2f}, Quantity={trade[2]:.6f}, PnL=${trade[3]:.2f}")
        
        # Option 1: Delete all BNBBTC trades with unrealistic prices
        logger.info("3. Removing problematic BNBBTC trades...")
        
        # Auto-approve removal of clearly erroneous trades
        if bad_bnbbtc_trades > 0:
            logger.info("   🔧 Auto-removing BNBBTC trades with unrealistic prices > $1000...")
            user_input = 'y'
        else:
            user_input = 'n'
        
        if user_input == 'y':
            # Get PnL from trades we're about to delete
            c.execute("SELECT SUM(pnl) FROM trades WHERE symbol = 'BNBBTC' AND price > 1000")
            deleted_pnl = c.fetchone()[0] or 0
            
            # Delete the problematic trades
            c.execute("DELETE FROM trades WHERE symbol = 'BNBBTC' AND price > 1000")
            deleted_count = c.rowcount
            
            logger.info(f"   ✅ Deleted {deleted_count} problematic BNBBTC trades")
            logger.info(f"   💰 Removed ${deleted_pnl:.2f} in erroneous PnL")
            
            # Commit the changes
            conn.commit()
            
            # Check new totals
            c.execute("SELECT SUM(pnl) FROM trades WHERE side != 'TEST'")
            new_total_pnl = c.fetchone()[0] or 0
            
            c.execute("SELECT COUNT(*) FROM trades")
            new_total_trades = c.fetchone()[0]
            
            logger.info(f"   📊 New totals: {new_total_trades} trades, ${new_total_pnl:.2f} PnL")
            logger.info(f"   📉 PnL reduction: ${total_pnl - new_total_pnl:.2f}")
            
        else:
            logger.info("   ⏭️ Skipping deletion - no changes made")
        
        # Option 2: Show realistic BNBBTC trades (if any)
        logger.info("4. Checking for realistic BNBBTC trades...")
        c.execute("""
            SELECT COUNT(*) FROM trades 
            WHERE symbol = 'BNBBTC' AND price < 0.1 AND price > 0.001
        """)
        good_bnbbtc_trades = c.fetchone()[0]
        
        if good_bnbbtc_trades > 0:
            logger.info(f"   ✅ Found {good_bnbbtc_trades} realistic BNBBTC trades (price between $0.001-$0.1)")
            
            c.execute("""
                SELECT id, price, quantity, pnl FROM trades 
                WHERE symbol = 'BNBBTC' AND price < 0.1 AND price > 0.001 
                LIMIT 3
            """)
            examples = c.fetchall()
            
            logger.info("   Examples of realistic trades:")
            for trade in examples:
                logger.info(f"     ID {trade[0]}: Price=${trade[1]:.6f}, Quantity={trade[2]:.6f}, PnL=${trade[3]:.2f}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error fixing database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def reset_bnb_balance():
    """Reset BNB balance to a reasonable amount"""
    logger.info("\n💰 Resetting BNB Balance")
    logger.info("-" * 50)
    
    try:
        from utils.paper_trade_db import get_conn
        
        conn = get_conn()
        if not conn:
            logger.error("Cannot connect to database")
            return False
        
        c = conn.cursor()
        
        # Check current balances
        try:
            c.execute("SELECT asset, balance FROM balances")
            balances = c.fetchall()
            
            logger.info("Current balances:")
            for asset, balance in balances:
                logger.info(f"   {asset}: {balance}")
            
            # Reset BNB to a reasonable amount (e.g., 1.0 BNB)
            logger.info("   🔧 Auto-resetting BNB balance to 1.0 BNB...")
            user_input = 'y'
            
            if user_input == 'y':
                c.execute("INSERT OR REPLACE INTO balances (asset, balance) VALUES ('BNB', 1.0)")
                conn.commit()
                logger.info("   ✅ BNB balance reset to 1.0 BNB")
            
        except Exception as e:
            logger.warning(f"Could not check/reset balances: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error resetting balance: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 PnL Database Fix Tool")
    logger.info("=" * 60)
    
    logger.info("This tool will help fix the unrealistic PnL calculation")
    logger.info("by removing erroneous BNBBTC trades with incorrect prices.")
    logger.info("")
    
    # Check 1: Fix database
    fix_success = fix_pnl_database()
    
    # Check 2: Reset balances
    if fix_success:
        reset_bnb_balance()
    
    logger.info("=" * 60)
    logger.info("📋 SUMMARY:")
    logger.info("The bot was executing BNBBTC trades with Bitcoin prices")
    logger.info("instead of BNB/BTC rates, causing artificial PnL inflation.")
    logger.info("")
    logger.info("✅ Root cause identified and fixed in main.py")
    logger.info("✅ BNBBTC conversion temporarily disabled")
    logger.info("✅ Mock price function updated with realistic values")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run the bot to verify PnL is now realistic")
    logger.info("2. Re-enable BNBBTC conversion with proper price validation")
    logger.info("3. Monitor future trades for correct price handling")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()