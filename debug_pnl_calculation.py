#!/usr/bin/env python3
"""
Debug the PnL calculation issue - $150,210.22 seems unrealistic
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

def check_trade_database():
    """Check what's in the trade database"""
    logger.info("🔍 Checking Trade Database")
    logger.info("=" * 50)
    
    try:
        from utils.paper_trade_db import get_all_trades, get_all_balances, get_conn
        
        logger.info("1. Getting all trades...")
        trades = get_all_trades(limit=100)
        
        if not trades:
            logger.warning("   ⚠️ No trades found in database")
            return False
        
        logger.info(f"   Found {len(trades)} trades")
        
        # Analyze trades
        total_pnl = 0.0
        total_fees = 0.0
        
        logger.info("\n2. Trade Analysis:")
        logger.info("   ID | Timestamp           | Symbol   | Side | Price      | Quantity   | PnL        | Fee")
        logger.info("   " + "-" * 90)
        
        for i, trade in enumerate(trades[-20:]):  # Show last 20 trades
            trade_id, timestamp, symbol, side, price, quantity, pnl, fee = trade
            pnl_val = float(pnl) if pnl is not None else 0.0
            fee_val = float(fee) if fee is not None else 0.0
            
            total_pnl += pnl_val
            total_fees += fee_val
            
            logger.info(f"   {trade_id:2d} | {timestamp} | {symbol:8s} | {side:4s} | ${float(price):8.2f} | {float(quantity):8.6f} | ${pnl_val:8.2f} | ${fee_val:6.2f}")
        
        if len(trades) > 20:
            logger.info(f"   ... and {len(trades) - 20} more trades")
        
        logger.info("   " + "-" * 90)
        logger.info(f"   TOTALS (last 20): PnL=${total_pnl:.2f}, Fees=${total_fees:.2f}, Net=${total_pnl - total_fees:.2f}")
        
        # Calculate from all trades
        all_pnl = sum(float(trade[6]) if trade[6] is not None else 0.0 for trade in trades)
        all_fees = sum(float(trade[7]) if trade[7] is not None else 0.0 for trade in trades)
        
        logger.info(f"\n3. ALL TRADES SUMMARY:")
        logger.info(f"   Total PnL: ${all_pnl:.2f}")
        logger.info(f"   Total Fees: ${all_fees:.2f}")
        logger.info(f"   Net Realized PnL: ${all_pnl - all_fees:.2f}")
        
        # Check for suspicious trades
        logger.info("\n4. Suspicious Trades (large PnL):")
        suspicious_count = 0
        for trade in trades:
            pnl_val = float(trade[6]) if trade[6] is not None else 0.0
            if abs(pnl_val) > 1000:  # PnL > $1000
                suspicious_count += 1
                logger.warning(f"   🚨 Trade {trade[0]}: {trade[2]} {trade[3]} PnL=${pnl_val:.2f}")
        
        if suspicious_count == 0:
            logger.info("   ✅ No suspicious large PnL trades found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking trade database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_balance_calculation():
    """Check how balances are calculated"""
    logger.info("\n💰 Checking Balance Calculation")
    logger.info("-" * 50)
    
    try:
        from trading.execution.binance_executor import BinanceExecutor
        
        logger.info("1. Creating BinanceExecutor...")
        executor = BinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=False  # Paper trading
        )
        
        executor.initialize()
        
        logger.info("2. Getting calculated balance...")
        balance = executor.get_balance()
        
        logger.info(f"   Balance result: {balance}")
        
        # Check the calculation method
        logger.info("3. Analyzing balance calculation...")
        paper_balance = executor._calculate_paper_balance()
        
        logger.info(f"   Paper balance calculation: {paper_balance}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking balance calculation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_pnl_sources():
    """Check where the PnL numbers are coming from"""
    logger.info("\n📊 Checking PnL Sources")
    logger.info("-" * 50)
    
    try:
        # Check if there are any test trades with huge PnL
        from utils.paper_trade_db import get_conn
        
        conn = get_conn()
        if not conn:
            logger.error("Cannot connect to database")
            return False
        
        c = conn.cursor()
        
        # Look for trades with large PnL
        logger.info("1. Searching for trades with PnL > $1000...")
        c.execute("SELECT * FROM trades WHERE ABS(pnl) > 1000 ORDER BY pnl DESC")
        large_pnl_trades = c.fetchall()
        
        if large_pnl_trades:
            logger.warning(f"   🚨 Found {len(large_pnl_trades)} trades with large PnL:")
            for trade in large_pnl_trades:
                logger.warning(f"      ID {trade[0]}: {trade[2]} {trade[3]} PnL=${trade[6]:.2f}")
        else:
            logger.info("   ✅ No trades with PnL > $1000")
        
        # Check for TEST trades
        logger.info("2. Checking for TEST trades...")
        c.execute("SELECT COUNT(*) FROM trades WHERE side = 'TEST'")
        test_count = c.fetchone()[0]
        logger.info(f"   Found {test_count} TEST trades")
        
        # Check total PnL calculation
        logger.info("3. Calculating total PnL from database...")
        c.execute("SELECT SUM(pnl), SUM(fee), COUNT(*) FROM trades WHERE side != 'TEST'")
        result = c.fetchone()
        total_pnl, total_fees, trade_count = result
        
        total_pnl = total_pnl or 0.0
        total_fees = total_fees or 0.0
        
        logger.info(f"   Total trades (excluding TEST): {trade_count}")
        logger.info(f"   Sum of PnL: ${total_pnl:.2f}")
        logger.info(f"   Sum of fees: ${total_fees:.2f}")
        logger.info(f"   Net realized PnL: ${total_pnl - total_fees:.2f}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking PnL sources: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main debug function"""
    logger.info("🚀 PnL Calculation Debug")
    logger.info("=" * 60)
    
    # Check 1: Trade database
    check1_success = check_trade_database()
    
    # Check 2: Balance calculation
    check2_success = check_balance_calculation()
    
    # Check 3: PnL sources
    check3_success = check_pnl_sources()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 DEBUG SUMMARY:")
    logger.info(f"   Trade Database Check: {'✅' if check1_success else '❌'}")
    logger.info(f"   Balance Calculation Check: {'✅' if check2_success else '❌'}")
    logger.info(f"   PnL Sources Check: {'✅' if check3_success else '❌'}")
    
    logger.info("\n🔍 INVESTIGATION RESULTS:")
    logger.info("   Look for the root cause of the $150,210.22 realized PnL")
    logger.info("   Check for:")
    logger.info("   • Trades with unrealistic PnL values")
    logger.info("   • Calculation errors in paper trading")
    logger.info("   • Test data that shouldn't be included")
    logger.info("   • Currency conversion errors")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()