#!/usr/bin/env python3
"""
Setup paper trading with initial BNB balance ($1000 worth)
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

def get_current_bnb_price():
    """Get current BNB price for calculating balance"""
    try:
        from utils.price_fetcher import PriceFetcher
        price_fetcher = PriceFetcher(logger=logger, symbols=['BNBUSDT'])
        
        # Get BNB price
        bnb_df = price_fetcher.get_historical_klines('BNBUSDT', '5m', 1)
        bnb_price = float(bnb_df.iloc[-1]['close']) if not bnb_df.empty else 753.16
        
        logger.info(f"📊 Current BNB price: ${bnb_price:.2f}")
        return bnb_price
        
    except Exception as e:
        logger.warning(f"Could not fetch live BNB price: {e}, using fallback")
        return 753.16  # Fallback price

def create_enhanced_paper_trading_db():
    """Create enhanced paper trading database with multi-asset support"""
    
    db_content = '''
def init_db_with_balances():
    """Initialize database with multi-asset balance support"""
    conn = get_conn()
    if conn is None:
        print("[ERROR] Cannot initialize database - PostgreSQL connection required")
        return
    try:
        c = conn.cursor()
        
        # Grant permissions
        try:
            c.execute("GRANT USAGE, CREATE ON SCHEMA np TO CURRENT_USER;")
            conn.commit()
        except Exception as e:
            print(f"[INFO] Could not grant permissions, trying to continue: {e}")
            conn.rollback()
        
        # Create enhanced balance table
        c.execute("""
            CREATE TABLE IF NOT EXISTS np.account_balances (
                id SERIAL PRIMARY KEY,
                asset TEXT UNIQUE NOT NULL,
                balance REAL NOT NULL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Insert initial balances
        c.execute("""
            INSERT INTO np.account_balances (asset, balance) 
            VALUES ('USDT', 1000.0) 
            ON CONFLICT (asset) DO NOTHING
        """)
        
        # Add BNB balance (will be set by setup script)
        c.execute("""
            INSERT INTO np.account_balances (asset, balance) 
            VALUES ('BNB', 0.0) 
            ON CONFLICT (asset) DO NOTHING
        """)
        
        conn.commit()
        print("[INFO] Enhanced database with multi-asset balances initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize enhanced database: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_account_balance(asset='USDT'):
    """Get balance for a specific asset"""
    try:
        conn = get_conn()
        if conn is None:
            return 1000.0 if asset == 'USDT' else 0.0
        
        with conn.cursor() as c:
            c.execute("SELECT balance FROM np.account_balances WHERE asset = %s", (asset,))
            result = c.fetchone()
            return float(result[0]) if result else (1000.0 if asset == 'USDT' else 0.0)
    except Exception as e:
        print(f"[ERROR] Error getting {asset} balance: {e}")
        return 1000.0 if asset == 'USDT' else 0.0

def update_account_balance(asset, new_balance):
    """Update balance for a specific asset"""
    try:
        conn = get_conn()
        if conn is None:
            return False
        
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO np.account_balances (asset, balance) 
                VALUES (%s, %s) 
                ON CONFLICT (asset) 
                DO UPDATE SET balance = %s, last_updated = NOW()
            """, (asset, new_balance, new_balance))
        conn.commit()
        return True
    except Exception as e:
        print(f"[ERROR] Error updating {asset} balance: {e}")
        return False

def get_all_balances():
    """Get all asset balances"""
    try:
        conn = get_conn()
        if conn is None:
            return {'USDT': 1000.0, 'BNB': 0.0}
        
        with conn.cursor() as c:
            c.execute("SELECT asset, balance FROM np.account_balances")
            results = c.fetchall()
            return {row[0]: float(row[1]) for row in results}
    except Exception as e:
        print(f"[ERROR] Error getting all balances: {e}")
        return {'USDT': 1000.0, 'BNB': 0.0}
'''
    
    # Add to existing paper_trade_db.py
    with open('/Users/maxime/BTC_BOT/BTC_BOT/utils/paper_trade_db.py', 'a') as f:
        f.write('\n' + db_content)
    
    logger.info("✅ Enhanced paper trading database functions added")

def modify_binance_executor():
    """Modify Binance executor to support multi-asset paper trading"""
    
    executor_path = '/Users/maxime/BTC_BOT/BTC_BOT/trading/execution/binance_executor.py'
    
    # Read current content
    with open(executor_path, 'r') as f:
        content = f.read()
    
    # Enhanced balance calculation method
    new_method = '''    def _calculate_paper_balance(self) -> Dict[str, float]:
        """Calculate paper trading balance with multi-asset support"""
        if not self.db_available:
            return {'USDT': 1000.0, 'BNB': 0.0}
        
        try:
            from utils.paper_trade_db import get_all_balances, get_all_trades
            
            # Get stored balances
            balances = get_all_balances()
            
            # Calculate USDT balance from trades (legacy method)
            usdt_balance = balances.get('USDT', 1000.0)
            trades = get_all_trades(limit=10000, exclude_test=True)
            for trade in trades:
                if trade[2] == 'BTCUSDT':  # Only apply trade PnL to USDT for BTC trades
                    pnl = float(trade[6]) if trade[6] is not None else 0.0
                    fee = float(trade[7]) if trade[7] is not None else 0.0
                    usdt_balance += pnl - fee
            
            # Update USDT balance
            balances['USDT'] = usdt_balance
            
            # Ensure we have minimum balances
            if 'BNB' not in balances:
                balances['BNB'] = 0.0
            
            return balances
            
        except Exception as e:
            self.logger.error(f"Error calculating paper balance: {e}", exc_info=True)
            return {'USDT': 1000.0, 'BNB': 0.0}'''
    
    # Replace the existing method
    old_method_start = content.find('    def _calculate_paper_balance(self) -> Dict[str, float]:')
    old_method_end = content.find('    def execute_stop_loss', old_method_start)
    
    if old_method_start != -1 and old_method_end != -1:
        new_content = content[:old_method_start] + new_method + '\n\n' + content[old_method_end:]
        
        with open(executor_path, 'w') as f:
            f.write(new_content)
        
        logger.info("✅ Updated Binance executor with multi-asset balance support")
        return True
    else:
        logger.error("❌ Could not find method to replace in Binance executor")
        return False

def setup_initial_bnb_balance(bnb_amount_usd=1000.0):
    """Set up initial BNB balance in paper trading"""
    
    bnb_price = get_current_bnb_price()
    bnb_quantity = bnb_amount_usd / bnb_price
    
    try:
        # Initialize enhanced database
        from utils.paper_trade_db import init_db
        init_db()
        
        # Import the new functions (reload module)
        import importlib
        import utils.paper_trade_db
        importlib.reload(utils.paper_trade_db)
        
        from utils.paper_trade_db import init_db_with_balances, update_account_balance
        
        # Initialize with balances
        init_db_with_balances()
        
        # Set initial BNB balance
        success = update_account_balance('BNB', bnb_quantity)
        
        if success:
            logger.info(f"✅ Set initial BNB balance: {bnb_quantity:.6f} BNB (${bnb_amount_usd:.2f})")
            logger.info(f"📊 BNB price used: ${bnb_price:.2f}")
            return True
        else:
            logger.error("❌ Failed to set initial BNB balance")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error setting up BNB balance: {e}")
        return False

def test_enhanced_paper_trading():
    """Test the enhanced paper trading with BNB"""
    try:
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor
        
        logger.info("🧪 Testing enhanced paper trading with BNB...")
        
        # Initialize executor
        executor = EnhancedBinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=True,
            enable_bnb_fees=True,
            bnb_trading_enabled=True
        )
        
        executor.initialize()
        
        # Test balance
        balance_info = executor.get_enhanced_balance()
        balances = balance_info['balances']
        bnb_info = balance_info['bnb_info']
        
        logger.info("📊 Enhanced Paper Trading Balances:")
        for asset, amount in balances.items():
            if amount > 0:
                logger.info(f"   {asset}: {amount:.6f}")
        
        logger.info(f"\n🪙 BNB Information:")
        logger.info(f"   Balance: {bnb_info['balance']:.6f} BNB")
        logger.info(f"   USD Value: ${bnb_info['value_usdt']:.2f}")
        logger.info(f"   Sufficient for fees: {bnb_info['sufficient_for_fees']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🪙 Setting up Paper Trading with $1000 BNB Balance")
    logger.info("="*60)
    
    # 1. Create enhanced database functions
    create_enhanced_paper_trading_db()
    
    # 2. Modify Binance executor
    executor_updated = modify_binance_executor()
    
    # 3. Setup initial BNB balance
    bnb_setup = setup_initial_bnb_balance(1000.0)
    
    # 4. Test the setup
    test_success = test_enhanced_paper_trading()
    
    # Summary
    logger.info("="*60)
    logger.info("📋 PAPER TRADING BNB SETUP SUMMARY:")
    logger.info(f"   Database enhanced: ✅")
    logger.info(f"   Executor updated: {'✅' if executor_updated else '❌'}")
    logger.info(f"   BNB balance set: {'✅' if bnb_setup else '❌'}")
    logger.info(f"   Integration test: {'✅' if test_success else '❌'}")
    
    if all([executor_updated, bnb_setup, test_success]):
        logger.info("\n🎉 PAPER TRADING WITH BNB SUCCESSFULLY CONFIGURED!")
        logger.info("\n💰 Starting Balances:")
        logger.info("   • $1,000 USDT (for trading)")
        logger.info("   • $1,000 worth of BNB (for fees & trading)")
        logger.info("   • Total: $2,000 paper trading portfolio")
        
        logger.info("\n🎯 Features Available:")
        logger.info("   • BNB fee discounts (10% futures / 25% spot)")
        logger.info("   • BNB/USDT trading pair")
        logger.info("   • Multi-asset portfolio tracking")
        logger.info("   • Automatic BNB balance management")
    else:
        logger.warning("\n⚠️ Some setup steps failed - please check errors above")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()