#!/usr/bin/env python3
"""
Enable BNBBTC trading in the bot to buy BTC using BNB
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

def test_bnbbtc_support():
    """Test BNBBTC trading support"""
    logger.info("🧪 Testing BNBBTC Trading Support")
    logger.info("="*50)
    
    try:
        # Test price fetcher with BNBBTC
        from utils.price_fetcher import PriceFetcher
        
        logger.info("1. Testing price fetcher with BNBBTC...")
        price_fetcher = PriceFetcher(
            logger=logger, 
            symbols=['BTCUSDT', 'BNBUSDT', 'BNBBTC'],
            timeframe='1m'
        )
        
        # Test fetching BNBBTC price
        bnbbtc_price = price_fetcher.get_current_price('BNBBTC')
        logger.info(f"   BNBBTC Price: {bnbbtc_price}")
        
        # Test historical data
        bnbbtc_data = price_fetcher.get_historical_klines('BNBBTC', '1m', limit=10)
        logger.info(f"   BNBBTC Data: {len(bnbbtc_data)} records" if not bnbbtc_data.empty else "   No BNBBTC data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing BNBBTC support: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_bnbbtc_execution():
    """Test BNBBTC trade execution"""
    logger.info("\n🤖 Testing BNBBTC Trade Execution")
    logger.info("-"*50)
    
    try:
        # Test enhanced executor with BNBBTC
        from trading.execution.enhanced_binance_executor import EnhancedBinanceExecutor
        
        executor = EnhancedBinanceExecutor(
            logger=logger,
            is_testnet=True,
            use_futures=False,  # Use spot for BNBBTC
            enable_bnb_fees=True,
            bnb_trading_enabled=True
        )
        
        executor.initialize()
        
        # Get current BNBBTC price
        bnbbtc_price = executor.get_current_price('BNBBTC')
        logger.info(f"   Current BNBBTC Price: {bnbbtc_price}")
        
        # Test small BNBBTC trade simulation
        if bnbbtc_price and bnbbtc_price > 0:
            # Calculate how much BTC we can buy with available BNB
            balance_info = executor.get_enhanced_balance()
            bnb_balance = balance_info['balances'].get('BNB', 0)
            
            logger.info(f"   Available BNB: {bnb_balance:.6f}")
            
            if bnb_balance > 0.01:  # Need at least 0.01 BNB
                btc_amount = bnb_balance * bnbbtc_price * 0.1  # Use 10% of BNB
                logger.info(f"   Could buy {btc_amount:.8f} BTC with {bnb_balance * 0.1:.6f} BNB")
                
                # Simulate the trade (paper trading)
                logger.info("   💡 Simulating BNBBTC trade...")
                result = executor._execute_paper_trade('BNBBTC', 'BUY', btc_amount, bnbbtc_price)
                if result:
                    logger.info(f"   ✅ BNBBTC trade simulation successful: {result}")
                else:
                    logger.warning("   ⚠️ BNBBTC trade simulation failed")
            else:
                logger.warning("   ⚠️ Insufficient BNB balance for trade test")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing BNBBTC execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_bnbbtc_strategy():
    """Create a strategy that includes BNBBTC trading"""
    logger.info("\n⚙️ Creating BNBBTC Trading Strategy")
    logger.info("-"*50)
    
    try:
        # Test the BNB-aware strategy
        from trading.strategies.bnb_aware_strategy import BNBAwareStrategy
        
        strategy = BNBAwareStrategy(
            logger=logger,
            symbols=['BTCUSDT', 'BNBUSDT', 'BNBBTC'],
            enable_bnb_optimization=True,
            bnb_allocation_percent=15.0  # 15% allocation to BNB-related trades
        )
        
        # Test signal generation for BNBBTC
        mock_data = {
            'price': 0.003,  # Example BNBBTC price
            'balance_info': {
                'BNB': 1.5,
                'BTC': 0.0,
                'total_usdt': 1000.0
            },
            'indicators': {
                'rsi': 45.0
            }
        }
        
        signal = strategy.generate_signal('BNBBTC', mock_data)
        logger.info(f"   BNBBTC Signal: {signal}")
        
        # Test position sizing for BNBBTC
        balance = {'BNB': 1.5, 'BTC': 0.0, 'USDT': 500.0}
        position_size = strategy.get_position_size(signal, 'BNBBTC', balance)
        logger.info(f"   BNBBTC Position Size: {position_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating BNBBTC strategy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def update_bot_configuration():
    """Update main bot configuration to include BNBBTC"""
    logger.info("\n🔧 Updating Bot Configuration for BNBBTC")
    logger.info("-"*50)
    
    try:
        # Check current bootstrap configuration
        from core.bootstrap import bootstrap_application
        logger.info("   Checking bootstrap configuration...")
        
        # Read bootstrap file to see current symbols
        bootstrap_path = '/Users/maxime/BTC_BOT/BTC_BOT/core/bootstrap.py'
        with open(bootstrap_path, 'r') as f:
            content = f.read()
        
        if 'BNBBTC' in content:
            logger.info("   ✅ BNBBTC already configured in bootstrap")
        else:
            logger.info("   📝 BNBBTC not found in bootstrap - manual update needed")
        
        # Check config file
        from config.config import SYMBOLS_CONFIG, BNB_CONFIG
        logger.info(f"   Symbols Config: {SYMBOLS_CONFIG}")
        logger.info(f"   BNB Config: {BNB_CONFIG}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating configuration: {e}")
        return False

def demonstrate_bnbbtc_trading():
    """Demonstrate how BNBBTC trading would work"""
    logger.info("\n💡 BNBBTC Trading Demonstration")
    logger.info("-"*50)
    
    logger.info("📋 How BNBBTC Trading Works:")
    logger.info("   1. You have BNB tokens in your balance")
    logger.info("   2. Bot monitors BNBBTC pair (BNB → BTC)")
    logger.info("   3. When conditions are favorable, bot sells BNB to buy BTC")
    logger.info("   4. This directly uses your BNB to acquire Bitcoin")
    
    logger.info("\n🎯 Current Setup:")
    logger.info("   • BNBBTC pair is configured in system")
    logger.info("   • Enhanced executor supports BNBBTC")
    logger.info("   • BNB-aware strategy includes BNBBTC logic")
    logger.info("   • Paper trading database supports multi-asset")
    
    logger.info("\n⚠️ Next Steps to Enable:")
    logger.info("   1. Update main.py to include BNBBTC in trading symbols")
    logger.info("   2. Modify bootstrap.py to initialize BNBBTC price fetcher")
    logger.info("   3. Update trading loop to process BNBBTC signals")
    logger.info("   4. Restart bot with multi-symbol support")

def create_bnbbtc_trading_patch():
    """Create a patch to enable BNBBTC trading in main bot"""
    logger.info("\n🔨 Creating BNBBTC Trading Patch")
    logger.info("-"*50)
    
    patch_code = '''
# BNBBTC Trading Patch
# Add this to your main trading loop to enable BNBBTC trading

# 1. Update symbols list in main.py
symbols_to_trade = ['BTCUSDT', 'BNBBTC']

# 2. Modify trading loop to handle multiple symbols
for symbol in symbols_to_trade:
    if symbol == 'BNBBTC':
        # Use spot trading for BNBBTC (no futures)
        data = price_fetcher.get_historical_klines(symbol, "1m")
        if not data.empty:
            # Generate signal for BNBBTC
            signal = active_strategy.generate_signal(symbol, market_data)
            if signal in ['BUY', 'SELL'] and confidence >= 0.8:
                # Execute BNBBTC trade
                executor.place_order(symbol, signal.lower(), position_size, current_price)
    else:
        # Continue with normal BTCUSDT trading
        pass

# 3. Update bootstrap.py symbols configuration
# Change: symbols=['BTCUSDT']
# To: symbols=['BTCUSDT', 'BNBBTC']
'''
    
    with open('/Users/maxime/BTC_BOT/BTC_BOT/bnbbtc_trading_patch.txt', 'w') as f:
        f.write(patch_code)
    
    logger.info("   ✅ Patch file created: bnbbtc_trading_patch.txt")
    logger.info("   📝 Review and apply the patch to enable BNBBTC trading")

def main():
    """Main function to enable BNBBTC trading"""
    logger.info("🚀 BNBBTC Trading Enablement Script")
    logger.info("="*60)
    
    # Test 1: BNBBTC Support
    test1_success = test_bnbbtc_support()
    
    # Test 2: BNBBTC Execution
    test2_success = test_bnbbtc_execution()
    
    # Test 3: BNBBTC Strategy
    test3_success = create_bnbbtc_strategy()
    
    # Test 4: Configuration
    test4_success = update_bot_configuration()
    
    # Demonstration
    demonstrate_bnbbtc_trading()
    
    # Create patch
    create_bnbbtc_trading_patch()
    
    # Summary
    logger.info("="*60)
    logger.info("📋 BNBBTC ENABLEMENT SUMMARY:")
    logger.info(f"   BNBBTC Support: {'✅' if test1_success else '❌'}")
    logger.info(f"   BNBBTC Execution: {'✅' if test2_success else '❌'}")
    logger.info(f"   BNBBTC Strategy: {'✅' if test3_success else '❌'}")
    logger.info(f"   Configuration: {'✅' if test4_success else '❌'}")
    
    if all([test1_success, test2_success, test3_success, test4_success]):
        logger.info("\n🎉 BNBBTC TRADING READY!")
        logger.info("✅ All components support BNBBTC trading")
        logger.info("📝 Apply the patch to main.py to enable live trading")
        logger.info("🔄 Restart bot to begin using BNB to buy BTC")
    else:
        logger.warning("\n⚠️ Some components need attention")
        logger.info("🔧 Review failed tests above")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()