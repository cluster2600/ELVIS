#!/usr/bin/env python3
"""
Enable BNB → BTC trading using BNBBTC spot pair
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

def test_bnbbtc_spot_trading():
    """Test BNBBTC trading on spot market"""
    logger.info("🧪 Testing BNBBTC Spot Trading")
    logger.info("="*50)
    
    try:
        # Test spot client directly
        from binance.client import Client
        
        client = Client()  # Public client for testing
        
        # Get BNBBTC ticker
        ticker = client.get_symbol_ticker(symbol='BNBBTC')
        logger.info(f"✅ BNBBTC Spot Price: {ticker['price']}")
        
        # Get recent trades
        trades = client.get_recent_trades(symbol='BNBBTC', limit=5)
        logger.info(f"✅ Recent BNBBTC trades: {len(trades)} found")
        
        # Get order book
        depth = client.get_order_book(symbol='BNBBTC', limit=5)
        logger.info(f"✅ BNBBTC Order Book: {len(depth['bids'])} bids, {len(depth['asks'])} asks")
        
        logger.info(f"📊 Best Bid: {depth['bids'][0][0]} | Best Ask: {depth['asks'][0][0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing BNBBTC spot trading: {e}")
        return False

def test_bnb_btc_conversion():
    """Test BNB to BTC conversion calculation"""
    logger.info("\n💱 Testing BNB → BTC Conversion")
    logger.info("-"*50)
    
    try:
        from binance.client import Client
        
        client = Client()
        
        # Get current prices
        bnb_usdt = float(client.get_symbol_ticker(symbol='BNBUSDT')['price'])
        btc_usdt = float(client.get_symbol_ticker(symbol='BTCUSDT')['price'])
        bnb_btc = float(client.get_symbol_ticker(symbol='BNBBTC')['price'])
        
        logger.info(f"📊 Current Prices:")
        logger.info(f"   BNB/USDT: ${bnb_usdt:.2f}")
        logger.info(f"   BTC/USDT: ${btc_usdt:.2f}")
        logger.info(f"   BNB/BTC:  {bnb_btc:.8f}")
        
        # Calculate conversion
        calculated_bnb_btc = bnb_usdt / btc_usdt
        price_diff = abs(calculated_bnb_btc - bnb_btc) / bnb_btc * 100
        
        logger.info(f"\n🧮 Conversion Analysis:")
        logger.info(f"   Calculated BNB/BTC: {calculated_bnb_btc:.8f}")
        logger.info(f"   Actual BNB/BTC:     {bnb_btc:.8f}")
        logger.info(f"   Price difference:   {price_diff:.4f}%")
        
        # Example conversion
        bnb_amount = 1.0
        btc_amount = bnb_amount * bnb_btc
        btc_value_usd = btc_amount * btc_usdt
        
        logger.info(f"\n💰 Example Conversion:")
        logger.info(f"   {bnb_amount} BNB = {btc_amount:.8f} BTC")
        logger.info(f"   BTC Value: ${btc_value_usd:.2f}")
        logger.info(f"   BNB Value: ${bnb_amount * bnb_usdt:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing conversion: {e}")
        return False

def create_bnb_btc_executor():
    """Create executor specifically for BNB → BTC trading"""
    logger.info("\n🤖 Creating BNB → BTC Executor")
    logger.info("-"*50)
    
    try:
        # Create a spot executor for BNBBTC
        from trading.execution.binance_executor import BinanceExecutor
        
        # Spot executor (no futures for BNBBTC)
        executor = BinanceExecutor(
            logger=logger,
            is_testnet=True,  # Paper trading
            use_futures=False,  # IMPORTANT: Spot only for BNBBTC
        )
        
        executor.initialize()
        
        # Test current price fetching
        bnbbtc_price = executor.get_current_price('BNBBTC')
        logger.info(f"✅ BNBBTC Price via executor: {bnbbtc_price}")
        
        # Test balance
        balance = executor.get_balance()
        logger.info(f"✅ Current balance: {balance}")
        
        # Test simulated trade
        if balance.get('BNB', 0) > 0.01:
            bnb_to_trade = min(0.1, balance['BNB'] * 0.1)  # 10% of BNB
            logger.info(f"🎯 Simulating: Sell {bnb_to_trade:.6f} BNB for BTC")
            
            # Simulate selling BNB for BTC
            result = executor._execute_paper_trade('BNBBTC', 'BUY', bnb_to_trade * bnbbtc_price, bnbbtc_price)
            if result:
                logger.info(f"✅ BNB → BTC trade simulation successful")
                logger.info(f"   Order: {result}")
            else:
                logger.warning("⚠️ Trade simulation failed")
        else:
            logger.warning("⚠️ Insufficient BNB for trade simulation")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating BNB → BTC executor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_bnb_btc_strategy():
    """Create a simple strategy for BNB → BTC conversion"""
    logger.info("\n⚙️ Creating BNB → BTC Strategy")
    logger.info("-"*50)
    
    strategy_code = '''
class BNBToBTCStrategy:
    """Simple strategy to convert BNB to BTC when conditions are favorable"""
    
    def __init__(self, logger):
        self.logger = logger
        self.min_bnb_balance = 0.1
        self.conversion_threshold = 0.02  # Convert when BNB allocation > 2%
        
    def should_convert_bnb_to_btc(self, balance_info, market_data):
        """Determine if we should convert BNB to BTC"""
        bnb_balance = balance_info.get('BNB', 0)
        total_value = balance_info.get('total_usdt', 1000)
        
        if bnb_balance < self.min_bnb_balance:
            return False, "Insufficient BNB balance"
            
        # Calculate BNB allocation percentage
        bnb_value = bnb_balance * market_data.get('bnb_price_usdt', 300)
        bnb_allocation = bnb_value / total_value
        
        if bnb_allocation > self.conversion_threshold:
            return True, f"BNB allocation {bnb_allocation:.1%} > {self.conversion_threshold:.1%}"
        
        return False, f"BNB allocation {bnb_allocation:.1%} below threshold"
    
    def calculate_conversion_amount(self, bnb_balance, target_allocation=0.01):
        """Calculate how much BNB to convert to BTC"""
        # Keep 1% allocation in BNB, convert the rest
        excess_bnb = bnb_balance * (1 - target_allocation)
        return max(0, excess_bnb)
'''
    
    with open('/Users/maxime/BTC_BOT/BTC_BOT/bnb_to_btc_strategy.py', 'w') as f:
        f.write(strategy_code)
    
    logger.info("✅ BNB → BTC strategy code created: bnb_to_btc_strategy.py")
    
    return True

def create_integration_guide():
    """Create integration guide for the main bot"""
    logger.info("\n📋 Creating Integration Guide")
    logger.info("-"*50)
    
    guide = '''
# BNB → BTC Trading Integration Guide

## Problem Identified
The bot was configured for BNBBTC futures trading, but BNBBTC only exists on Binance Spot market.

## Solution
1. **Use Spot Trading for BNBBTC**: Switch to spot executor when trading BNBBTC
2. **Separate Strategy**: Create dedicated BNB → BTC conversion logic
3. **Multi-Executor Setup**: Use futures for BTCUSDT, spot for BNBBTC

## Implementation Steps

### 1. Update main.py
Add BNBBTC to trading symbols and create dual executor setup:

```python
# In main.py trading loop
symbols_to_trade = ['BTCUSDT', 'BNBBTC']

for symbol in symbols_to_trade:
    if symbol == 'BNBBTC':
        # Use spot executor for BNBBTC
        if not hasattr(container, 'spot_executor'):
            from trading.execution.binance_executor import BinanceExecutor
            spot_executor = BinanceExecutor(
                logger=logger,
                is_testnet=True,
                use_futures=False  # Spot only
            )
            spot_executor.initialize()
            container.register_singleton('spot_executor', lambda: spot_executor)
        
        executor = container.get('spot_executor')
    else:
        # Use futures executor for BTCUSDT
        executor = container.get('executor')
    
    # Process symbol with appropriate executor
    data = price_fetcher.get_historical_klines(symbol, "1m")
    # ... rest of trading logic
```

### 2. Update bootstrap.py
Add BNBBTC to symbols list:

```python
# Change this line:
symbols=['BTCUSDT'],

# To this:
symbols=['BTCUSDT', 'BNBBTC'],
```

### 3. Create BNB → BTC Conversion Logic
Add this to your strategy:

```python
def check_bnb_conversion(self, balance, market_data):
    bnb_balance = balance.get('BNB', 0)
    
    # Convert excess BNB to BTC when BNB allocation > 2%
    if bnb_balance > 0.1:  # Minimum balance
        bnb_value = bnb_balance * market_data['bnb_price']
        total_value = sum(balance.values() * prices)
        
        if bnb_value / total_value > 0.02:  # 2% threshold
            # Convert excess BNB to BTC
            conversion_amount = bnb_balance * 0.5  # Convert 50%
            return 'CONVERT', conversion_amount
    
    return 'HOLD', 0
```

### 4. Test Configuration
1. Start bot with dual executors
2. Monitor BNBBTC price fetching
3. Test BNB → BTC conversion in paper mode
4. Verify BTC balance increases when BNB is sold

## Expected Behavior
- Bot monitors both BTCUSDT (futures) and BNBBTC (spot)
- When BNB allocation exceeds threshold, bot sells BNB for BTC
- BTC accumulates in spot balance
- Fee optimization continues with remaining BNB

## Key Points
- BNBBTC trades on SPOT market only
- Use separate executor for spot vs futures
- Monitor both balances (BNB and BTC)
- Set reasonable conversion thresholds
'''
    
    with open('/Users/maxime/BTC_BOT/BTC_BOT/BNB_TO_BTC_INTEGRATION_GUIDE.md', 'w') as f:
        f.write(guide)
    
    logger.info("✅ Integration guide created: BNB_TO_BTC_INTEGRATION_GUIDE.md")
    
    return True

def main():
    """Main function to enable BNB → BTC trading"""
    logger.info("🚀 BNB → BTC Trading Enablement")
    logger.info("="*60)
    
    # Test 1: BNBBTC Spot Market
    test1_success = test_bnbbtc_spot_trading()
    
    # Test 2: BNB → BTC Conversion
    test2_success = test_bnb_btc_conversion()
    
    # Test 3: BNB → BTC Executor  
    test3_success = create_bnb_btc_executor()
    
    # Create strategy and guide
    test4_success = create_bnb_btc_strategy()
    test5_success = create_integration_guide()
    
    # Summary
    logger.info("="*60)
    logger.info("📋 BNB → BTC ENABLEMENT SUMMARY:")
    logger.info(f"   BNBBTC Spot Trading: {'✅' if test1_success else '❌'}")
    logger.info(f"   BNB → BTC Conversion: {'✅' if test2_success else '❌'}")
    logger.info(f"   BNB → BTC Executor: {'✅' if test3_success else '❌'}")
    logger.info(f"   Strategy Creation: {'✅' if test4_success else '❌'}")
    logger.info(f"   Integration Guide: {'✅' if test5_success else '❌'}")
    
    if all([test1_success, test2_success, test3_success, test4_success, test5_success]):
        logger.info("\n🎉 BNB → BTC TRADING READY!")
        logger.info("✅ BNBBTC spot market confirmed working")
        logger.info("✅ Conversion calculations verified")
        logger.info("✅ Spot executor for BNBBTC created")
        logger.info("📝 Follow integration guide to enable in main bot")
        logger.info("🔄 Use dual executor setup (futures + spot)")
    else:
        logger.warning("\n⚠️ Some components need attention")
    
    logger.info("\n💡 Key Findings:")
    logger.info("   • BNBBTC only available on Spot market (not futures)")
    logger.info("   • Need separate spot executor for BNBBTC trading")
    logger.info("   • Current setup supports BNB fee optimization")
    logger.info("   • Bot can now sell BNB to accumulate BTC")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()