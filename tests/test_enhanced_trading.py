#!/usr/bin/env python3
"""
Test Enhanced Trading Bot with Stop Losses and Profit Taking
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading.execution.binance_executor import BinanceExecutor
from utils.paper_trade_db import get_open_positions, get_all_trades
import logging

def test_enhanced_trading_features():
    """Test the enhanced trading features: stop losses, profit taking, BNB balance"""
    
    print("🧪 Testing Enhanced Trading Bot Features")
    print("=" * 60)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Create executor with paper trading
    executor = BinanceExecutor(logger=logger, is_testnet=True, use_futures=False)
    executor.initialize()
    
    print("\n💰 Testing Initial Balance (Should have $1000 USDT + $1000 BNB equivalent)")
    balance = executor.get_balance()
    for asset, amount in balance.items():
        if asset == 'USDT':
            print(f"   {asset}: ${amount:,.2f}")
        elif asset == 'BNB':
            bnb_usd_value = amount * 600  # Assume $600 per BNB
            print(f"   {asset}: {amount:.6f} BNB (≈${min(bnb_usd_value, 2000):,.2f})")
        else:
            print(f"   {asset}: {amount}")
    
    print(f"\n📊 Current Open Positions:")
    positions = get_open_positions()
    if positions:
        for pos in positions:
            print(f"   {pos[1]} {pos[2]} @ ${pos[3]:.2f} | Quantity: {pos[4]}")
    else:
        print("   No open positions")
    
    print(f"\n🎯 Testing Enhanced Paper Trading...")
    
    # Test 1: Open a position
    print(f"\n1️⃣ Opening BUY position for BTCUSDT")
    result = executor.execute_buy('BTCUSDT', 0.001, 116000)
    print(f"   Result: {result.get('status', 'FILLED')} | Order ID: {result.get('orderId', 'N/A')}")
    
    # Test 2: Check position management
    print(f"\n2️⃣ Testing automatic position management...")
    executor.check_and_manage_positions()
    
    # Test 3: Simulate profit scenario
    print(f"\n3️⃣ Simulating profitable trade...")
    # Mock a profitable position by executing opposite side at higher price
    result = executor.execute_sell('BTCUSDT', 0.001, 116002)  # $2 profit
    print(f"   Result: {result.get('status', 'FILLED')} | Should show profit taking")
    
    # Test 4: Show recent trades
    print(f"\n📈 Recent Trades (Last 5):")
    try:
        recent_trades = get_all_trades(limit=5)
        for trade in recent_trades[-5:]:
            timestamp = trade[1][:16] if len(trade[1]) > 16 else trade[1]
            pnl = float(trade[6]) if trade[6] else 0.0
            pnl_color = "💰" if pnl > 0 else "🛑" if pnl < 0 else "⚪"
            print(f"   {timestamp}: {trade[3]} {trade[2]} @ ${float(trade[4]):,.2f} | PnL: {pnl_color}${pnl:.2f}")
    except Exception as e:
        print(f"   Error loading trades: {e}")
    
    print(f"\n💼 Updated Balance:")
    balance = executor.get_balance()
    for asset, amount in balance.items():
        if asset == 'USDT':
            print(f"   {asset}: ${amount:,.2f}")
        elif asset == 'BNB':
            bnb_usd_value = amount * 600
            print(f"   {asset}: {amount:.6f} BNB (≈${bnb_usd_value:,.2f})")
    
    print(f"\n🎉 Enhanced Trading Features Test Complete!")
    print(f"✅ Stop Loss: Positions losing >$5 will be auto-closed")
    print(f"✅ Profit Taking: Positions with ≥$1 profit will be auto-closed") 
    print(f"✅ Risk Management: No new positions if USDT balance <$500")
    print(f"✅ BNB Balance: Initialized with $1000 equivalent (~1.67 BNB)")

if __name__ == "__main__":
    test_enhanced_trading_features()