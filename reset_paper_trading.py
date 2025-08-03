#!/usr/bin/env python3
"""
Reset Paper Trading Script

This script resets the paper trading environment to the starting state:
- $1000 USDT balance
- $1000 BNB balance  
- Clears all trade history
- Clears all open positions
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.paper_trade_db import initialize_paper_trading, get_all_balances

def main():
    """Reset paper trading to initial state."""
    print("🔄 Resetting Paper Trading Environment")
    print("=" * 50)
    
    try:
        # Initialize paper trading with clean state
        success = initialize_paper_trading()
        
        if success:
            # Verify the balances
            balances = get_all_balances()
            
            print("\n✅ Paper Trading Reset Complete!")
            print("\n📊 Starting Balances:")
            for asset, balance in balances.items():
                print(f"   {asset}: ${balance:,.2f}")
            
            print("\n🎯 Ready for paper trading with:")
            print("   • $1000 USDT for cryptocurrency trading")
            print("   • $1000 BNB for Binance ecosystem trading")
            print("   • Clean trade history")
            print("   • No open positions")
            print("\n🚀 You can now start the trading bot in paper mode!")
            
        else:
            print("\n❌ Failed to reset paper trading environment")
            print("Check database connection and permissions.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error resetting paper trading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()