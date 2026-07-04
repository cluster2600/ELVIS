#!/usr/bin/env python3
"""
Check Paper Trading Balances

Quick script to check current paper trading balances.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.paper_trade_db import get_all_balances


def main():
    """Check current paper trading balances."""
    print("💰 Current Paper Trading Balances")
    print("=" * 40)

    try:
        balances = get_all_balances()

        if balances:
            total_value = 0.0
            for asset, balance in balances.items():
                print(f"   {asset}: ${balance:,.2f}")
                total_value += balance

            print("-" * 40)
            print(f"   Total Portfolio Value: ${total_value:,.2f}")

            # Check if this is the initial state
            if balances.get("USDT", 0) == 1000.0 and balances.get("BNB", 0) == 1000.0:
                print("\n✅ Paper trading is at initial state")
                print("🚀 Ready to start trading!")
            else:
                print("\n📈 Paper trading is active with modified balances")
        else:
            print("❌ No balance data found")
            print("Run: python reset_paper_trading.py to initialize")

    except Exception as e:
        print(f"❌ Error checking balances: {e}")


if __name__ == "__main__":
    main()
