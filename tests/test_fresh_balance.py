#!/usr/bin/env python3
"""
Test Fresh Balance Initialization - Always $1000 USDT + $1000 BNB
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from trading.execution.binance_executor import BinanceExecutor


def test_fresh_balance():
    """Test that paper trading always starts with fresh $1000 + $1000 balance"""

    print("💰 Testing Fresh Balance Initialization")
    print("=" * 50)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Create executor
    executor = BinanceExecutor(logger=logger, is_testnet=True, use_futures=False)
    executor.initialize()

    print("\n🧪 Testing Balance - Should ALWAYS be $1000 USDT + $1000 BNB")
    balance = executor.get_balance()

    print(f"\n📊 Fresh Balance Results:")
    usdt_amount = balance.get("USDT", 0)
    bnb_amount = balance.get("BNB", 0)
    bnb_usd_value = bnb_amount * 600  # Assuming $600 per BNB

    print(f"   💵 USDT: ${usdt_amount:,.2f}")
    print(f"   💎 BNB: {bnb_amount:.6f} BNB (≈${bnb_usd_value:,.2f})")
    print(f"   💰 Total Value: ${usdt_amount + bnb_usd_value:,.2f}")

    # Verify correct amounts
    usdt_correct = abs(usdt_amount - 1000.0) < 0.01
    bnb_correct = abs(bnb_usd_value - 1000.0) < 10.0  # Allow small variance for price

    print(f"\n✅ Validation:")
    print(f"   USDT = $1000: {'✅ PASS' if usdt_correct else '❌ FAIL'}")
    print(f"   BNB ≈ $1000: {'✅ PASS' if bnb_correct else '❌ FAIL'}")

    if usdt_correct and bnb_correct:
        print(f"\n🎉 SUCCESS: Paper mode starts with exactly $1000 USDT + $1000 BNB!")
        print(f"🔄 This will be the balance EVERY time you start the bot fresh")
    else:
        print(f"\n❌ Issue detected with balance initialization")

    return usdt_correct and bnb_correct


if __name__ == "__main__":
    success = test_fresh_balance()

    if success:
        print(f"\n🚀 Ready! Run 'python main.py --mode paper --log-level INFO' to see:")
        print(
            f"   💰 PAPER TRADING: Fresh start - $1000 USDT + 1.67 BNB ($1000 equivalent)"
        )
        print(f"   📊 Clean slate every time you restart the bot")
    else:
        print(f"\n⚠️ Balance initialization needs adjustment")
