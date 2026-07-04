#!/usr/bin/env python3
"""
Force trading test - bypasses conservative signals to test trade execution
"""

import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from core.bootstrap import bootstrap_application
from core.di import container


def test_forced_trading():
    """Test by forcing BUY and SELL orders regardless of market conditions"""

    print("🚀 FORCE TRADING TEST - BYPASSING CONSERVATIVE SIGNALS")
    print("=" * 60)

    # Bootstrap the application
    bootstrapper = bootstrap_application("paper", "INFO")
    logger = container.get("logger")

    try:
        # Get components
        executor = container.get("executor")
        strategy = container.get("strategy")

        print(f"✅ Executor loaded: {type(executor).__name__}")
        print(f"✅ Strategy loaded: {type(strategy).__name__}")

        # Create mock market data
        mock_data = pd.DataFrame(
            {
                "open": [97000] * 10,
                "high": [97500] * 10,
                "low": [96500] * 10,
                "close": [97000] * 10,
                "volume": [1000] * 10,
                "rsi": [45] * 10,  # Slightly oversold
                "macd": [50] * 10,
                "signal_line": [40] * 10,  # MACD > signal = bullish
                "sma_20": [96800] * 10,  # Price above SMA = uptrend
                "atr": [100] * 10,
            }
        )

        current_price = 97000.0
        print(f"�� Mock market data created - Price: ${current_price}")

        # Test 1: Force BUY order
        print("\n🔥 TEST 1: FORCING BUY ORDER")
        print("-" * 30)

        available_balance = executor.get_account_balance()
        print(f"Available balance: ${available_balance}")

        # Calculate position size using strategy
        position_size = strategy.calculate_position_size(
            mock_data,
            current_price,
            available_balance,
            leverage=10,
            signal_confidence=0.9,
        )

        print(f"Calculated position size: {position_size:.6f} BTC")
        print(f"Position value: ${position_size * current_price:.2f}")

        # Execute BUY order
        print("Executing BUY order...")
        buy_result = executor.place_order(
            "BTCUSDT", "buy", position_size, current_price
        )

        if buy_result:
            print(
                f"✅ BUY order successful: {position_size:.6f} BTC at ${current_price}"
            )
            time.sleep(1)  # Wait for execution
        else:
            print("❌ BUY order failed")

        # Test 2: Force SELL order
        print("\n🔥 TEST 2: FORCING SELL ORDER")
        print("-" * 30)

        # Simulate price change
        current_price = 97200.0
        print(f"New price: ${current_price}")

        # Execute SELL order
        print("Executing SELL order...")
        sell_result = executor.place_order(
            "BTCUSDT", "sell", position_size, current_price
        )

        if sell_result:
            print(
                f"✅ SELL order successful: {position_size:.6f} BTC at ${current_price}"
            )
            time.sleep(1)  # Wait for execution
        else:
            print("❌ SELL order failed")

        # Test 3: Check trade history
        print("\n📊 TEST 3: CHECKING TRADE HISTORY")
        print("-" * 30)

        try:
            from utils.paper_trade_db import get_all_trades, get_trade_count

            trade_count = get_trade_count(exclude_test=True)
            print(f"Total trades in database: {trade_count}")

            if trade_count > 0:
                recent_trades = get_all_trades(limit=5, exclude_test=True)
                print("\nRecent trades:")
                for trade in recent_trades:
                    # trade: (id, timestamp, symbol, side, price, quantity, pnl, fee)
                    if len(trade) >= 7:
                        print(
                            f"  {trade[2]} {trade[3]} {trade[5]:.6f} @ ${trade[4]:.2f} | PnL: ${trade[6]:.2f}"
                        )
            else:
                print("⚠️  No trades found in database")

        except Exception as e:
            print(f"❌ Error checking trade history: {e}")

        # Test 4: Check balance after trades
        print("\n💰 TEST 4: FINAL BALANCE CHECK")
        print("-" * 30)

        final_balance = executor.get_account_balance()
        print(f"Final balance: ${final_balance}")
        print(f"Balance change: ${final_balance - available_balance:.2f}")

        if abs(final_balance - available_balance) > 0.01:
            print("✅ Balance changed - trades were executed successfully!")
        else:
            print("⚠️  No balance change detected")

        print("\n🎯 FORCE TRADING TEST COMPLETE")
        print("=" * 60)

        # Summary
        success_count = 0
        if buy_result:
            success_count += 1
        if sell_result:
            success_count += 1

        print(f"📈 Summary: {success_count}/2 forced trades executed")

        if success_count == 2:
            print("🎉 SUCCESS: Bot can execute trades when forced!")
            print("🔧 Issue: Conservative signals preventing normal trading")
            print("💡 Solution: Signals have been made more aggressive")
        else:
            print("❌ ISSUE: Trade execution is still failing")
            print("🔧 Check: Executor configuration and paper trading setup")

    except Exception as e:
        logger.error(f"Error in force trading test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        bootstrapper.cleanup()


if __name__ == "__main__":
    test_forced_trading()
