#!/usr/bin/env python3
"""
Test script to verify dashboard position display logic
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from datetime import datetime

import requests

from utils.paper_trade_db import get_open_positions


def test_position_display():
    """Test the position display logic that the dashboard uses"""
    print("🔍 TESTING DASHBOARD POSITION DISPLAY LOGIC")
    print("=" * 50)

    try:
        # Get positions exactly like the dashboard does
        live_positions = get_open_positions()

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"⏰ Time: {timestamp}")
        print(f"📊 Found {len(live_positions) if live_positions else 0} positions")

        if live_positions:
            print("\n📈 POSITIONS:")
            displayed_positions = 0

            for pos in live_positions[:5]:
                print(f"   Raw data: {pos}")

                if len(pos) >= 5:
                    symbol = pos[1]
                    side = pos[2]
                    entry_price = float(pos[3])
                    quantity = float(pos[4])

                    # Get current price for P&L
                    try:
                        response = requests.get(
                            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}",
                            timeout=5,
                        )
                        current_price = float(response.json()["price"])
                    except:
                        current_price = entry_price

                    # Calculate P&L
                    if side.upper() == "BUY":
                        pnl = (current_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - current_price) * quantity

                    # Format quantity for display
                    if abs(quantity) >= 1.0:
                        qty_display = f"{abs(quantity):.3f}"
                    elif abs(quantity) >= 0.1:
                        qty_display = f"{abs(quantity):.3f}"
                    else:
                        qty_display = f"{abs(quantity):.6f}"

                    # This is what the dashboard should show
                    position_text = (
                        f"{symbol} {side} {qty_display} @ ${entry_price:.0f}"
                    )
                    current_price_text = f"Live: ${current_price:.2f}"
                    pnl_text = f"P&L: ${pnl:+.2f}"

                    print(f"   Dashboard display: {position_text}")
                    print(f"                      {current_price_text}")
                    print(f"                      {pnl_text}")
                    print()

                    displayed_positions += 1
                else:
                    print(
                        f"   ❌ Position skipped: insufficient fields ({len(pos)} < 5)"
                    )

            print(f"✅ Total positions that should display: {displayed_positions}")
        else:
            print("❌ No positions found - Dashboard would show 'No active positions'")

    except Exception as e:
        print(f"❌ Error (Dashboard would show 'Position data error'): {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_position_display()
