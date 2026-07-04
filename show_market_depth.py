#!/usr/bin/env python3
"""
Simple script to show market depth data clearly
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from utils.price_fetcher import PriceFetcher


def show_market_depth():
    """Display market depth in a simple, readable format"""

    print("=" * 60)
    print("🚀 ELVIS TRADING BOT - LIVE MARKET DEPTH")
    print("=" * 60)

    # Setup
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    logger = logging.getLogger(__name__)

    try:
        # Get live data
        print("📡 Fetching live order book from Binance...")
        price_fetcher = PriceFetcher(logger)
        order_book = price_fetcher.get_order_book("BTCUSDT", limit=10)

        if not order_book:
            print("❌ Failed to get order book data")
            return

        bids_data = order_book.get("bids", [])
        asks_data = order_book.get("asks", [])

        if not bids_data or not asks_data:
            print("❌ Order book data is empty")
            return

        print("✅ Live data received!")
        print()

        # Display the data
        print("📊 MARKET DEPTH - BTC/USDT")
        print("-" * 40)
        print("Price        | Quantity   | Volume")
        print("-" * 40)

        # Show asks (sell orders) - highest first
        print("🔴 ASKS (Sell Orders)")
        asks_display = asks_data[:5]
        asks_display.reverse()

        for ask in asks_display:
            price = float(ask[0])
            qty = float(ask[1])
            volume = price * qty
            print(f"${price:8.2f}  | {qty:8.4f}  | ${volume:8.0f}")

        # Calculate and show spread
        best_bid = float(bids_data[0][0])
        best_ask = float(asks_data[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100

        print("-" * 40)
        print(f"💰 SPREAD: ${spread:.2f} ({spread_pct:.3f}%)")
        print("-" * 40)

        # Show bids (buy orders)
        print("🟢 BIDS (Buy Orders)")
        for bid in bids_data[:5]:
            price = float(bid[0])
            qty = float(bid[1])
            volume = price * qty
            print(f"${price:8.2f}  | {qty:8.4f}  | ${volume:8.0f}")

        print("-" * 40)
        print(f"📈 Current BTC Price: ~${best_bid:,.2f}")
        print("=" * 60)

        print()
        print("✅ This is the EXACT same data that should appear")
        print("   in the console dashboard's Market Depth section!")
        print()
        print("🔍 To see it in the dashboard:")
        print("   1. Make sure terminal is at least 120 columns wide")
        print("   2. Run: python utils/console_dashboard.py")
        print("   3. Look for 'Market Depth' section")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    show_market_depth()
