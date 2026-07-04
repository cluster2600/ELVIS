#!/usr/bin/env python3
"""Test live signal generation with real market data to debug HOLD-only issue."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import time

from binance.client import Client

from trading.strategies.ensemble_strategy import EnsembleStrategy
from utils.price_fetcher import PriceFetcher

# Setup logger
logger = logging.getLogger("live_test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# Get real market data
price_fetcher = PriceFetcher(logger=logger)
strategy = EnsembleStrategy(logger)

print("🔴 DEBUGGING LIVE SIGNAL GENERATION - FINDING WHY ONLY HOLD SIGNALS")
print("=" * 70)

# Test with live BTC data
symbol = "BTCUSDT"
try:
    # Get real market data
    print(f"\n📊 FETCHING LIVE MARKET DATA FOR {symbol}...")

    # Try to get price data
    current_data = price_fetcher.get_current_price_data(symbol)
    print(f"Current price data: {current_data}")

    if current_data:
        # Try to get more comprehensive data
        df = price_fetcher.get_historical_data(symbol, "1h", 100)
        if not df.empty:
            print(f"Historical data shape: {df.shape}")
            print(f"Latest row from historical data:")
            latest = df.iloc[-1]
            print(f"  Close: {latest.get('close', 'N/A')}")
            print(f"  RSI: {latest.get('rsi', 'N/A')}")
            print(f"  MACD: {latest.get('macd', 'N/A')}")
            print(f"  SMA_20: {latest.get('sma_20', 'N/A')}")
            print(f"  SMA_50: {latest.get('sma_50', 'N/A')}")

            # Create market data dict like main.py does
            market_data = {
                "close": float(latest["close"]),
                "price": float(latest["close"]),
                "rsi": latest.get("rsi", 50.0),
                "macd": latest.get("macd", 0.0),
                "macd_signal": latest.get("macd_signal", 0.0),
                "macd_histogram": latest.get("macd_histogram", 0.0),
                "sma_20": latest.get("sma_20", latest["close"]),
                "sma_50": latest.get("sma_50", latest["close"]),
                "bb_upper": latest.get("bb_upper", latest["close"] * 1.02),
                "bb_lower": latest.get("bb_lower", latest["close"] * 0.98),
                "bb_middle": latest.get("bb_middle", latest["close"]),
                "volume": latest.get("volume", 1000.0),
                "atr": latest.get("atr", latest["close"] * 0.01),
                "adx": latest.get("adx", 25.0),
            }

            print(f"\n🎯 TESTING SIGNAL GENERATION WITH LIVE DATA:")
            print(f"Market data being passed to strategy:")
            for key, value in market_data.items():
                print(f"  {key}: {value}")

            # Generate signal
            signal, confidence = strategy.generate_signal(symbol, market_data)

            print(f"\n🚨 LIVE SIGNAL RESULT:")
            print(f"Signal: {signal}")
            print(f"Confidence: {confidence:.3f}")

            # Check what main.py confidence threshold is
            print(f"\n🔍 CHECKING CONFIDENCE THRESHOLDS:")
            if confidence >= 0.1:
                print(f"✅ Signal passes 0.1 threshold: {confidence:.3f} >= 0.1")
            else:
                print(f"❌ Signal FAILS 0.1 threshold: {confidence:.3f} < 0.1")

            if confidence >= 0.05:
                print(f"✅ Signal passes 0.05 threshold: {confidence:.3f} >= 0.05")
            else:
                print(f"❌ Signal FAILS 0.05 threshold: {confidence:.3f} < 0.05")

            # Test with more extreme market conditions
            print(f"\n🧪 TESTING WITH EXTREME CONDITIONS TO FORCE SIGNALS:")

            # Test 1: Force very bearish conditions
            extreme_bearish = market_data.copy()
            extreme_bearish.update(
                {
                    "rsi": 80.0,  # Very overbought
                    "macd": -5.0,  # Very negative
                    "macd_signal": -2.0,  # MACD below signal
                    "sma_20": market_data["price"] * 1.05,  # Price well below SMA20
                    "sma_50": market_data["price"] * 1.10,  # Price well below SMA50
                }
            )

            print(f"\n📉 EXTREME BEARISH TEST:")
            signal_bear, conf_bear = strategy.generate_signal(symbol, extreme_bearish)
            print(
                f"Extreme bearish signal: {signal_bear} with {conf_bear:.3f} confidence"
            )

            # Test 2: Force very bullish conditions
            extreme_bullish = market_data.copy()
            extreme_bullish.update(
                {
                    "rsi": 20.0,  # Very oversold
                    "macd": 5.0,  # Very positive
                    "macd_signal": 2.0,  # MACD above signal
                    "sma_20": market_data["price"] * 0.95,  # Price well above SMA20
                    "sma_50": market_data["price"] * 0.90,  # Price well above SMA50
                }
            )

            print(f"\n📈 EXTREME BULLISH TEST:")
            signal_bull, conf_bull = strategy.generate_signal(symbol, extreme_bullish)
            print(
                f"Extreme bullish signal: {signal_bull} with {conf_bull:.3f} confidence"
            )

        else:
            print("❌ Failed to get historical data")
    else:
        print("❌ Failed to get current price data")

except Exception as e:
    print(f"❌ Error testing live signals: {e}")
    import traceback

    traceback.print_exc()

print(f"\n📝 DIAGNOSIS COMPLETE!")
