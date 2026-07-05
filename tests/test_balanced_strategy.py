#!/usr/bin/env python3
"""
Test script to verify balanced strategy configuration and emergency fixes.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load .env file
from dotenv import load_dotenv

load_dotenv()


def test_balanced_strategy():
    """Test balanced strategy configuration"""
    print("=== TESTING BALANCED STRATEGY CONFIGURATION ===")

    # Test environment variable
    strategy_mode = os.getenv("STRATEGY_MODE", "ensemble")
    print(f"STRATEGY_MODE: {strategy_mode}")

    # Test bootstrap configuration
    try:
        from core.bootstrap import bootstrap
        from core.di.container import container

        # Initialize bootstrap
        bootstrap.initialize(mode="paper")

        # Get strategy
        strategy = container.get("strategy")
        print(f"Strategy type: {type(strategy).__name__}")

        # Test balanced strategy specific features
        if hasattr(strategy, "target_profit_per_trade"):
            print(f"Target profit per trade: ${strategy.target_profit_per_trade}")
            print(f"Daily trade target: {strategy.daily_trade_target}")
            print(f"Min position hold time: {strategy.min_position_hold_time}s")
            print(f"Adaptation interval: {strategy.adaptation_interval}s")

        # Test signal generation
        if hasattr(strategy, "generate_signal"):
            print("✅ generate_signal method available")

            # Test signal generation
            market_data = {
                "price": 97000.0,
                "close": 97000.0,
                "rsi": 55.0,
                "macd": 0.001,
                "volume": 1000.0,
            }

            signal, confidence = strategy.generate_signal("BTCUSDT", market_data)
            print(f"Test signal: {signal} (confidence: {confidence:.3f})")
        else:
            print("❌ generate_signal method not available")

    except Exception as e:
        print(f"❌ Error testing strategy: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== EXPECTED CONFIGURATION ===")
    print("- Strategy mode: balanced")
    print("- Target profit: $1.00 per trade")
    print("- Daily trades: 50 max")
    print("- Hold time: 600 seconds (10 minutes)")
    print("- Adaptation: 3600 seconds (1 hour)")
    print("- Emergency fixes: Active")


if __name__ == "__main__":
    test_balanced_strategy()
