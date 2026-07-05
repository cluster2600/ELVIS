#!/usr/bin/env python3
"""
Show Live LLM-Enhanced Trading Signals
Demonstrates exactly what ELVIS console dashboard shows with LLM integration
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from trading.advisors.llm_advisor import LLMTradingAdvisor
from trading.strategies.ensemble_strategy import EnsembleStrategy


def simulate_live_trading_with_llm():
    """Simulate what ELVIS console dashboard shows with LLM integration"""

    print("🎯 ELVIS Console Dashboard - LLM Enhanced Trading Signals")
    print("=" * 70)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Reduce noise

    # Create LLM advisor (exactly as ELVIS does)
    llm_advisor = LLMTradingAdvisor(
        llm_endpoint="http://localhost:1234",
        model_name="openai/gpt-oss-20b",
        logger=logger,
    )

    # Simulate current market conditions
    current_market_data = {
        "price": 116518.57,
        "rsi": 65.5,
        "macd": 0.0045,
        "volume": 1234567,
        "close": 116518.57,
    }

    print(f"📊 Current Market Data:")
    print(f"   BTC Price: ${current_market_data['price']:,.2f}")
    print(f"   RSI: {current_market_data['rsi']:.1f}")
    print(f"   MACD: {current_market_data['macd']:.4f}")
    print(f"   Volume: {current_market_data['volume']:,}")

    # Simulate trading signals (like ELVIS generates)
    signals = ["BUY", "SELL", "HOLD", "SELL"]
    confidences = [0.72, 0.85, 0.45, 0.91]

    for i, (signal, confidence) in enumerate(zip(signals, confidences), 1):
        print(f"\n{'─' * 70}")
        print(f"🔄 Trading Iteration #{i} - {time.strftime('%H:%M:%S')}")
        print(f"{'─' * 70}")

        # 1. Original Algorithm Signal
        print(f"🎯 Algorithm Signal: {signal} (Confidence: {confidence:.1%})")

        # 2. LLM Market Analysis
        print(f"🧠 LLM Market Analysis...")
        sentiment = llm_advisor.analyze_market_sentiment(current_market_data)

        print(
            f"   📊 Sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.0%} confidence)"
        )
        print(f"   ⚠️ Risk Level: {sentiment['risk_level']}")
        print(f"   💡 Analysis: {sentiment['analysis'][:60]}...")

        # 3. LLM Signal Enhancement
        if signal in ["BUY", "SELL"]:
            print(f"🎯 LLM Signal Enhancement...")
            enhancement = llm_advisor.enhance_trading_signal(
                signal, confidence, current_market_data
            )

            original_confidence = confidence
            enhanced_confidence = enhancement["adjusted_confidence"]
            validation = enhancement["validation"]

            print(f"   ✅ Validation: {validation}")
            print(
                f"   📈 Confidence: {original_confidence:.1%} → {enhanced_confidence:.1%}"
            )
            print(f"   💡 Recommendation: {enhancement['recommendation'][:50]}...")

            # Final decision
            if enhanced_confidence >= 0.80:
                decision = f"🚀 EXECUTE {signal}"
                color = "🟢" if signal == "BUY" else "🔴"
            elif enhanced_confidence >= 0.60:
                decision = f"⚠️ CAUTIOUS {signal}"
                color = "🟡"
            else:
                decision = "🛑 HOLD"
                color = "⚪"

            print(
                f"   {color} Final Decision: {decision} (AI-Enhanced: {enhanced_confidence:.1%})"
            )
        else:
            print(f"   ⚪ Final Decision: HOLD (Low original confidence)")

        # Simulate price change
        price_change = (i - 2) * 15.50  # Simulate market movement
        current_market_data["price"] += price_change
        current_market_data["rsi"] += (i - 2) * 2.1

        time.sleep(1)  # Simulate real-time

    print(f"\n{'═' * 70}")
    print(f"🎉 DEMONSTRATION COMPLETE")
    print(f"{'═' * 70}")
    print(f"This is exactly what happens in the ELVIS Console Dashboard:")
    print(f"   🧠 AI analyzes every market condition")
    print(f"   🎯 LLM validates and enhances every trading signal")
    print(f"   📊 Real-time sentiment analysis guides decisions")
    print(f"   🚀 Enhanced confidence scores improve trade execution")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    simulate_live_trading_with_llm()
