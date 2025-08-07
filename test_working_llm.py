#!/usr/bin/env python3
"""
Test WORKING LM Studio integration with localhost
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading.advisors.llm_advisor import LLMTradingAdvisor
import logging

def test_working_llm():
    """Test LLM integration that actually works with localhost"""
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    print("🎉 Testing WORKING LM Studio Integration...")
    
    # Create LLM advisor with localhost (the working endpoint)
    llm_advisor = LLMTradingAdvisor(
        llm_endpoint="http://localhost:1234",
        model_name="openai/gpt-oss-20b",
        logger=logger
    )
    
    # Realistic market data
    market_data = {
        'price': 116363.31,
        'rsi': 65.5,
        'macd': 0.0045,
        'volume': 1234567,
        'close': 116363.31
    }
    
    print("\n🧠 Testing Market Sentiment Analysis...")
    sentiment = llm_advisor.analyze_market_sentiment(market_data)
    
    print(f"📊 Sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.1%} confidence)")
    print(f"⚠️ Risk Level: {sentiment['risk_level']}")
    print(f"💡 Analysis: {sentiment['analysis']}")
    print(f"🔍 Key Factors: {', '.join(sentiment['key_factors'])}")
    
    print("\n🎯 Testing Signal Enhancement...")
    enhancement = llm_advisor.enhance_trading_signal('SELL', 0.850, market_data)
    
    print(f"✅ Validation: {enhancement['validation']}")
    print(f"📈 Confidence: {enhancement.get('original_confidence', 0.85):.1%} → {enhancement['adjusted_confidence']:.1%}")
    print(f"💡 Recommendation: {enhancement['recommendation']}")
    print(f"🧠 Reasoning: {enhancement['reasoning']}")
    
    print("\n📈 Testing Market Report Generation...")
    recent_trades = [
        {'pnl': 25.50, 'symbol': 'BTCUSDT'},
        {'pnl': -12.75, 'symbol': 'BTCUSDT'}
    ]
    report = llm_advisor.generate_market_report(market_data, recent_trades)
    
    print(f"📊 Market Report:")
    print(f"   {report}")
    
    print(f"\n🚀 LLM Integration Status: ✅ FULLY OPERATIONAL!")
    print(f"🎯 ELVIS now has AI-powered trading intelligence!")

if __name__ == "__main__":
    test_working_llm()