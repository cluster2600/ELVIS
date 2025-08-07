"""
LLM Trading Advisor - Provides intelligent market analysis using local LLM
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class LLMTradingAdvisor:
    """
    Trading advisor powered by local LLM for market analysis and insights.
    """
    
    def __init__(self, llm_endpoint: str = "http://192.168.1.171:1234", model_name: str = "openai/gpt-oss-20b", logger: Optional[logging.Logger] = None):
        """
        Initialize LLM Trading Advisor.
        
        Args:
            llm_endpoint: Base URL of the LLM server
            model_name: Name of the LLM model to use
            logger: Logger instance
        """
        self.llm_endpoint = llm_endpoint
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        self.chat_endpoint = f"{llm_endpoint}/v1/chat/completions"
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to LLM server."""
        try:
            response = self._make_llm_request(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            if response:
                self.logger.info("🤖 LLM connection established successfully")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️ LLM connection failed: {e}")
        return False
    
    def _make_llm_request(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 300) -> Optional[str]:
        """
        Make a request to the LLM server with multiple fallback approaches.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text or None if failed
        """
        # Try multiple request approaches for LM Studio compatibility
        approaches = [
            # Approach 1: Standard with Authorization header
            {
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer lm-studio",
                    "Accept": "application/json"
                },
                "name": "LM Studio format"
            },
            # Approach 2: Simple format
            {
                "headers": {
                    "Content-Type": "application/json"
                },
                "name": "Simple format"
            },
            # Approach 3: With User-Agent
            {
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "ELVIS-Trading-Bot/1.0",
                    "Accept": "application/json"
                },
                "name": "With User-Agent"
            }
        ]
        
        for approach in approaches:
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                
                response = requests.post(
                    self.chat_endpoint,
                    headers=approach["headers"],
                    json=payload,  # Use json parameter instead of data
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        self.logger.debug(f"✅ LLM request successful with {approach['name']}")
                        return content
                else:
                    self.logger.debug(f"❌ {approach['name']} failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.debug(f"❌ {approach['name']} error: {e}")
                continue
        
        # All approaches failed
        self.logger.error(f"All LLM request approaches failed for endpoint: {self.chat_endpoint}")
        return None
    
    def analyze_market_sentiment(self, market_data: Dict[str, Any], recent_news: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze current market sentiment using LLM.
        
        Args:
            market_data: Current market data including price, indicators, etc.
            recent_news: Optional list of recent news headlines
            
        Returns:
            Dictionary with sentiment analysis results
        """
        current_price = market_data.get('price', market_data.get('close', 0))
        rsi = market_data.get('rsi', 50)
        macd = market_data.get('macd', 0)
        volume = market_data.get('volume', 0)
        
        # Build context for LLM
        system_prompt = """You are an expert cryptocurrency trading analyst. Analyze the provided market data and give a concise trading sentiment assessment. 

Focus on:
1. Overall market sentiment (Bullish/Bearish/Neutral)
2. Key technical indicators interpretation
3. Risk level (Low/Medium/High)
4. Brief reasoning (2-3 sentences max)

Respond in JSON format:
{
  "sentiment": "Bullish/Bearish/Neutral",
  "confidence": 0.0-1.0,
  "risk_level": "Low/Medium/High", 
  "analysis": "Brief analysis here",
  "key_factors": ["factor1", "factor2"]
}"""

        market_context = f"""Current Bitcoin Market Data:
- Price: ${current_price:,.2f}
- RSI: {rsi:.1f}
- MACD: {macd:.4f}
- Volume: {volume:,.0f}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        if recent_news:
            market_context += f"\n- Recent News: {'; '.join(recent_news[:3])}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": market_context}
        ]
        
        try:
            response = self._make_llm_request(messages, temperature=0.3, max_tokens=200)
            
            if response:
                # Try to parse JSON response
                try:
                    # Clean up response if it contains extra text
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Add metadata
                        result['timestamp'] = datetime.now().isoformat()
                        result['raw_response'] = response
                        
                        self.logger.info(f"🧠 LLM Analysis: {result.get('sentiment', 'Unknown')} sentiment with {result.get('confidence', 0):.1%} confidence")
                        return result
                        
                except json.JSONDecodeError:
                    self.logger.warning("LLM response not in valid JSON format")
                    
                # Fallback: parse text response
                return {
                    'sentiment': 'Neutral',
                    'confidence': 0.5,
                    'risk_level': 'Medium',
                    'analysis': response[:200],
                    'key_factors': [],
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': response
                }
                    
        except Exception as e:
            self.logger.error(f"Error in LLM sentiment analysis: {e}")
        
        # Fallback response
        return {
            'sentiment': 'Neutral',
            'confidence': 0.0,
            'risk_level': 'High',
            'analysis': 'LLM analysis unavailable',
            'key_factors': [],
            'timestamp': datetime.now().isoformat(),
            'error': 'LLM request failed'
        }
    
    def enhance_trading_signal(self, current_signal: str, signal_confidence: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to enhance/validate a trading signal.
        
        Args:
            current_signal: Current trading signal (BUY/SELL/HOLD)
            signal_confidence: Confidence of current signal (0-1)
            market_data: Current market data
            
        Returns:
            Enhanced signal analysis
        """
        system_prompt = """You are a senior cryptocurrency trading advisor. A trading algorithm has generated a signal, and you need to provide additional analysis.

Evaluate:
1. Does the signal make sense given current market conditions?
2. What are the potential risks?
3. Should confidence be adjusted up or down?
4. Any critical factors the algorithm might have missed?

Respond with brief, actionable analysis in JSON:
{
  "validation": "CONFIRM/CAUTION/REJECT",
  "confidence_adjustment": -0.2 to +0.2,
  "risk_factors": ["factor1", "factor2"],
  "recommendation": "Brief recommendation",
  "reasoning": "Why this assessment"
}"""

        current_price = market_data.get('price', market_data.get('close', 0))
        
        user_prompt = f"""Algorithm Signal Analysis:
- Generated Signal: {current_signal}
- Algorithm Confidence: {signal_confidence:.1%}
- Current BTC Price: ${current_price:,.2f}
- RSI: {market_data.get('rsi', 50):.1f}
- MACD: {market_data.get('macd', 0):.4f}

Please validate this signal."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._make_llm_request(messages, temperature=0.4, max_tokens=150)
            
            if response:
                try:
                    # Extract JSON
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Apply confidence adjustment
                        adjusted_confidence = max(0.0, min(1.0, signal_confidence + result.get('confidence_adjustment', 0)))
                        
                        result['original_confidence'] = signal_confidence
                        result['adjusted_confidence'] = adjusted_confidence
                        result['timestamp'] = datetime.now().isoformat()
                        
                        validation = result.get('validation', 'CAUTION')
                        self.logger.info(f"🧠 LLM Signal Validation: {validation} - Confidence {signal_confidence:.1%} → {adjusted_confidence:.1%}")
                        
                        return result
                        
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error in LLM signal enhancement: {e}")
        
        # Fallback
        return {
            'validation': 'CAUTION',
            'confidence_adjustment': 0,
            'adjusted_confidence': signal_confidence,
            'risk_factors': ['LLM analysis unavailable'],
            'recommendation': 'Proceed with original signal',
            'reasoning': 'Unable to get LLM validation'
        }
    
    def generate_market_report(self, market_data: Dict[str, Any], recent_trades: List[Dict] = None) -> str:
        """
        Generate a comprehensive market report using LLM.
        
        Args:
            market_data: Current market data
            recent_trades: List of recent trades
            
        Returns:
            Market report text
        """
        system_prompt = """You are a professional cryptocurrency market analyst. Generate a concise market report (3-4 sentences) covering:
1. Current market conditions
2. Key technical levels
3. Risk assessment
4. Brief outlook

Be professional but accessible. No emojis."""

        current_price = market_data.get('price', market_data.get('close', 0))
        
        context = f"""Market Data for {datetime.now().strftime('%Y-%m-%d %H:%M')}:
- Bitcoin Price: ${current_price:,.2f}
- RSI: {market_data.get('rsi', 50):.1f}
- MACD: {market_data.get('macd', 0):.4f}
- Volume: {market_data.get('volume', 0):,.0f}"""

        if recent_trades:
            total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades[-5:])
            context += f"\n- Recent Trading P&L: ${total_pnl:.2f}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        response = self._make_llm_request(messages, temperature=0.6, max_tokens=200)
        
        if response:
            self.logger.info("📊 Generated LLM market report")
            return response.strip()
        
        return f"Market Report ({datetime.now().strftime('%H:%M')}): BTC at ${current_price:,.2f}. LLM analysis unavailable."