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
    
    def __init__(self, llm_endpoint: str = "http://localhost:1234", model_name: str = "openai/gpt-oss-20b", logger: Optional[logging.Logger] = None):
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
    
    def analyze_trading_performance(self, recent_trades: List[Dict], portfolio_data: Dict[str, Any], 
                                  open_positions: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze recent trading performance and provide insights.
        
        Args:
            recent_trades: List of recent trade data
            portfolio_data: Current portfolio information
            open_positions: Current open positions
            
        Returns:
            Dict containing performance analysis and recommendations
        """
        try:
            # Calculate performance metrics
            total_trades = len(recent_trades)
            if total_trades == 0:
                return {
                    'analysis': 'No recent trades to analyze.',
                    'performance_score': 0,
                    'recommendations': ['Wait for market opportunities'],
                    'risk_assessment': 'LOW'
                }
            
            # Calculate win rate and P&L metrics
            profitable_trades = [t for t in recent_trades if float(t.get('pnl', 0)) > 0]
            losing_trades = [t for t in recent_trades if float(t.get('pnl', 0)) < 0]
            
            win_rate = len(profitable_trades) / total_trades * 100
            total_pnl = sum(float(t.get('pnl', 0)) for t in recent_trades)
            avg_win = sum(float(t.get('pnl', 0)) for t in profitable_trades) / max(len(profitable_trades), 1)
            avg_loss = sum(float(t.get('pnl', 0)) for t in losing_trades) / max(len(losing_trades), 1)
            
            # Prepare context for LLM
            context = f"""
            TRADING PERFORMANCE ANALYSIS REQUEST
            
            Portfolio Status:
            - Total Value: ${portfolio_data.get('total_value', 3000):.2f}
            - Realized P&L: ${portfolio_data.get('realized_pnl', 0):.2f}
            - Unrealized P&L: ${portfolio_data.get('unrealized_pnl', 0):.2f}
            
            Recent Trading Activity ({total_trades} trades):
            - Win Rate: {win_rate:.1f}%
            - Total P&L: ${total_pnl:.2f}
            - Average Win: ${avg_win:.2f}
            - Average Loss: ${avg_loss:.2f}
            - Profitable Trades: {len(profitable_trades)}
            - Losing Trades: {len(losing_trades)}
            
            Open Positions: {len(open_positions or [])}
            
            Recent Trade Details:
            """
            
            # Add recent trade details
            for i, trade in enumerate(recent_trades[-5:], 1):  # Last 5 trades
                symbol = trade.get('symbol', 'N/A')
                side = trade.get('side', 'N/A')
                pnl = float(trade.get('pnl', 0))
                price = float(trade.get('price', 0))
                timestamp = trade.get('timestamp', 'N/A')
                
                context += f"\n{i}. {symbol} {side} @ ${price:.2f} | P&L: ${pnl:.2f} | {timestamp}"
            
            # LLM prompt for analysis
            prompt = f"""{context}
            
            As a professional trading analyst, provide a comprehensive analysis of this trading performance:
            
            1. PERFORMANCE ASSESSMENT (rate 1-10):
               - Overall performance quality
               - Risk management effectiveness
               - Trading discipline
            
            2. PATTERN ANALYSIS:
               - What patterns do you see in the trading behavior?
               - Are there any concerning trends?
               - What's working well?
            
            3. STRATEGIC RECOMMENDATIONS (3-5 specific actions):
               - Immediate adjustments needed
               - Risk management improvements
               - Strategy optimizations
            
            4. MARKET OUTLOOK:
               - How should the bot adapt to current conditions?
               - What opportunities or risks do you see?
            
            Keep analysis concise but actionable. Focus on data-driven insights.
            """
            
            messages = [{"role": "user", "content": prompt}]
            analysis = self._make_llm_request(messages, temperature=0.3, max_tokens=500)
            
            if analysis:
                # Extract performance score (simple heuristic)
                performance_score = min(10, max(1, 5 + (win_rate - 50) / 10 + (total_pnl / 100)))
                
                # Determine risk level
                risk_level = "HIGH" if total_pnl < -50 or win_rate < 30 else "MEDIUM" if total_pnl < 0 or win_rate < 45 else "LOW"
                
                return {
                    'analysis': analysis,
                    'performance_score': round(performance_score, 1),
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(total_pnl, 2),
                    'total_trades': total_trades,
                    'risk_assessment': risk_level,
                    'recommendations': self._extract_recommendations(analysis)
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing trading performance: {e}")
            
        return {
            'analysis': 'Performance analysis temporarily unavailable.',
            'performance_score': 5,
            'recommendations': ['Continue current strategy'],
            'risk_assessment': 'UNKNOWN'
        }
    
    def analyze_market_trends(self, price_data: pd.DataFrame, volume_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze current market trends and provide predictions.
        
        Args:
            price_data: Historical price data
            volume_data: Volume and market depth data
            
        Returns:
            Dict containing trend analysis and predictions
        """
        try:
            if price_data.empty:
                return {'trend': 'UNKNOWN', 'analysis': 'Insufficient data for trend analysis'}
            
            # Calculate trend indicators
            current_price = float(price_data['close'].iloc[-1])
            price_24h_ago = float(price_data['close'].iloc[-24]) if len(price_data) >= 24 else current_price
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Calculate moving averages
            ma_20 = price_data['close'].rolling(20).mean().iloc[-1] if len(price_data) >= 20 else current_price
            ma_50 = price_data['close'].rolling(50).mean().iloc[-1] if len(price_data) >= 50 else current_price
            
            # Calculate volatility
            volatility = price_data['close'].pct_change().std() * 100 if len(price_data) > 1 else 0
            
            # Determine trend direction
            if current_price > ma_20 > ma_50:
                trend = "BULLISH"
            elif current_price < ma_20 < ma_50:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS"
            
            # Prepare context for LLM
            context = f"""
            MARKET TREND ANALYSIS REQUEST
            
            Current Market Data:
            - Current Price: ${current_price:.2f}
            - 24h Change: {price_change_24h:+.2f}%
            - 20-period MA: ${ma_20:.2f}
            - 50-period MA: ${ma_50:.2f}
            - Volatility: {volatility:.2f}%
            - Technical Trend: {trend}
            
            Recent Price Action (last 10 candles):
            """
            
            # Add recent price action
            for i, (idx, row) in enumerate(price_data.tail(10).iterrows(), 1):
                open_p = float(row['open'])
                high_p = float(row['high'])
                low_p = float(row['low'])
                close_p = float(row['close'])
                change = ((close_p - open_p) / open_p) * 100
                
                context += f"\n{i}. O:${open_p:.2f} H:${high_p:.2f} L:${low_p:.2f} C:${close_p:.2f} ({change:+.2f}%)"
            
            # LLM prompt for trend analysis
            prompt = f"""{context}
            
            As a technical analyst, provide a comprehensive market trend analysis:
            
            1. TREND CONFIRMATION:
               - Is the technical trend ({trend}) reliable?
               - What key levels should we watch?
               - How strong is the current momentum?
            
            2. SHORT-TERM OUTLOOK (next 1-4 hours):
               - Expected price direction
               - Key support/resistance levels
               - Volatility expectations
            
            3. TRADING IMPLICATIONS:
               - Best strategy for current conditions
               - Entry/exit timing considerations
               - Risk management advice
            
            4. MARKET STRUCTURE:
               - What's driving current price action?
               - Any concerning patterns or signals?
            
            Be specific with price levels and actionable insights.
            """
            
            messages = [{"role": "user", "content": prompt}]
            analysis = self._make_llm_request(messages, temperature=0.2, max_tokens=400)
            
            if analysis:
                return {
                    'trend': trend,
                    'current_price': round(current_price, 2),
                    'price_change_24h': round(price_change_24h, 2),
                    'volatility': round(volatility, 2),
                    'analysis': analysis,
                    'key_levels': {
                        'ma_20': round(ma_20, 2),
                        'ma_50': round(ma_50, 2),
                    },
                    'momentum': 'STRONG' if abs(price_change_24h) > 3 else 'MODERATE' if abs(price_change_24h) > 1 else 'WEAK'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing market trends: {e}")
            
        return {
            'trend': 'UNKNOWN',
            'analysis': 'Trend analysis temporarily unavailable.',
            'momentum': 'UNKNOWN'
        }
    
    def generate_trading_insights(self, performance_data: Dict[str, Any], trend_data: Dict[str, Any], 
                                market_conditions: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive trading insights combining performance and trend analysis.
        
        Args:
            performance_data: Output from analyze_trading_performance
            trend_data: Output from analyze_market_trends
            market_conditions: Additional market context
            
        Returns:
            Formatted insights report
        """
        try:
            # Prepare combined analysis context
            context = f"""
            COMPREHENSIVE TRADING INSIGHTS REQUEST
            
            PERFORMANCE SUMMARY:
            - Performance Score: {performance_data.get('performance_score', 'N/A')}/10
            - Win Rate: {performance_data.get('win_rate', 'N/A')}%
            - Total P&L: ${performance_data.get('total_pnl', 'N/A')}
            - Risk Level: {performance_data.get('risk_assessment', 'N/A')}
            - Total Trades: {performance_data.get('total_trades', 'N/A')}
            
            MARKET TREND ANALYSIS:
            - Current Trend: {trend_data.get('trend', 'N/A')}
            - Price Change 24h: {trend_data.get('price_change_24h', 'N/A')}%
            - Momentum: {trend_data.get('momentum', 'N/A')}
            - Volatility: {trend_data.get('volatility', 'N/A')}%
            - Current Price: ${trend_data.get('current_price', 'N/A')}
            """
            
            if market_conditions:
                context += f"\n\nMARKET CONDITIONS:\n"
                for key, value in market_conditions.items():
                    context += f"- {key}: {value}\n"
            
            # LLM prompt for comprehensive insights
            prompt = f"""{context}
            
            As an expert trading strategist, synthesize this information into actionable insights:
            
            1. STRATEGIC ASSESSMENT:
               - How well is the current strategy performing given market conditions?
               - What adjustments are needed for optimal performance?
            
            2. IMMEDIATE ACTION ITEMS (next 1-2 hours):
               - Specific trading adjustments
               - Risk management changes
               - Position sizing recommendations
            
            3. PERFORMANCE OPTIMIZATION:
               - What's limiting current performance?
               - How can win rate and profitability be improved?
            
            4. MARKET ADAPTATION:
               - How should the bot adapt to current trends?
               - What opportunities are being missed?
            
            Provide 3-5 concrete, actionable recommendations. Be specific and direct.
            """
            
            messages = [{"role": "user", "content": prompt}]
            insights = self._make_llm_request(messages, temperature=0.4, max_tokens=600)
            
            if insights:
                return f"""
🤖 LLM TRADING INSIGHTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{insights}

📊 KEY METRICS:
• Performance Score: {performance_data.get('performance_score', 'N/A')}/10
• Win Rate: {performance_data.get('win_rate', 'N/A')}%
• Current Trend: {trend_data.get('trend', 'N/A')}
• Risk Level: {performance_data.get('risk_assessment', 'N/A')}
""".strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trading insights: {e}")
            
        return f"""
🤖 LLM TRADING INSIGHTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Trading insights temporarily unavailable due to technical issues.
Continuing with current strategy parameters.

📊 KEY METRICS:
• Performance Score: {performance_data.get('performance_score', 'N/A')}/10
• Win Rate: {performance_data.get('win_rate', 'N/A')}%
• Current Trend: {trend_data.get('trend', 'N/A')}
• Risk Level: {performance_data.get('risk_assessment', 'N/A')}
"""
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract actionable recommendations from LLM analysis."""
        try:
            # Simple extraction of recommendations (could be improved with better parsing)
            recommendations = []
            lines = analysis.split('\n')
            
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['recommend', 'should', 'consider', 'adjust', 'improve']):
                    if len(line) > 10 and not line.endswith(':'):
                        recommendations.append(line)
            
            # Fallback to generic recommendations
            if not recommendations:
                recommendations = [
                    'Continue monitoring market conditions',
                    'Maintain current risk management',
                    'Review position sizing'
                ]
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception:
            return ['Continue current trading strategy']