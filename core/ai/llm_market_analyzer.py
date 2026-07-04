#!/usr/bin/env python3
"""
LLM-Powered Market Analysis and Reasoning for ELVIS Trading System
Advanced AI integration with Large Language Models for intelligent trading decisions
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np

# LLM API clients
import openai
import pandas as pd

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketSentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketAnalysis:
    """Container for LLM market analysis results"""

    timestamp: datetime
    symbol: str
    sentiment: MarketSentiment
    confidence: ConfidenceLevel
    reasoning: str
    key_factors: List[str]
    price_prediction: Dict[str, float]  # {"direction": +1/-1, "magnitude": 0.0-1.0}
    risk_assessment: Dict[str, Any]
    trading_recommendation: str
    supporting_data: Dict[str, Any]


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: str  # "openai", "anthropic", "gemini", "local"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30


class LLMMarketAnalyzer:
    """Advanced LLM-powered market analysis system"""

    def __init__(
        self, config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None
    ):
        self.config = config
        self.fallback_configs = fallback_configs or []
        self.current_provider = None

        # Analysis history
        self.analysis_history = []
        self.max_history = 1000

        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_confidence": 0.0,
            "accuracy_tracking": [],
            "response_times": [],
        }

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize LLM providers based on configuration"""
        providers = {}

        # OpenAI
        if self.config.provider == "openai" or any(
            c.provider == "openai" for c in self.fallback_configs
        ):
            if self.config.api_key:
                openai.api_key = self.config.api_key
                providers["openai"] = True
                logger.info("✅ OpenAI provider initialized")

        # Anthropic Claude
        if ANTHROPIC_AVAILABLE and (
            self.config.provider == "anthropic"
            or any(c.provider == "anthropic" for c in self.fallback_configs)
        ):
            if self.config.api_key:
                providers["anthropic"] = anthropic.Client(api_key=self.config.api_key)
                logger.info("✅ Anthropic provider initialized")

        # Google Gemini
        if GEMINI_AVAILABLE and (
            self.config.provider == "gemini"
            or any(c.provider == "gemini" for c in self.fallback_configs)
        ):
            if self.config.api_key:
                genai.configure(api_key=self.config.api_key)
                providers["gemini"] = True
                logger.info("✅ Gemini provider initialized")

        # Local LLM (LM Studio, Ollama, etc.)
        if self.config.provider == "local":
            providers["local"] = True
            logger.info("✅ Local LLM provider configured")

        self.providers = providers
        self.current_provider = (
            self.config.provider if self.config.provider in providers else None
        )

        if not self.current_provider:
            logger.warning(
                "⚠️  No LLM providers available, analysis will use fallback logic"
            )

    async def analyze_market_conditions(
        self,
        market_data: Dict[str, Any],
        news_data: Optional[List[Dict]] = None,
        social_data: Optional[Dict] = None,
        technical_analysis: Optional[Dict] = None,
    ) -> MarketAnalysis:
        """
        Comprehensive market analysis using LLM reasoning

        Args:
            market_data: Current price, volume, and market metrics
            news_data: Recent news articles and headlines
            social_data: Social media sentiment and trends
            technical_analysis: Technical indicator values and signals

        Returns:
            MarketAnalysis with LLM-generated insights
        """

        start_time = time.time()

        try:
            # Prepare context for LLM
            context = self._prepare_analysis_context(
                market_data, news_data, social_data, technical_analysis
            )

            # Generate analysis prompt
            prompt = self._create_analysis_prompt(context)

            # Get LLM analysis
            llm_response = await self._query_llm(prompt)

            # Parse and structure response
            analysis = self._parse_llm_response(llm_response, market_data)

            # Update performance metrics
            response_time = time.time() - start_time
            self._update_metrics(analysis, response_time)

            # Store in history
            self.analysis_history.append(analysis)
            if len(self.analysis_history) > self.max_history:
                self.analysis_history.pop(0)

            logger.info(
                f"📊 Market analysis complete: {analysis.sentiment.value} ({analysis.confidence.value})"
            )
            return analysis

        except Exception as e:
            logger.error(f"❌ Market analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(market_data)

    def _prepare_analysis_context(
        self,
        market_data: Dict,
        news_data: Optional[List],
        social_data: Optional[Dict],
        technical_analysis: Optional[Dict],
    ) -> Dict:
        """Prepare structured context for LLM analysis"""

        context = {
            "timestamp": datetime.now().isoformat(),
            "symbol": market_data.get("symbol", "BTCUSDT"),
            "market_data": {
                "current_price": market_data.get("price", 0),
                "24h_change": market_data.get("price_change_24h", 0),
                "24h_volume": market_data.get("volume_24h", 0),
                "market_cap": market_data.get("market_cap"),
                "volatility": market_data.get("volatility", 0),
            },
        }

        # Add technical analysis
        if technical_analysis:
            context["technical_indicators"] = {
                "rsi": technical_analysis.get("rsi", 50),
                "macd": technical_analysis.get("macd", {}),
                "bollinger_bands": technical_analysis.get("bollinger_bands", {}),
                "moving_averages": technical_analysis.get("moving_averages", {}),
                "support_resistance": technical_analysis.get("support_resistance", {}),
                "trend_analysis": technical_analysis.get("trend", "neutral"),
            }

        # Add news sentiment
        if news_data:
            context["news_analysis"] = {
                "recent_headlines": [item.get("title", "") for item in news_data[:5]],
                "news_sentiment": self._analyze_news_sentiment(news_data),
                "key_topics": self._extract_news_topics(news_data),
            }

        # Add social media data
        if social_data:
            context["social_sentiment"] = {
                "twitter_sentiment": social_data.get("twitter_sentiment", 0),
                "reddit_sentiment": social_data.get("reddit_sentiment", 0),
                "social_volume": social_data.get("social_volume", 0),
                "trending_topics": social_data.get("trending_topics", []),
            }

        return context

    def _create_analysis_prompt(self, context: Dict) -> str:
        """Create comprehensive analysis prompt for LLM"""

        symbol = context["symbol"]
        market_data = context["market_data"]

        prompt = f"""
You are an expert cryptocurrency trader and market analyst with decades of experience. Analyze the following market data for {symbol} and provide a comprehensive trading assessment.

MARKET DATA:
- Current Price: ${market_data['current_price']:,.2f}
- 24h Change: {market_data['24h_change']:+.2f}%
- 24h Volume: ${market_data['24h_volume']:,.0f}
- Volatility: {market_data['volatility']:.2f}%

"""

        # Add technical analysis if available
        if "technical_indicators" in context:
            tech = context["technical_indicators"]
            prompt += f"""
TECHNICAL ANALYSIS:
- RSI: {tech['rsi']:.1f}
- Trend: {tech['trend_analysis']}
- MACD Signal: {tech.get('macd', {}).get('signal', 'neutral')}
- Bollinger Position: {tech.get('bollinger_bands', {}).get('position', 'middle')}

"""

        # Add news analysis if available
        if "news_analysis" in context:
            news = context["news_analysis"]
            prompt += f"""
NEWS SENTIMENT:
- Overall News Sentiment: {news['news_sentiment']:.2f} (-1 to +1 scale)
- Key Headlines: {', '.join(news['recent_headlines'][:3])}
- Major Topics: {', '.join(news['key_topics'][:3])}

"""

        # Add social sentiment if available
        if "social_sentiment" in context:
            social = context["social_sentiment"]
            prompt += f"""
SOCIAL MEDIA SENTIMENT:
- Twitter Sentiment: {social['twitter_sentiment']:.2f}
- Reddit Sentiment: {social['reddit_sentiment']:.2f}
- Social Volume: {social['social_volume']}
- Trending: {', '.join(social['trending_topics'][:3])}

"""

        prompt += """
Based on this comprehensive data, provide your analysis in the following JSON format:

{
    "sentiment": "bullish|bearish|neutral|volatile",
    "confidence": "low|medium|high|very_high",
    "reasoning": "Detailed explanation of your analysis (2-3 sentences)",
    "key_factors": ["factor1", "factor2", "factor3"],
    "price_prediction": {
        "direction": 1 or -1,
        "magnitude": 0.0 to 1.0
    },
    "risk_assessment": {
        "risk_level": "low|medium|high",
        "volatility_warning": true or false,
        "key_risks": ["risk1", "risk2"]
    },
    "trading_recommendation": "One clear trading recommendation"
}

Focus on actionable insights and be specific about your reasoning. Consider both short-term (1-4 hours) and medium-term (1-3 days) perspectives.
"""

        return prompt

    async def _query_llm(self, prompt: str) -> str:
        """Query the configured LLM provider"""

        try:
            if self.current_provider == "openai":
                return await self._query_openai(prompt)
            elif self.current_provider == "anthropic":
                return await self._query_anthropic(prompt)
            elif self.current_provider == "gemini":
                return await self._query_gemini(prompt)
            elif self.current_provider == "local":
                return await self._query_local_llm(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.current_provider}")

        except Exception as e:
            logger.warning(f"⚠️  Primary LLM failed: {e}")
            # Try fallback providers
            for fallback_config in self.fallback_configs:
                try:
                    old_config = self.config
                    self.config = fallback_config
                    self.current_provider = fallback_config.provider

                    result = await self._query_llm(prompt)
                    logger.info(
                        f"✅ Fallback LLM succeeded: {fallback_config.provider}"
                    )
                    return result

                except Exception as fallback_error:
                    logger.warning(
                        f"⚠️  Fallback {fallback_config.provider} failed: {fallback_error}"
                    )
                    continue
                finally:
                    self.config = old_config

            # All providers failed
            raise Exception("All LLM providers failed")

    async def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.config.model or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    async def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude API"""
        if not ANTHROPIC_AVAILABLE:
            raise Exception("Anthropic library not available")

        try:
            client = self.providers.get("anthropic")
            response = await asyncio.to_thread(
                client.messages.create,
                model=self.config.model or "claude-3-sonnet-20240229",
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")

    async def _query_gemini(self, prompt: str) -> str:
        """Query Google Gemini API"""
        if not GEMINI_AVAILABLE:
            raise Exception("Gemini library not available")

        try:
            model = genai.GenerativeModel(self.config.model or "gemini-pro")
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    async def _query_local_llm(self, prompt: str) -> str:
        """Query local LLM (LM Studio, Ollama, etc.)"""
        base_url = self.config.base_url or "http://localhost:1234/v1"

        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "model": self.config.model or "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                }

                async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Local LLM returned status {response.status}")

            except Exception as e:
                raise Exception(f"Local LLM error: {e}")

    def _parse_llm_response(self, response: str, market_data: Dict) -> MarketAnalysis:
        """Parse LLM response into structured MarketAnalysis"""

        try:
            # Try to extract JSON from response
            response_clean = response.strip()

            # Find JSON in response (handle cases where LLM adds extra text)
            json_start = response_clean.find("{")
            json_end = response_clean.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_clean[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Create MarketAnalysis object
            analysis = MarketAnalysis(
                timestamp=datetime.now(),
                symbol=market_data.get("symbol", "BTCUSDT"),
                sentiment=MarketSentiment(parsed.get("sentiment", "neutral")),
                confidence=ConfidenceLevel(parsed.get("confidence", "medium")),
                reasoning=parsed.get("reasoning", "LLM analysis completed"),
                key_factors=parsed.get("key_factors", []),
                price_prediction=parsed.get(
                    "price_prediction", {"direction": 0, "magnitude": 0.5}
                ),
                risk_assessment=parsed.get(
                    "risk_assessment",
                    {"risk_level": "medium", "volatility_warning": False},
                ),
                trading_recommendation=parsed.get(
                    "trading_recommendation", "Hold current position"
                ),
                supporting_data={"raw_response": response, "parsed_data": parsed},
            )

            return analysis

        except Exception as e:
            logger.warning(f"⚠️  Failed to parse LLM response: {e}")
            return self._create_fallback_analysis(market_data, response)

    def _create_fallback_analysis(
        self, market_data: Dict, llm_response: str = None
    ) -> MarketAnalysis:
        """Create fallback analysis when LLM fails"""

        # Simple technical analysis fallback
        price_change = market_data.get("price_change_24h", 0)
        volatility = market_data.get("volatility", 10)

        if price_change > 5:
            sentiment = MarketSentiment.BULLISH
            direction = 1
        elif price_change < -5:
            sentiment = MarketSentiment.BEARISH
            direction = -1
        else:
            sentiment = MarketSentiment.NEUTRAL
            direction = 0

        confidence = ConfidenceLevel.LOW if volatility > 20 else ConfidenceLevel.MEDIUM

        return MarketAnalysis(
            timestamp=datetime.now(),
            symbol=market_data.get("symbol", "BTCUSDT"),
            sentiment=sentiment,
            confidence=confidence,
            reasoning=f"Fallback analysis based on {price_change:+.2f}% price change and {volatility:.1f}% volatility",
            key_factors=["price_momentum", "volatility_level"],
            price_prediction={
                "direction": direction,
                "magnitude": min(abs(price_change) / 20, 1.0),
            },
            risk_assessment={
                "risk_level": "medium",
                "volatility_warning": volatility > 20,
            },
            trading_recommendation="Monitor market conditions closely",
            supporting_data={"fallback_mode": True, "llm_response": llm_response},
        )

    def _analyze_news_sentiment(self, news_data: List[Dict]) -> float:
        """Simple news sentiment analysis (placeholder for more sophisticated analysis)"""
        # This would integrate with sentiment analysis libraries like TextBlob, VADER, etc.
        positive_keywords = [
            "bullish",
            "surge",
            "rally",
            "breakthrough",
            "adoption",
            "positive",
        ]
        negative_keywords = ["bearish", "crash", "drop", "concern", "regulation", "ban"]

        sentiment_score = 0
        total_articles = len(news_data)

        for article in news_data:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            text = f"{title} {content}"

            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)

            if positive_count > negative_count:
                sentiment_score += 1
            elif negative_count > positive_count:
                sentiment_score -= 1

        return sentiment_score / max(total_articles, 1)

    def _extract_news_topics(self, news_data: List[Dict]) -> List[str]:
        """Extract key topics from news data"""
        # Simplified topic extraction
        topics = set()
        keywords = [
            "bitcoin",
            "ethereum",
            "regulation",
            "adoption",
            "institutional",
            "technical",
            "trading",
        ]

        for article in news_data:
            title = article.get("title", "").lower()
            for keyword in keywords:
                if keyword in title:
                    topics.add(keyword)

        return list(topics)[:5]

    def _update_metrics(self, analysis: MarketAnalysis, response_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] += 1
        self.performance_metrics["successful_analyses"] += 1
        self.performance_metrics["response_times"].append(response_time)

        # Keep only recent response times
        if len(self.performance_metrics["response_times"]) > 100:
            self.performance_metrics["response_times"] = self.performance_metrics[
                "response_times"
            ][-100:]

        # Update average confidence (simplified)
        confidence_values = {"low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 1.0}
        current_confidence = confidence_values.get(analysis.confidence.value, 0.5)

        total = self.performance_metrics["total_analyses"]
        avg = self.performance_metrics["average_confidence"]
        self.performance_metrics["average_confidence"] = (
            (avg * (total - 1)) + current_confidence
        ) / total

    def get_analysis_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent analyses"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_analyses = [
            a for a in self.analysis_history if a.timestamp >= cutoff_time
        ]

        if not recent_analyses:
            return {"message": "No recent analyses available"}

        sentiment_counts = {}
        confidence_counts = {}

        for analysis in recent_analyses:
            sentiment_counts[analysis.sentiment.value] = (
                sentiment_counts.get(analysis.sentiment.value, 0) + 1
            )
            confidence_counts[analysis.confidence.value] = (
                confidence_counts.get(analysis.confidence.value, 0) + 1
            )

        return {
            "period_hours": hours,
            "total_analyses": len(recent_analyses),
            "sentiment_distribution": sentiment_counts,
            "confidence_distribution": confidence_counts,
            "latest_analysis": asdict(recent_analyses[-1]) if recent_analyses else None,
            "performance_metrics": self.performance_metrics,
        }

    async def batch_analyze(self, market_data_list: List[Dict]) -> List[MarketAnalysis]:
        """Analyze multiple markets concurrently"""
        tasks = []
        for market_data in market_data_list:
            task = self.analyze_market_conditions(market_data)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for item {i}: {result}")
                analyses.append(self._create_fallback_analysis(market_data_list[i]))
            else:
                analyses.append(result)

        return analyses


# Usage example and testing
async def test_llm_analyzer():
    """Test the LLM market analyzer"""

    # Configuration for different providers
    configs = [
        LLMConfig(
            provider="local",
            model="llama-3.1-8b",
            base_url="http://localhost:1234/v1",
            temperature=0.3,
        ),
        LLMConfig(
            provider="openai", model="gpt-4", api_key="your-openai-key", temperature=0.3
        ),
    ]

    # Create analyzer with fallbacks
    analyzer = LLMMarketAnalyzer(
        config=configs[0],  # Primary: local LLM
        fallback_configs=configs[1:],  # Fallback: OpenAI
    )

    # Sample market data
    market_data = {
        "symbol": "BTCUSDT",
        "price": 65000,
        "price_change_24h": 5.2,
        "volume_24h": 25000000,
        "volatility": 15.3,
    }

    # Sample technical analysis
    technical_analysis = {
        "rsi": 65,
        "trend": "bullish",
        "macd": {"signal": "buy"},
        "bollinger_bands": {"position": "upper"},
    }

    # Sample news data
    news_data = [
        {
            "title": "Bitcoin Surges as Institutional Adoption Increases",
            "content": "Major companies announce Bitcoin treasury adoption...",
        },
        {
            "title": "Crypto Market Shows Strong Bullish Momentum",
            "content": "Technical indicators suggest continued upward trend...",
        },
    ]

    print("🧠 Testing LLM Market Analyzer")

    try:
        # Perform analysis
        analysis = await analyzer.analyze_market_conditions(
            market_data=market_data,
            news_data=news_data,
            technical_analysis=technical_analysis,
        )

        print(f"\n📊 Analysis Results:")
        print(f"   Sentiment: {analysis.sentiment.value}")
        print(f"   Confidence: {analysis.confidence.value}")
        print(f"   Reasoning: {analysis.reasoning}")
        print(f"   Key Factors: {analysis.key_factors}")
        print(f"   Price Direction: {analysis.price_prediction['direction']}")
        print(f"   Recommendation: {analysis.trading_recommendation}")

        # Get summary
        summary = analyzer.get_analysis_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total Analyses: {summary['performance_metrics']['total_analyses']}")
        print(
            f"   Success Rate: {summary['performance_metrics']['successful_analyses']}/{summary['performance_metrics']['total_analyses']}"
        )
        print(
            f"   Average Confidence: {summary['performance_metrics']['average_confidence']:.2f}"
        )

        return analysis

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(test_llm_analyzer())
