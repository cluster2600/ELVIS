"""
Sentiment Analysis strategy for the ELVIS project.
This module provides a concrete implementation of the BaseStrategy using sentiment analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional
import talib
import requests
import json
from datetime import datetime, timedelta

from trading.strategies.base_strategy import BaseStrategy
from config import TRADING_CONFIG, API_CONFIG

class SentimentStrategy(BaseStrategy):
    """
    Strategy based on sentiment analysis.
    Uses news sentiment and social media data to generate trading signals.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the sentiment strategy.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger, **kwargs)
        
        # Strategy parameters
        self.sentiment_threshold_positive = kwargs.get('sentiment_threshold_positive', 0.6)
        self.sentiment_threshold_negative = kwargs.get('sentiment_threshold_negative', -0.3)
        self.sentiment_weight = kwargs.get('sentiment_weight', 0.5)
        self.technical_weight = kwargs.get('technical_weight', 0.5)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', TRADING_CONFIG['STOP_LOSS_PCT'])
        self.take_profit_pct = kwargs.get('take_profit_pct', TRADING_CONFIG['TAKE_PROFIT_PCT'])
        self.position_size_pct = kwargs.get('position_size_pct', 0.5)  # 50% of available capital
        
        # API configuration
        self.news_api_key = API_CONFIG.get('NEWS_API_KEY', '')
        self.twitter_api_key = API_CONFIG.get('TWITTER_API_KEY', '')
        self.sentiment_cache = {}
        self.sentiment_cache_expiry = {}
        self.cache_duration = timedelta(hours=1)
    
    def _get_news_sentiment(self, symbol: str) -> float:
        """
        Get sentiment from news articles.
        
        Args:
            symbol (str): The trading symbol.
            
        Returns:
            float: The sentiment score (-1.0 to 1.0).
        """
        try:
            # Check cache
            cache_key = f"news_{symbol}"
            current_time = datetime.now()
            
            if (cache_key in self.sentiment_cache and 
                cache_key in self.sentiment_cache_expiry and 
                current_time < self.sentiment_cache_expiry[cache_key]):
                self.logger.info(f"Using cached news sentiment for {symbol}")
                return self.sentiment_cache[cache_key]
            
            # In a real implementation, this would call a news API
            # For demonstration, we'll simulate a sentiment score
            
            # Simulated API call
            self.logger.info(f"Fetching news sentiment for {symbol}")
            
            # Simulate sentiment based on current hour (for demonstration)
            hour = current_time.hour
            sentiment = np.sin(hour / 24 * 2 * np.pi) * 0.5
            
            # Cache the result
            self.sentiment_cache[cache_key] = sentiment
            self.sentiment_cache_expiry[cache_key] = current_time + self.cache_duration
            
            self.logger.info(f"News sentiment for {symbol}: {sentiment:.2f}")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return 0.0
    
    def _get_social_sentiment(self, symbol: str) -> float:
        """
        Get sentiment from social media.
        
        Args:
            symbol (str): The trading symbol.
            
        Returns:
            float: The sentiment score (-1.0 to 1.0).
        """
        try:
            # Check cache
            cache_key = f"social_{symbol}"
            current_time = datetime.now()
            
            if (cache_key in self.sentiment_cache and 
                cache_key in self.sentiment_cache_expiry and 
                current_time < self.sentiment_cache_expiry[cache_key]):
                self.logger.info(f"Using cached social sentiment for {symbol}")
                return self.sentiment_cache[cache_key]
            
            # In a real implementation, this would call a social media API
            # For demonstration, we'll simulate a sentiment score
            
            # Simulated API call
            self.logger.info(f"Fetching social sentiment for {symbol}")
            
            # Simulate sentiment based on current minute (for demonstration)
            minute = current_time.minute
            sentiment = np.cos(minute / 60 * 2 * np.pi) * 0.7
            
            # Cache the result
            self.sentiment_cache[cache_key] = sentiment
            self.sentiment_cache_expiry[cache_key] = current_time + self.cache_duration
            
            self.logger.info(f"Social sentiment for {symbol}: {sentiment:.2f}")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting social sentiment: {e}")
            return 0.0
    
    def _get_combined_sentiment(self, symbol: str) -> float:
        """
        Get combined sentiment from all sources.
        
        Args:
            symbol (str): The trading symbol.
            
        Returns:
            float: The combined sentiment score (-1.0 to 1.0).
        """
        news_sentiment = self._get_news_sentiment(symbol)
        social_sentiment = self._get_social_sentiment(symbol)
        
        # Combine sentiments (equal weight)
        combined_sentiment = (news_sentiment + social_sentiment) / 2
        
        self.logger.info(f"Combined sentiment for {symbol}: {combined_sentiment:.2f}")
        return combined_sentiment
    
    def _get_technical_signal(self, data: pd.DataFrame) -> float:
        """
        Get signal from technical indicators.
        
        Args:
            data (pd.DataFrame): The data to generate signals from.
            
        Returns:
            float: The technical signal (-1.0 to 1.0).
        """
        try:
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Check if required indicators are present
            required_indicators = ['rsi', 'macd', 'macdsignal']
            if not all(indicator in latest for indicator in required_indicators):
                self.logger.warning(f"Missing required indicators: {[ind for ind in required_indicators if ind not in latest]}")
                return 0.0
            
            # Extract indicators
            rsi = latest['rsi']
            macd = latest['macd']
            macdsignal = latest['macdsignal']
            
            # Calculate technical signal
            rsi_signal = (rsi - 50) / 50  # -1.0 to 1.0
            macd_signal = 1.0 if macd > macdsignal else -1.0
            
            # Combine signals (equal weight)
            technical_signal = (rsi_signal + macd_signal) / 2
            
            self.logger.info(f"Technical signal: {technical_signal:.2f} (RSI: {rsi:.2f}, MACD: {macd:.2f}, Signal: {macdsignal:.2f})")
            
            return technical_signal
            
        except Exception as e:
            self.logger.error(f"Error generating technical signal: {e}")
            return 0.0
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy/sell signals based on the data.
        
        Args:
            data (pd.DataFrame): The data to generate signals from.
            
        Returns:
            Tuple[bool, bool]: A tuple of (buy_signal, sell_signal).
        """
        if data.empty:
            self.logger.warning("Empty data provided to generate_signals")
            return False, False
        
        try:
            # Extract symbol from data
            symbol = data['tic'].iloc[0] if 'tic' in data.columns else 'BTC/USDT'
            
            # Get sentiment signal
            sentiment_signal = self._get_combined_sentiment(symbol)
            
            # Get technical signal
            technical_signal = self._get_technical_signal(data)
            
            # Combine signals with weights
            combined_signal = (
                sentiment_signal * self.sentiment_weight + 
                technical_signal * self.technical_weight
            )
            
            self.logger.info(f"Combined signal: {combined_signal:.2f} (Sentiment: {sentiment_signal:.2f}, Technical: {technical_signal:.2f})")
            
            # Generate buy/sell signals
            buy_signal = combined_signal > self.sentiment_threshold_positive
            sell_signal = combined_signal < self.sentiment_threshold_negative
            
            self.logger.info(f"Signal Check: Combined={combined_signal:.2f}, Buy={buy_signal}, Sell={sell_signal}")
            
            return buy_signal, sell_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return False, False
    
    def calculate_position_size(self, data: pd.DataFrame, current_price: float, available_capital: float) -> float:
        """
        Calculate the position size based on the data and available capital.
        
        Args:
            data (pd.DataFrame): The data to calculate position size from.
            current_price (float): The current price.
            available_capital (float): The available capital.
            
        Returns:
            float: The position size.
        """
        try:
            # Extract symbol from data
            symbol = data['tic'].iloc[0] if 'tic' in data.columns else 'BTC/USDT'
            
            # Get sentiment signal
            sentiment_signal = self._get_combined_sentiment(symbol)
            
            # Calculate position size as a percentage of available capital
            base_position_value = available_capital * self.position_size_pct
            
            # Adjust position size based on sentiment strength
            sentiment_factor = abs(sentiment_signal)
            adjusted_position_value = base_position_value * sentiment_factor
            
            # Convert to quantity
            quantity = adjusted_position_value / current_price
            
            # Adjust for minimum quantity
            min_quantity = 0.001  # Minimum BTC quantity
            if quantity < min_quantity:
                self.logger.warning(f"Calculated quantity {quantity:.8f} is below minimum {min_quantity}. Using minimum.")
                quantity = min_quantity
            
            self.logger.info(f"Calculated position size: {quantity:.8f} BTC (value: {quantity * current_price:.2f} USD)")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the stop loss price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate stop loss from.
            entry_price (float): The entry price.
            
        Returns:
            float: The stop loss price.
        """
        try:
            # Calculate stop loss based on ATR if available
            if 'atr' in data.columns:
                atr = data.iloc[-1]['atr']
                # Use 2 * ATR for stop loss
                stop_loss = entry_price - (2 * atr)
                
                # Ensure stop loss is not more than stop_loss_pct away from entry price
                min_stop_loss = entry_price * (1 - self.stop_loss_pct)
                stop_loss = max(stop_loss, min_stop_loss)
            else:
                # Use fixed percentage stop loss
                stop_loss = entry_price * (1 - self.stop_loss_pct)
            
            self.logger.info(f"Calculated stop loss: {stop_loss:.2f} USD (entry: {entry_price:.2f} USD)")
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price * (1 - self.stop_loss_pct)
    
    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate the take profit price based on the data and entry price.
        
        Args:
            data (pd.DataFrame): The data to calculate take profit from.
            entry_price (float): The entry price.
            
        Returns:
            float: The take profit price.
        """
        try:
            # Extract symbol from data
            symbol = data['tic'].iloc[0] if 'tic' in data.columns else 'BTC/USDT'
            
            # Get sentiment signal
            sentiment_signal = self._get_combined_sentiment(symbol)
            
            # Adjust take profit based on sentiment strength
            sentiment_factor = abs(sentiment_signal)
            adjusted_take_profit_pct = self.take_profit_pct * (1 + sentiment_factor)
            
            # Calculate take profit
            take_profit = entry_price * (1 + adjusted_take_profit_pct)
            
            self.logger.info(f"Calculated take profit: {take_profit:.2f} USD (entry: {entry_price:.2f} USD)")
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * (1 + self.take_profit_pct)
