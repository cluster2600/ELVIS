"""
Data processing module for the ELVIS trading system.
Handles alternative data sources, feature engineering, and data quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
import ccxt
import requests
import json
from datetime import datetime, timedelta
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    exchange: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    onchain_api_key: Optional[str] = None
    data_dir: str = "data"
    cache_dir: str = "cache"

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    technical_indicators: List[str] = None
    market_regime_features: bool = True
    onchain_features: bool = True
    orderbook_features: bool = True
    funding_features: bool = True
    feature_selection: bool = True
    n_features: int = 50

@dataclass
class DataQualityConfig:
    """Configuration for data quality."""
    handle_missing_data: bool = True
    detect_outliers: bool = True
    outlier_threshold: float = 3.0
    validation_rules: Dict = None

class DataProcessor:
    """Processes and enhances trading data."""
    
    def __init__(self, 
                 source_config: DataSourceConfig = None,
                 feature_config: FeatureConfig = None,
                 quality_config: DataQualityConfig = None):
        """
        Initialize the data processor.
        
        Args:
            source_config: Data source configuration
            feature_config: Feature engineering configuration
            quality_config: Data quality configuration
        """
        self.source_config = source_config or DataSourceConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.quality_config = quality_config or DataQualityConfig()
        
        if self.feature_config.technical_indicators is None:
            self.feature_config.technical_indicators = [
                'rsi', 'stoch', 'macd', 'bbands', 'atr', 'ema', 'vwap'
            ]
            
        if self.quality_config.validation_rules is None:
            self.quality_config.validation_rules = {
                'price': {'min': 0, 'max': 1e6},
                'volume': {'min': 0},
                'returns': {'min': -1, 'max': 1}
            }
            
        self.logger = logging.getLogger(__name__)
        self.exchange = self._init_exchange()
        self.scaler = RobustScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = None
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize cryptocurrency exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.source_config.exchange)
            exchange = exchange_class({
                'apiKey': self.source_config.api_key,
                'secret': self.source_config.api_secret,
                'enableRateLimit': True
            })
            return exchange
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            raise
            
    def fetch_market_data(self,
                         symbol: str,
                         timeframe: str = '1h',
                         limit: int = 1000) -> pd.DataFrame:
        """
        Fetch market data from exchange.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with market data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Fetch order book data
            if self.feature_config.orderbook_features:
                orderbook = self.exchange.fetch_order_book(symbol)
                df['bid'] = orderbook['bids'][0][0]
                df['ask'] = orderbook['asks'][0][0]
                df['spread'] = df['ask'] - df['bid']
                
            # Fetch funding rates
            if self.feature_config.funding_features:
                funding = self.exchange.fetch_funding_rate(symbol)
                df['funding_rate'] = funding['fundingRate']
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise
            
    def fetch_onchain_data(self,
                          symbol: str,
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """
        Fetch on-chain data for cryptocurrency.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            DataFrame with on-chain data
        """
        try:
            # Fetch data from blockchain explorer API
            base_url = "https://api.blockchain.info"
            headers = {'api_key': self.source_config.onchain_api_key}
            
            # Fetch network statistics
            stats = requests.get(
                f"{base_url}/stats",
                headers=headers
            ).json()
            
            # Fetch transaction volume
            tx_volume = requests.get(
                f"{base_url}/charts/transactions-per-second",
                headers=headers,
                params={
                    'timespan': '1day',
                    'start': int(start_time.timestamp()),
                    'end': int(end_time.timestamp())
                }
            ).json()
            
            # Create DataFrame
            df = pd.DataFrame({
                'hash_rate': stats['hash_rate'],
                'difficulty': stats['difficulty'],
                'tx_volume': [x['y'] for x in tx_volume['values']],
                'timestamp': pd.date_range(start_time, end_time, freq='1H')
            })
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain data: {str(e)}")
            raise
            
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            # RSI
            if 'rsi' in self.feature_config.technical_indicators:
                rsi = RSIIndicator(close=df['close'])
                df['rsi'] = rsi.rsi()
                
            # Stochastic Oscillator
            if 'stoch' in self.feature_config.technical_indicators:
                stoch = StochasticOscillator(
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                
            # MACD
            if 'macd' in self.feature_config.technical_indicators:
                macd = MACD(close=df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()
                
            # Bollinger Bands
            if 'bbands' in self.feature_config.technical_indicators:
                bbands = BollingerBands(close=df['close'])
                df['bb_high'] = bbands.bollinger_hband()
                df['bb_low'] = bbands.bollinger_lband()
                df['bb_mid'] = bbands.bollinger_mavg()
                
            # ATR
            if 'atr' in self.feature_config.technical_indicators:
                atr = AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
                df['atr'] = atr.average_true_range()
                
            # EMA
            if 'ema' in self.feature_config.technical_indicators:
                ema = EMAIndicator(close=df['close'])
                df['ema_20'] = ema.ema_indicator()
                
            # VWAP
            if 'vwap' in self.feature_config.technical_indicators:
                vwap = VolumeWeightedAveragePrice(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=df['volume']
                )
                df['vwap'] = vwap.volume_weighted_average_price()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features to DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added regime features
        """
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Volatility regime
            df['volatility'] = df['returns'].rolling(20).std()
            df['high_vol'] = (df['volatility'] > df['volatility'].rolling(100).mean() * 1.5).astype(int)
            df['low_vol'] = (df['volatility'] < df['volatility'].rolling(100).mean() * 0.5).astype(int)
            
            # Trend regime
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['uptrend'] = (df['sma_20'] > df['sma_50']).astype(int)
            df['downtrend'] = (df['sma_20'] < df['sma_50']).astype(int)
            
            # Momentum regime
            df['momentum'] = df['close'].pct_change(20)
            df['high_momentum'] = (df['momentum'] > df['momentum'].rolling(100).mean() + 
                                 df['momentum'].rolling(100).std()).astype(int)
            df['low_momentum'] = (df['momentum'] < df['momentum'].rolling(100).mean() - 
                                df['momentum'].rolling(100).std()).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding market regime features: {str(e)}")
            raise
            
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in DataFrame.
        
        Args:
            df: DataFrame with potential missing data
            
        Returns:
            DataFrame with handled missing data
        """
        try:
            # Forward fill for price data
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill()
            
            # Fill missing volume with 0
            df['volume'] = df['volume'].fillna(0)
            
            # Use KNN imputer for other features
            other_cols = [col for col in df.columns if col not in price_cols + ['volume']]
            if other_cols:
                df[other_cols] = self.imputer.fit_transform(df[other_cols])
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing data: {str(e)}")
            raise
            
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in DataFrame.
        
        Args:
            df: DataFrame with potential outliers
            
        Returns:
            DataFrame with handled outliers
        """
        try:
            # Calculate z-scores
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
            
            # Replace outliers with rolling median
            for col in numeric_cols:
                mask = z_scores[col] > self.quality_config.outlier_threshold
                df.loc[mask, col] = df[col].rolling(20, min_periods=1).median()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            raise
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data according to rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Whether data is valid
        """
        try:
            for col, rules in self.quality_config.validation_rules.items():
                if col in df.columns:
                    if 'min' in rules and (df[col] < rules['min']).any():
                        self.logger.warning(f"Values below minimum in {col}")
                        return False
                    if 'max' in rules and (df[col] > rules['max']).any():
                        self.logger.warning(f"Values above maximum in {col}")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return False
            
    def select_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Select most relevant features.
        
        Args:
            df: DataFrame with all features
            target: Target variable name
            
        Returns:
            DataFrame with selected features
        """
        try:
            if not self.feature_config.feature_selection:
                return df
                
            # Prepare data
            X = df.drop(columns=[target])
            y = df[target]
            
            # Select features
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=min(self.feature_config.n_features, X.shape[1])
            )
            selector.fit(X, y)
            
            # Get selected features
            selected_features = X.columns[selector.get_support()]
            self.feature_selector = selector
            
            return df[selected_features]
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return df
            
    def process_data(self,
                    symbol: str,
                    start_time: datetime,
                    end_time: datetime,
                    target: str = 'returns') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process and enhance trading data.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for data
            end_time: End time for data
            target: Target variable name
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Fetch market data
            df = self.fetch_market_data(symbol)
            
            # Fetch on-chain data
            if self.feature_config.onchain_features:
                onchain_df = self.fetch_onchain_data(symbol, start_time, end_time)
                df = df.join(onchain_df)
                
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add market regime features
            if self.feature_config.market_regime_features:
                df = self.add_market_regime_features(df)
                
            # Handle missing data
            if self.quality_config.handle_missing_data:
                df = self.handle_missing_data(df)
                
            # Detect and handle outliers
            if self.quality_config.detect_outliers:
                df = self.detect_outliers(df)
                
            # Validate data
            if not self.validate_data(df):
                raise ValueError("Data validation failed")
                
            # Select features
            df = self.select_features(df, target)
            
            # Split features and target
            X = df.drop(columns=[target])
            y = df[target]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise 