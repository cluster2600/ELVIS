"""
Binance processor for the ELVIS project.
This module provides a concrete implementation of the BaseProcessor for Binance data.
"""

import os
import pandas as pd
import numpy as np
import ccxt
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
import talib

from core.data.processors.base_processor import BaseProcessor
from config import API_CONFIG

class BinanceProcessor(BaseProcessor):
    """
    Processor for Binance data.
    Fetches and processes data from Binance exchange.
    """
    
    def __init__(self, start_date: str, end_date: str, time_interval: str, logger: logging.Logger, **kwargs):
        """
        Initialize the Binance processor.
        
        Args:
            start_date (str): The start date.
            end_date (str): The end date.
            time_interval (str): The time interval.
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__('binance', start_date, end_date, time_interval, logger, **kwargs)
        
        # Initialize Binance client
        self.api_key = API_CONFIG['BINANCE_API_KEY']
        self.api_secret = API_CONFIG['BINANCE_API_SECRET']
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Cache for OHLCV data to reduce API calls
        self.ohlcv_cache = {}
        self.last_cache_update = {}
        self.cache_expiry = kwargs.get('cache_expiry', 300)  # 5 minutes by default
    
    def download_data(self, ticker_list: List[str]) -> pd.DataFrame:
        """
        Download data for the specified tickers.
        
        Args:
            ticker_list (List[str]): The list of tickers to download data for.
            
        Returns:
            pd.DataFrame: The downloaded data.
        """
        self.logger.info(f"Downloading data for {ticker_list} from {self.start_date} to {self.end_date}")
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        
        # Initialize DataFrame
        df_list = []
        
        # Download data for each ticker
        for ticker in ticker_list:
            try:
                # Check if we have cached data that's still valid
                current_time = time.time()
                if (ticker in self.ohlcv_cache and 
                    ticker in self.last_cache_update and 
                    current_time - self.last_cache_update[ticker] < self.cache_expiry):
                    self.logger.debug(f"Using cached data for {ticker}")
                    ohlcv = self.ohlcv_cache[ticker]
                else:
                    self.logger.info(f"Fetching OHLCV data for {ticker}")
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        ticker, 
                        timeframe=self.time_interval,
                        since=start_timestamp,
                        limit=1000  # Maximum limit
                    )
                    
                    # Cache the data
                    self.ohlcv_cache[ticker] = ohlcv
                    self.last_cache_update[ticker] = current_time
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add ticker column
                df['ticker'] = ticker
                
                # Add to list
                df_list.append(df)
                
            except Exception as e:
                self.logger.error(f"Error downloading data for {ticker}: {e}")
        
        # Combine all DataFrames
        if df_list:
            data = pd.concat(df_list, ignore_index=True)
            data = data.rename(columns={'timestamp': 'date'})
            data = data.sort_values('date')
            return data
        else:
            self.logger.warning("No data downloaded")
            return pd.DataFrame()
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the downloaded data.
        
        Returns:
            pd.DataFrame: The cleaned data.
        """
        if self.data is None or self.data.empty:
            self.logger.warning("No data to clean")
            return pd.DataFrame()
        
        self.logger.info("Cleaning data")
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Handle missing values
        self.data = self.data.fillna(method='ffill')
        
        # Filter by date range
        self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]
        
        return self.data
    
    def add_technical_indicator(self, tech_indicator_list: List[str]) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            tech_indicator_list (List[str]): The list of technical indicators to add.
            
        Returns:
            pd.DataFrame: The data with technical indicators.
        """
        if self.data is None or self.data.empty:
            self.logger.warning("No data to add technical indicators to")
            return pd.DataFrame()
        
        self.logger.info(f"Adding technical indicators: {tech_indicator_list}")
        
        # Group by ticker
        for ticker in self.data['ticker'].unique():
            ticker_data = self.data[self.data['ticker'] == ticker].copy()
            
            # Calculate technical indicators
            for indicator in tech_indicator_list:
                if indicator == 'macd':
                    # MACD
                    macd, macdsignal, macdhist = talib.MACD(
                        ticker_data['close'].values,
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                    ticker_data['macd'] = macd
                    ticker_data['macdsignal'] = macdsignal
                    ticker_data['macdhist'] = macdhist
                
                elif indicator == 'rsi':
                    # RSI
                    ticker_data['rsi'] = talib.RSI(ticker_data['close'].values, timeperiod=14)
                
                elif indicator == 'cci':
                    # CCI
                    ticker_data['cci'] = talib.CCI(
                        ticker_data['high'].values,
                        ticker_data['low'].values,
                        ticker_data['close'].values,
                        timeperiod=14
                    )
                
                elif indicator == 'dx':
                    # DX (Directional Movement Index)
                    ticker_data['dx'] = talib.DX(
                        ticker_data['high'].values,
                        ticker_data['low'].values,
                        ticker_data['close'].values,
                        timeperiod=14
                    )
                
                elif indicator == 'obv':
                    # OBV (On Balance Volume)
                    ticker_data['obv'] = talib.OBV(
                        ticker_data['close'].values,
                        ticker_data['volume'].values
                    )
                
                elif indicator == 'atr':
                    # ATR (Average True Range)
                    ticker_data['atr'] = talib.ATR(
                        ticker_data['high'].values,
                        ticker_data['low'].values,
                        ticker_data['close'].values,
                        timeperiod=14
                    )
                
                elif indicator == 'adx':
                    # ADX (Average Directional Movement Index)
                    ticker_data['adx'] = talib.ADX(
                        ticker_data['high'].values,
                        ticker_data['low'].values,
                        ticker_data['close'].values,
                        timeperiod=14
                    )
                
                elif indicator == 'bbands':
                    # Bollinger Bands
                    upperband, middleband, lowerband = talib.BBANDS(
                        ticker_data['close'].values,
                        timeperiod=20,
                        nbdevup=2,
                        nbdevdn=2,
                        matype=0
                    )
                    ticker_data['upperband'] = upperband
                    ticker_data['middleband'] = middleband
                    ticker_data['lowerband'] = lowerband
                
                elif indicator == 'sma':
                    # SMA (Simple Moving Average)
                    ticker_data['sma_5'] = talib.SMA(ticker_data['close'].values, timeperiod=5)
                    ticker_data['sma_10'] = talib.SMA(ticker_data['close'].values, timeperiod=10)
                    ticker_data['sma_20'] = talib.SMA(ticker_data['close'].values, timeperiod=20)
                    ticker_data['sma_50'] = talib.SMA(ticker_data['close'].values, timeperiod=50)
                    ticker_data['sma_100'] = talib.SMA(ticker_data['close'].values, timeperiod=100)
                    ticker_data['sma_200'] = talib.SMA(ticker_data['close'].values, timeperiod=200)
            
            # Update the main DataFrame
            self.data.loc[self.data['ticker'] == ticker] = ticker_data
        
        # Drop rows with NaN values
        self.data = self.data.dropna()
        
        return self.data
    
    def df_to_array(self, tech_indicator_list: List[str], if_vix: bool) -> tuple:
        """
        Convert the DataFrame to arrays.
        
        Args:
            tech_indicator_list (List[str]): The list of technical indicators.
            if_vix (bool): Whether to include VIX.
            
        Returns:
            tuple: The arrays (data_from_processor, price_array, tech_array, time_array).
        """
        if self.data is None or self.data.empty:
            self.logger.warning("No data to convert to arrays")
            return None, None, None, None
        
        self.logger.info("Converting DataFrame to arrays")
        
        # Get unique tickers
        unique_tickers = self.data['ticker'].unique()
        
        # Initialize arrays
        price_array = np.zeros((len(self.data), len(unique_tickers)))
        tech_array = np.zeros((len(self.data), len(tech_indicator_list) * len(unique_tickers)))
        
        # Get time array
        time_array = self.data['date'].unique()
        time_array = pd.to_datetime(time_array)
        
        # Fill arrays
        for i, ticker in enumerate(unique_tickers):
            ticker_data = self.data[self.data['ticker'] == ticker]
            
            # Fill price array
            price_array[:, i] = ticker_data['close'].values
            
            # Fill tech array
            for j, indicator in enumerate(tech_indicator_list):
                if indicator in ticker_data.columns:
                    tech_array[:, i * len(tech_indicator_list) + j] = ticker_data[indicator].values
        
        return self.data, price_array, tech_array, time_array
