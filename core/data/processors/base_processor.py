"""
Base processor class for the BTC_BOT project.
This module defines the interface that all data processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional
import pandas as pd
import numpy as np
import logging

class BaseProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, logger: logging.Logger, **kwargs):
        """
        Initialize the processor.
        
        Args:
            data_source (str): The data source.
            start_date (str): The start date.
            end_date (str): The end date.
            time_interval (str): The time interval.
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.logger = logger
        self.kwargs = kwargs
        self.data = None
    
    @abstractmethod
    def download_data(self, ticker_list: List[str]) -> pd.DataFrame:
        """
        Download data for the specified tickers.
        
        Args:
            ticker_list (List[str]): The list of tickers to download data for.
            
        Returns:
            pd.DataFrame: The downloaded data.
        """
        pass
    
    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the downloaded data.
        
        Returns:
            pd.DataFrame: The cleaned data.
        """
        pass
    
    @abstractmethod
    def add_technical_indicator(self, tech_indicator_list: List[str]) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            tech_indicator_list (List[str]): The list of technical indicators to add.
            
        Returns:
            pd.DataFrame: The data with technical indicators.
        """
        pass
    
    @abstractmethod
    def df_to_array(self, tech_indicator_list: List[str], if_vix: bool) -> tuple:
        """
        Convert the DataFrame to arrays.
        
        Args:
            tech_indicator_list (List[str]): The list of technical indicators.
            if_vix (bool): Whether to include VIX.
            
        Returns:
            tuple: The arrays.
        """
        pass
    
    def run(self, ticker_list: List[str], technical_indicator_list: List[str], if_vix: bool = False) -> tuple:
        """
        Run the processor.
        
        Args:
            ticker_list (List[str]): The list of tickers.
            technical_indicator_list (List[str]): The list of technical indicators.
            if_vix (bool, optional): Whether to include VIX. Defaults to False.
            
        Returns:
            tuple: The processed data.
        """
        self.logger.info(f"Running processor for {ticker_list} from {self.start_date} to {self.end_date}")
        
        # Download data
        self.data = self.download_data(ticker_list)
        
        # Clean data
        self.data = self.clean_data()
        
        # Add technical indicators
        self.data = self.add_technical_indicator(technical_indicator_list)
        
        # Convert to arrays
        return self.df_to_array(technical_indicator_list, if_vix)
