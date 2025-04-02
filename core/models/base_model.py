"""
Base model interface for the BTC_BOT project.
This module defines the interface that all models must implement.
"""

from abc import ABC, abstractmethod
import os
import pickle
import joblib
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Base class for all trading models."""
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model.
        
        Args:
            X: The features.
            y: The target.
        """
        pass
        
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: The features.
            
        Returns:
            np.ndarray: The predictions.
        """
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: The path to save the model to.
        """
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: The path to load the model from.
            
        Returns:
            BaseModel: The loaded model.
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: The model parameters.
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Args:
            **params: The model parameters.
        """
        pass
