# Trading Orders Module
"""
Advanced Order Types for Trading System

This module provides sophisticated order types for professional trading:
- OCO (One-Cancels-Other): Combines limit and stop orders
- Iceberg: Breaks large orders into smaller chunks to hide size
- TWAP (Time-Weighted Average Price): Executes orders over time periods
"""

from .base_order import (
    BaseOrder,
    SimpleOrder,
    OrderStatus,
    OrderSide,
    OrderType,
    TimeInForce
)

from .oco_order import (
    OCOOrder,
    OCOOrderManager
)

from .iceberg_order import (
    IcebergOrder,
    IcebergOrderManager
)

from .twap_order import (
    TWAPOrder,
    TWAPOrderManager
)

from .advanced_order_manager import AdvancedOrderManager

__all__ = [
    # Base classes and enums
    'BaseOrder',
    'SimpleOrder',
    'OrderStatus',
    'OrderSide',
    'OrderType',
    'TimeInForce',
    
    # OCO Orders
    'OCOOrder',
    'OCOOrderManager',
    
    # Iceberg Orders
    'IcebergOrder',
    'IcebergOrderManager',
    
    # TWAP Orders
    'TWAPOrder',
    'TWAPOrderManager',
    
    # Unified Manager
    'AdvancedOrderManager'
]