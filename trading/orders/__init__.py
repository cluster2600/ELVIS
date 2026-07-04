# Trading Orders Module
"""
Advanced Order Types for Trading System

This module provides sophisticated order types for professional trading:
- OCO (One-Cancels-Other): Combines limit and stop orders
- Iceberg: Breaks large orders into smaller chunks to hide size
- TWAP (Time-Weighted Average Price): Executes orders over time periods
"""

from .advanced_order_manager import AdvancedOrderManager
from .base_order import (
    BaseOrder,
    OrderSide,
    OrderStatus,
    OrderType,
    SimpleOrder,
    TimeInForce,
)
from .iceberg_order import IcebergOrder, IcebergOrderManager
from .oco_order import OCOOrder, OCOOrderManager
from .twap_order import TWAPOrder, TWAPOrderManager

__all__ = [
    # Base classes and enums
    "BaseOrder",
    "SimpleOrder",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    # OCO Orders
    "OCOOrder",
    "OCOOrderManager",
    # Iceberg Orders
    "IcebergOrder",
    "IcebergOrderManager",
    # TWAP Orders
    "TWAPOrder",
    "TWAPOrderManager",
    # Unified Manager
    "AdvancedOrderManager",
]
