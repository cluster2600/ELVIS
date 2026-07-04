"""
Base Order Classes
Defines the foundation for all order types in the trading system
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    OCO = "oco"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class TimeInForce(Enum):
    """Time in force enumeration"""

    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class BaseOrder(ABC):
    """
    Base class for all order types
    Provides common functionality and interface
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        exchange: str = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize base order

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Type of order
            price: Order price (None for market orders)
            time_in_force: Time in force
            exchange: Target exchange
            logger: Logger instance
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.original_quantity = quantity
        self.order_type = order_type
        self.price = price
        self.time_in_force = time_in_force
        self.exchange = exchange

        # Order state
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.remaining_quantity = quantity
        self.average_fill_price = 0.0
        self.total_fees = 0.0

        # Timestamps
        self.created_at = datetime.now()
        self.submitted_at = None
        self.filled_at = None
        self.cancelled_at = None
        self.expires_at = None

        # Execution tracking
        self.fills = []  # List of fill records
        self.exchange_order_id = None
        self.error_message = None

        # Metadata
        self.metadata = {}
        self.logger = logger

        if self.time_in_force == TimeInForce.GTD:
            self.expires_at = self.created_at + timedelta(days=1)  # Default 1 day

    @abstractmethod
    def validate(self) -> bool:
        """Validate order parameters"""
        pass

    @abstractmethod
    def to_exchange_format(self) -> Dict[str, Any]:
        """Convert order to exchange-specific format"""
        pass

    def update_status(self, status: OrderStatus, message: str = None):
        """Update order status"""
        old_status = self.status
        self.status = status

        if status == OrderStatus.SUBMITTED and not self.submitted_at:
            self.submitted_at = datetime.now()
        elif status == OrderStatus.FILLED and not self.filled_at:
            self.filled_at = datetime.now()
        elif status == OrderStatus.CANCELLED and not self.cancelled_at:
            self.cancelled_at = datetime.now()

        if message:
            self.error_message = message

        if self.logger:
            self.logger.info(
                f"Order {self.id} status: {old_status.value} -> {status.value}"
            )

    def add_fill(
        self,
        quantity: float,
        price: float,
        fee: float = 0.0,
        fill_id: str = None,
        timestamp: datetime = None,
    ):
        """Add a fill record"""
        fill = {
            "id": fill_id or str(uuid.uuid4()),
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "timestamp": timestamp or datetime.now(),
        }

        self.fills.append(fill)
        self.filled_quantity += quantity
        self.remaining_quantity = max(0, self.original_quantity - self.filled_quantity)
        self.total_fees += fee

        # Update average fill price
        total_value = sum(f["quantity"] * f["price"] for f in self.fills)
        self.average_fill_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        )

        # Update status
        if self.remaining_quantity <= 0.001:  # Account for floating point precision
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)

    def cancel(self, reason: str = "User cancelled"):
        """Cancel the order"""
        if self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]:
            self.update_status(OrderStatus.CANCELLED, reason)
            return True
        return False

    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def is_expired(self) -> bool:
        """Check if order has expired"""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    def get_fill_percentage(self) -> float:
        """Get percentage of order filled"""
        if self.original_quantity == 0:
            return 0.0
        return (self.filled_quantity / self.original_quantity) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "original_quantity": self.original_quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "time_in_force": self.time_in_force.value,
            "exchange": self.exchange,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "total_fees": self.total_fees,
            "created_at": self.created_at.isoformat(),
            "submitted_at": (
                self.submitted_at.isoformat() if self.submitted_at else None
            ),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": (
                self.cancelled_at.isoformat() if self.cancelled_at else None
            ),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "exchange_order_id": self.exchange_order_id,
            "error_message": self.error_message,
            "fills": self.fills,
            "metadata": self.metadata,
            "fill_percentage": self.get_fill_percentage(),
        }

    def __str__(self) -> str:
        """String representation of order"""
        return (
            f"{self.order_type.value.upper()} {self.side.value.upper()} "
            f"{self.quantity} {self.symbol} @ {self.price or 'MARKET'} "
            f"({self.status.value})"
        )

    def __repr__(self) -> str:
        """Detailed representation of order"""
        return f"<{self.__class__.__name__} {self.id}: {str(self)}>"


class SimpleOrder(BaseOrder):
    """Simple market/limit order implementation"""

    def validate(self) -> bool:
        """Validate simple order parameters"""
        if self.quantity <= 0:
            return False

        if self.order_type == OrderType.LIMIT and (not self.price or self.price <= 0):
            return False

        if self.order_type == OrderType.MARKET and self.price is not None:
            # Market orders shouldn't have price
            self.price = None

        return True

    def to_exchange_format(self) -> Dict[str, Any]:
        """Convert to exchange format"""
        order_data = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": "market" if self.order_type == OrderType.MARKET else "limit",
            "quantity": self.quantity,
            "timeInForce": self.time_in_force.value.upper(),
        }

        if self.price and self.order_type == OrderType.LIMIT:
            order_data["price"] = self.price

        return order_data
