"""
Event-driven architecture module for ELVIS Trading Bot.
Provides event bus, event types, and event handling mechanisms.
"""

from .decorators import async_event_handler, event_handler
from .event_bus import Event, EventBus, EventHandler, event_bus
from .event_types import (
    MarketDataEvent,
    OrderEvent,
    RiskEvent,
    SystemEvent,
    TradingSignalEvent,
)

__all__ = [
    "EventBus",
    "Event",
    "EventHandler",
    "event_bus",
    "MarketDataEvent",
    "TradingSignalEvent",
    "OrderEvent",
    "RiskEvent",
    "SystemEvent",
    "event_handler",
    "async_event_handler",
]
