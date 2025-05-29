"""
Event-driven architecture module for ELVIS Trading Bot.
Provides event bus, event types, and event handling mechanisms.
"""

from .event_bus import EventBus, Event, EventHandler
from .event_types import (
    MarketDataEvent,
    TradingSignalEvent,
    OrderEvent,
    RiskEvent,
    SystemEvent
)
from .decorators import event_handler, async_event_handler

__all__ = [
    'EventBus',
    'Event',
    'EventHandler',
    'MarketDataEvent',
    'TradingSignalEvent',
    'OrderEvent',
    'RiskEvent',
    'SystemEvent',
    'event_handler',
    'async_event_handler'
]
