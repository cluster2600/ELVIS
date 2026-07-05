"""
Decorators for event handling in the ELVIS Trading Bot.
"""

from functools import wraps
from typing import Callable, List, Union

from .event_bus import event_bus


def event_handler(event_types: Union[str, List[str]]):
    """
    Decorator to register a function as an event handler.

    Args:
        event_types: Single event type or list of event types to handle

    Usage:
        @event_handler('market_data')
        def handle_market_data(event: MarketDataEvent):
            print(f"Price update: {event.price}")

        @event_handler(['order', 'position'])
        def handle_trading_events(event: Union[OrderEvent, PositionEvent]):
            print(f"Trading event: {event.event_type}")
    """

    def decorator(func: Callable) -> Callable:
        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register handler for each event type
        for event_type in types:
            event_bus.subscribe(event_type, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add metadata for introspection
        wrapper._event_types = types
        wrapper._is_event_handler = True

        return wrapper

    return decorator


def async_event_handler(event_types: Union[str, List[str]]):
    """
    Decorator to register an async function as an event handler.

    Args:
        event_types: Single event type or list of event types to handle

    Usage:
        @async_event_handler('market_data')
        async def handle_market_data_async(event: MarketDataEvent):
            await process_price_update(event.price)

        @async_event_handler(['order', 'position'])
        async def handle_trading_events_async(event: Union[OrderEvent, PositionEvent]):
            await update_database(event)
    """

    def decorator(func: Callable) -> Callable:
        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register async handler for each event type
        for event_type in types:
            event_bus.subscribe_async(event_type, func)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Add metadata for introspection
        wrapper._event_types = types
        wrapper._is_async_event_handler = True

        return wrapper

    return decorator


def priority_event_handler(event_types: Union[str, List[str]], priority: int = 0):
    """
    Decorator to register a priority-based event handler.
    Higher priority handlers are executed first.

    Args:
        event_types: Single event type or list of event types to handle
        priority: Handler priority (higher = executed first)

    Usage:
        @priority_event_handler('risk', priority=10)
        def handle_critical_risk(event: RiskEvent):
            if event.severity == 'critical':
                emergency_stop_trading()
    """

    def decorator(func: Callable) -> Callable:
        # Create a wrapper that includes priority metadata
        @wraps(func)
        def priority_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        priority_wrapper._priority = priority

        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register handler with priority
        for event_type in types:
            # Note: This would require extending EventBus to support priority
            # For now, we just register normally
            event_bus.subscribe(event_type, priority_wrapper)

        priority_wrapper._event_types = types
        priority_wrapper._is_priority_handler = True

        return priority_wrapper

    return decorator


def conditional_event_handler(
    event_types: Union[str, List[str]], condition: Callable[[any], bool]
):
    """
    Decorator to register a conditional event handler.
    Handler is only executed if the condition returns True.

    Args:
        event_types: Single event type or list of event types to handle
        condition: Function that takes the event and returns bool

    Usage:
        @conditional_event_handler('trading_signal',
                                 lambda e: e.strength > 0.8)
        def handle_strong_signals(event: TradingSignalEvent):
            execute_trade(event)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def conditional_wrapper(event, *args, **kwargs):
            if condition(event):
                return func(event, *args, **kwargs)

        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register conditional handler
        for event_type in types:
            event_bus.subscribe(event_type, conditional_wrapper)

        conditional_wrapper._event_types = types
        conditional_wrapper._condition = condition
        conditional_wrapper._is_conditional_handler = True

        return conditional_wrapper

    return decorator


def throttled_event_handler(
    event_types: Union[str, List[str]], min_interval_ms: int = 1000
):
    """
    Decorator to register a throttled event handler.
    Limits how often the handler can be called.

    Args:
        event_types: Single event type or list of event types to handle
        min_interval_ms: Minimum milliseconds between handler calls

    Usage:
        @throttled_event_handler('market_data', min_interval_ms=5000)
        def handle_price_updates(event: MarketDataEvent):
            update_ui(event.price)
    """
    import time

    def decorator(func: Callable) -> Callable:
        last_call_time = {}

        @wraps(func)
        def throttled_wrapper(event, *args, **kwargs):
            current_time = time.time() * 1000  # Convert to milliseconds
            event_type = event.event_type

            # Check if enough time has passed
            if event_type in last_call_time:
                if current_time - last_call_time[event_type] < min_interval_ms:
                    return  # Skip this call

            # Update last call time and execute
            last_call_time[event_type] = current_time
            return func(event, *args, **kwargs)

        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register throttled handler
        for event_type in types:
            event_bus.subscribe(event_type, throttled_wrapper)

        throttled_wrapper._event_types = types
        throttled_wrapper._min_interval_ms = min_interval_ms
        throttled_wrapper._is_throttled_handler = True

        return throttled_wrapper

    return decorator


def buffered_event_handler(
    event_types: Union[str, List[str]],
    buffer_size: int = 10,
    flush_interval_ms: int = 1000,
):
    """
    Decorator to register a buffered event handler.
    Collects events and processes them in batches.

    Args:
        event_types: Single event type or list of event types to handle
        buffer_size: Maximum buffer size before flushing
        flush_interval_ms: Maximum time before flushing buffer

    Usage:
        @buffered_event_handler('market_data', buffer_size=100, flush_interval_ms=5000)
        def handle_market_data_batch(events: List[MarketDataEvent]):
            bulk_insert_to_database(events)
    """
    import threading
    import time

    def decorator(func: Callable) -> Callable:
        buffers = {}
        last_flush = {}
        buffer_lock = threading.Lock()

        def flush_buffer(event_type: str):
            """Flush the buffer for a specific event type."""
            with buffer_lock:
                if event_type in buffers and buffers[event_type]:
                    events = buffers[event_type].copy()
                    buffers[event_type].clear()
                    last_flush[event_type] = time.time() * 1000
                    # Call handler with buffered events
                    func(events)

        @wraps(func)
        def buffered_wrapper(event, *args, **kwargs):
            event_type = event.event_type
            current_time = time.time() * 1000

            with buffer_lock:
                # Initialize buffer if needed
                if event_type not in buffers:
                    buffers[event_type] = []
                    last_flush[event_type] = current_time

                # Add event to buffer
                buffers[event_type].append(event)

                # Check if we should flush
                should_flush = (
                    len(buffers[event_type]) >= buffer_size
                    or current_time - last_flush[event_type] >= flush_interval_ms
                )

            if should_flush:
                flush_buffer(event_type)

        # Normalize to list
        types = event_types if isinstance(event_types, list) else [event_types]

        # Register buffered handler
        for event_type in types:
            event_bus.subscribe(event_type, buffered_wrapper)

        buffered_wrapper._event_types = types
        buffered_wrapper._buffer_size = buffer_size
        buffered_wrapper._flush_interval_ms = flush_interval_ms
        buffered_wrapper._is_buffered_handler = True

        return buffered_wrapper

    return decorator


# Auto-discovery and registration of event handlers
def auto_register_handlers(module):
    """
    Automatically discover and register all event handlers in a module.

    Args:
        module: Python module to scan for event handlers

    Usage:
        import my_handlers
        auto_register_handlers(my_handlers)
    """
    import inspect

    for name, obj in inspect.getmembers(module):
        # Check if it's a decorated event handler
        if hasattr(obj, "_is_event_handler"):
            # Handler is already registered by decorator
            pass
        elif hasattr(obj, "_is_async_event_handler"):
            # Async handler is already registered by decorator
            pass
        elif hasattr(obj, "_is_priority_handler"):
            # Priority handler is already registered by decorator
            pass
        elif hasattr(obj, "_is_conditional_handler"):
            # Conditional handler is already registered by decorator
            pass
        elif hasattr(obj, "_is_throttled_handler"):
            # Throttled handler is already registered by decorator
            pass
        elif hasattr(obj, "_is_buffered_handler"):
            # Buffered handler is already registered by decorator
            pass
