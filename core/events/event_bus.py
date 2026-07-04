"""
Event Bus implementation for event-driven architecture.
"""

import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event(ABC):
    """Base class for all events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type identifier."""
        pass


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle the event."""
        pass

    @property
    @abstractmethod
    def event_types(self) -> List[str]:
        """Return list of event types this handler processes."""
        pass


class EventBus:
    """
    Central event bus for publishing and subscribing to events.
    Supports both synchronous and asynchronous event handling.
    """

    def __init__(self, max_workers: int = 10, enable_async: bool = True):
        """
        Initialize the event bus.

        Args:
            max_workers: Maximum number of worker threads for async handling
            enable_async: Enable asynchronous event processing
        """
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._subscribers: Dict[str, Set[EventHandler]] = defaultdict(set)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._enable_async = enable_async
        self._event_queue: asyncio.Queue = None
        self._running = False
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = 1000

    def subscribe(
        self, event_type: str, handler: Union[Callable, EventHandler]
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler function or EventHandler instance
        """
        with self._lock:
            if isinstance(handler, EventHandler):
                self._subscribers[event_type].add(handler)
            else:
                self._handlers[event_type].append(handler)
            self._logger.debug(f"Subscribed handler to {event_type}")

    def subscribe_async(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe an async handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async handler function
        """
        with self._lock:
            self._async_handlers[event_type].append(handler)
            self._logger.debug(f"Subscribed async handler to {event_type}")

    def unsubscribe(
        self, event_type: str, handler: Union[Callable, EventHandler]
    ) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        with self._lock:
            if isinstance(handler, EventHandler):
                self._subscribers[event_type].discard(handler)
            else:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)
                if handler in self._async_handlers[event_type]:
                    self._async_handlers[event_type].remove(handler)
            self._logger.debug(f"Unsubscribed handler from {event_type}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Add to history
        self._add_to_history(event)

        # Get all handlers for this event type
        handlers = self._get_handlers(event.event_type)
        async_handlers = self._get_async_handlers(event.event_type)
        subscribers = self._get_subscribers(event.event_type)

        # Handle synchronous handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.error(f"Error in handler for {event.event_type}: {e}")

        # Handle EventHandler instances
        for subscriber in subscribers:
            try:
                subscriber.handle(event)
            except Exception as e:
                self._logger.error(f"Error in subscriber for {event.event_type}: {e}")

        # Handle async handlers
        if self._enable_async and async_handlers:
            if self._event_queue:
                asyncio.create_task(self._event_queue.put((event, async_handlers)))
            else:
                # Run in executor if no event loop
                for handler in async_handlers:
                    try:
                        self._executor.submit(self._run_async_handler, handler, event)
                    except RuntimeError:
                        # Executor already shut down (e.g. event published from
                        # the shutdown signal handler) - drop the event.
                        self._logger.debug(
                            f"Dropped {event.event_type}: executor shut down"
                        )

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.

        Args:
            event: Event to publish
        """
        # Add to history
        self._add_to_history(event)

        # Get handlers
        handlers = self._get_handlers(event.event_type)
        async_handlers = self._get_async_handlers(event.event_type)
        subscribers = self._get_subscribers(event.event_type)

        # Create tasks for all handlers
        tasks = []

        # Sync handlers in executor
        for handler in handlers:
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(self._executor, handler, event)
            )
            tasks.append(task)

        # EventHandler instances in executor
        for subscriber in subscribers:
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self._executor, subscriber.handle, event
                )
            )
            tasks.append(task)

        # Async handlers directly
        for handler in async_handlers:
            task = asyncio.create_task(handler(event))
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_handlers(self, event_type: str) -> List[Callable]:
        """Get all sync handlers for an event type."""
        with self._lock:
            return list(self._handlers[event_type])

    def _get_async_handlers(self, event_type: str) -> List[Callable]:
        """Get all async handlers for an event type."""
        with self._lock:
            return list(self._async_handlers[event_type])

    def _get_subscribers(self, event_type: str) -> Set[EventHandler]:
        """Get all EventHandler subscribers for an event type."""
        with self._lock:
            return set(self._subscribers[event_type])

    def _run_async_handler(self, handler: Callable, event: Event) -> None:
        """Run an async handler in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handler(event))
        except Exception as e:
            self._logger.error(f"Error in async handler: {e}")
        finally:
            loop.close()

    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size limit."""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

    def get_event_history(
        self, event_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self._lock:
            history = self._event_history.copy()

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        if limit:
            history = history[-limit:]

        return history

    async def start_async_processor(self) -> None:
        """Start the async event processor."""
        self._event_queue = asyncio.Queue()
        self._running = True

        while self._running:
            try:
                event, handlers = await self._event_queue.get()

                # Process all async handlers
                tasks = [handler(event) for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in async processor: {e}")

    def stop(self) -> None:
        """Stop the event bus and cleanup resources."""
        self._running = False
        self._executor.shutdown(wait=True)
        self._logger.info("Event bus stopped")

    def clear(self) -> None:
        """Clear all handlers and subscribers."""
        with self._lock:
            self._handlers.clear()
            self._async_handlers.clear()
            self._subscribers.clear()
            self._event_history.clear()
        self._logger.info("Event bus cleared")


# Global event bus instance
event_bus = EventBus()
