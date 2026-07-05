"""
Dependency injection providers for different object lifecycle management.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type


class Provider(ABC):
    """Base provider interface for dependency injection."""

    @abstractmethod
    def get(self) -> Any:
        """Get the provided instance."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the provider state."""
        pass


class SingletonProvider(Provider):
    """
    Provides a singleton instance of a class.
    Thread-safe implementation ensures only one instance exists.
    """

    def __init__(self, factory: Callable[[], Any]):
        """
        Initialize singleton provider.

        Args:
            factory: Callable that creates the instance
        """
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Get or create the singleton instance."""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def reset(self):
        """Reset the singleton instance."""
        with self._lock:
            self._instance = None


class FactoryProvider(Provider):
    """
    Provides a new instance each time get() is called.
    Useful for non-singleton dependencies.
    """

    def __init__(self, factory: Callable[[], Any]):
        """
        Initialize factory provider.

        Args:
            factory: Callable that creates new instances
        """
        self._factory = factory

    def get(self) -> Any:
        """Create and return a new instance."""
        return self._factory()

    def reset(self):
        """Factory provider has no state to reset."""
        pass


class ConfigurationProvider(Provider):
    """
    Provides configuration values from various sources.
    Supports environment variables, config files, and defaults.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration provider.

        Args:
            config_dict: Dictionary of configuration values
        """
        self._config = config_dict.copy()

    def get(self) -> Dict[str, Any]:
        """Return the configuration dictionary."""
        return self._config.copy()

    def reset(self):
        """Reset is not applicable for configuration."""
        pass

    def update(self, updates: Dict[str, Any]):
        """Update configuration values."""
        self._config.update(updates)


class LazyProvider(Provider):
    """
    Provides lazy initialization of dependencies.
    Instance is created only when first accessed.
    """

    def __init__(self, factory: Callable[[], Any]):
        """
        Initialize lazy provider.

        Args:
            factory: Callable that creates the instance when needed
        """
        self._factory = factory
        self._instance = None
        self._initialized = False
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Get the lazily initialized instance."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._instance = self._factory()
                    self._initialized = True
        return self._instance

    def reset(self):
        """Reset the lazy instance."""
        with self._lock:
            self._instance = None
            self._initialized = False


class ScopedProvider(Provider):
    """
    Provides scoped instances that exist within a specific context.
    Useful for request-scoped or session-scoped dependencies.
    """

    def __init__(self, factory: Callable[[], Any]):
        """
        Initialize scoped provider.

        Args:
            factory: Callable that creates instances
        """
        self._factory = factory
        self._instances = threading.local()

    def get(self) -> Any:
        """Get or create instance for current scope."""
        if not hasattr(self._instances, "value"):
            self._instances.value = self._factory()
        return self._instances.value

    def reset(self):
        """Reset the scoped instance for current thread."""
        if hasattr(self._instances, "value"):
            delattr(self._instances, "value")
