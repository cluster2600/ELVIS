"""
Dependency Injection Container for managing application dependencies.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
import logging
from .providers import Provider, SingletonProvider, FactoryProvider, ConfigurationProvider


T = TypeVar('T')


class Container:
    """
    Central dependency injection container for managing application dependencies.
    Supports registration of various provider types and dependency resolution.
    """
    
    def __init__(self):
        """Initialize the container."""
        self._providers: Dict[str, Provider] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_singleton(self, name: str, factory: Callable[[], T]) -> None:
        """
        Register a singleton dependency.
        
        Args:
            name: Name to register the dependency under
            factory: Factory function that creates the instance
        """
        self._providers[name] = SingletonProvider(factory)
        self._logger.debug(f"Registered singleton: {name}")
    
    def register_factory(self, name: str, factory: Callable[[], T]) -> None:
        """
        Register a factory dependency (new instance each time).
        
        Args:
            name: Name to register the dependency under
            factory: Factory function that creates new instances
        """
        self._providers[name] = FactoryProvider(factory)
        self._logger.debug(f"Registered factory: {name}")
    
    def register_configuration(self, name: str, config: Dict[str, Any]) -> None:
        """
        Register configuration values.
        
        Args:
            name: Name to register the configuration under
            config: Configuration dictionary
        """
        self._providers[name] = ConfigurationProvider(config)
        self._logger.debug(f"Registered configuration: {name}")
    
    def register_provider(self, name: str, provider: Provider) -> None:
        """
        Register a custom provider.
        
        Args:
            name: Name to register the provider under
            provider: Provider instance
        """
        self._providers[name] = provider
        self._logger.debug(f"Registered custom provider: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get a dependency from the container.
        
        Args:
            name: Name of the dependency
            
        Returns:
            The resolved dependency
            
        Raises:
            KeyError: If dependency is not registered
        """
        if name not in self._providers:
            raise KeyError(f"Dependency '{name}' not registered")
        
        return self._providers[name].get()
    
    def get_optional(self, name: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a dependency from the container, returning default if not found.
        
        Args:
            name: Name of the dependency
            default: Default value if dependency not found
            
        Returns:
            The resolved dependency or default value
        """
        try:
            return self.get(name)
        except KeyError:
            return default
    
    def has(self, name: str) -> bool:
        """
        Check if a dependency is registered.
        
        Args:
            name: Name of the dependency
            
        Returns:
            True if dependency is registered
        """
        return name in self._providers
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset one or all providers.
        
        Args:
            name: Name of specific provider to reset, or None for all
        """
        if name:
            if name in self._providers:
                self._providers[name].reset()
                self._logger.debug(f"Reset provider: {name}")
        else:
            for provider_name, provider in self._providers.items():
                provider.reset()
                self._logger.debug(f"Reset provider: {provider_name}")
    
    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._logger.debug("Cleared all providers")
    
    def get_all_names(self) -> list:
        """Get list of all registered dependency names."""
        return list(self._providers.keys())


# Global container instance
container = Container()


# Decorator for dependency injection
def inject(**dependencies):
    """
    Decorator for injecting dependencies into functions or methods.
    
    Usage:
        @inject(logger='logger', config='trading_config')
        def my_function(logger, config):
            # logger and config are automatically injected
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject dependencies
            for param_name, dep_name in dependencies.items():
                if param_name not in kwargs:
                    kwargs[param_name] = container.get(dep_name)
            return func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        
        return wrapper
    return decorator


# Context manager for scoped dependencies
class DependencyScope:
    """
    Context manager for creating scoped dependencies.
    Useful for request-scoped or transaction-scoped dependencies.
    """
    
    def __init__(self, container: Container):
        """
        Initialize dependency scope.
        
        Args:
            container: Container instance
        """
        self._container = container
        self._scoped_providers: Dict[str, Provider] = {}
        self._original_providers: Dict[str, Provider] = {}
    
    def register_scoped(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a scoped dependency.
        
        Args:
            name: Name of the dependency
            factory: Factory function
        """
        from .providers import ScopedProvider
        self._scoped_providers[name] = ScopedProvider(factory)
    
    def __enter__(self):
        """Enter the scope context."""
        # Replace providers with scoped ones
        for name, provider in self._scoped_providers.items():
            if self._container.has(name):
                self._original_providers[name] = self._container._providers[name]
            self._container.register_provider(name, provider)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the scope context."""
        # Restore original providers
        for name, provider in self._original_providers.items():
            self._container.register_provider(name, provider)
        
        # Reset scoped providers
        for provider in self._scoped_providers.values():
            provider.reset()
