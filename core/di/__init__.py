"""
Dependency Injection module for ELVIS Trading Bot.
Provides centralized dependency management and decoupling of components.
"""

from .container import Container, container
from .providers import ConfigurationProvider, FactoryProvider, SingletonProvider

__all__ = [
    "Container",
    "container",
    "SingletonProvider",
    "FactoryProvider",
    "ConfigurationProvider",
]
