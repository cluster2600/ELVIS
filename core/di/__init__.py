"""
Dependency Injection module for ELVIS Trading Bot.
Provides centralized dependency management and decoupling of components.
"""

from .container import Container, container
from .providers import (
    SingletonProvider,
    FactoryProvider,
    ConfigurationProvider
)

__all__ = [
    'Container',
    'container',
    'SingletonProvider',
    'FactoryProvider',
    'ConfigurationProvider'
]
