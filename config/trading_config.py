"""Loader for the unified trading_config.yaml documented in docs/trading_system.md."""
import os
import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_config.yaml")


def load_trading_config(path: str = None) -> dict:
    """Load and return the unified trading config as a dict.

    Applies the DEFAULT_LEVERAGE env override so the loaded config matches the
    runtime behaviour enforced by config.config.validate_leverage_config.
    """
    with open(path or _DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    lev = os.getenv("DEFAULT_LEVERAGE")
    if lev is not None and "trading" in cfg:
        cfg["trading"]["default_leverage"] = int(lev)
    return cfg
