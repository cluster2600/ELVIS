"""Documented import path for the risk manager.

Docs reference `trading/risk/advanced_risk_manager.py`; the class actually
lives in `risk_manager.py`. This module re-exports it so both paths work.
"""
from trading.risk.risk_manager import AdvancedRiskManager

__all__ = ["AdvancedRiskManager"]
