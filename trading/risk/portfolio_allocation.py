"""Multi-asset allocation, risk parity, and rebalancing helpers.

Pure numpy — no scipy — so it works on Python 3.14.

- inverse_volatility_weights: weights proportional to 1/sigma (normalized).
- risk_parity_weights: iterative equal-risk-contribution (ERC) allocation.
- rebalance_orders: given current $ positions, target weights, and equity,
  emit the buy/sell orders to reach the target (ignoring dust below min_trade).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def inverse_volatility_weights(returns) -> np.ndarray:
    """Weights ∝ 1/volatility, normalized to sum to 1.

    `returns` is a 2D array-like (rows=observations, cols=assets) or a
    pandas DataFrame.
    """
    arr = np.asarray(getattr(returns, "values", returns), dtype=float)
    vol = arr.std(axis=0, ddof=1)
    vol = np.where(vol <= 0, np.nan, vol)
    inv = 1.0 / vol
    inv = np.nan_to_num(inv, nan=0.0)
    total = inv.sum()
    if total <= 0:
        n = len(inv)
        return np.full(n, 1.0 / n)
    return inv / total


def risk_parity_weights(cov, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Equal-risk-contribution weights for a covariance matrix (numpy only).

    Iteratively rescales weights toward equal marginal risk contributions.
    Returns weights summing to 1.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    target = 1.0 / n
    for _ in range(iters):
        m = cov @ w  # marginal contributions
        port_var = float(w @ m)
        if port_var <= 0:
            break
        rc = w * m / port_var  # risk contribution shares (sum to 1)
        # nudge each weight toward equal risk share
        w = w * (target / np.where(rc <= 0, tol, rc)) ** 0.5
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            w = np.full(n, 1.0 / n)
            break
        w = w / s
        if np.max(np.abs(rc - target)) < tol:
            break
    return w


def rebalance_orders(
    current_positions: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
    min_trade_usd: float = 10.0,
) -> List[Dict[str, object]]:
    """Orders to move current $ positions to target weights * equity.

    Returns a list of {'symbol','side','usd_amount'}; trades smaller than
    min_trade_usd are skipped (no dust churn).
    """
    orders: List[Dict[str, object]] = []
    symbols = set(current_positions) | set(target_weights)
    for sym in sorted(symbols):
        cur = float(current_positions.get(sym, 0.0))
        tgt = float(target_weights.get(sym, 0.0)) * equity
        delta = tgt - cur
        if abs(delta) < min_trade_usd:
            continue
        orders.append(
            {
                "symbol": sym,
                "side": "BUY" if delta > 0 else "SELL",
                "usd_amount": round(abs(delta), 2),
            }
        )
    return orders
