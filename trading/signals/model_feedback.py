"""Per-model accuracy feedback for the adaptive ensemble (roadmap item #11).

How it works
------------
1. At trade entry ``record_entry`` stores every ensemble member's vote
   (``np.model_predictions`` via :mod:`utils.paper_trade_db`).
2. Each cycle ``score_closed_trades`` looks for the first trade of the same
   symbol carrying realized PnL after the votes were recorded (the close of
   that position) and scores each member: a model that voted the executed
   direction was right iff pnl > 0; a model that voted the opposite direction
   was right iff pnl < 0; HOLD votes are not scored.
3. Every right/wrong outcome EMA-updates the shared
   :class:`~trading.signals.adaptive_ensemble.AdaptiveEnsembleWeights`
   (persisted to ``models/adaptive_ensemble_weights.json``), which
   ``EnsembleStrategy`` reads to modulate its static source weights.

Weights start uniform, so behavior only drifts from real, scored trades —
never from assumed accuracies.

How to use
----------
Wired automatically in main.py behind ``ELVIS_ADAPTIVE_ENSEMBLE`` (default on).
Tuning knobs: ``ELVIS_ADAPTIVE_ALPHA`` (EMA weight of the newest outcome,
default 0.1 — one trade moves a model's accuracy estimate by at most 10%) and
``ELVIS_ADAPTIVE_STATE`` (weights file path).
"""

import logging
import os
from typing import Dict, Optional

from trading.signals.adaptive_ensemble import AdaptiveEnsembleWeights
from utils import paper_trade_db as _db

logger = logging.getLogger(__name__)

_shared_weights: Optional[AdaptiveEnsembleWeights] = None


def get_shared_weights() -> AdaptiveEnsembleWeights:
    """Process-wide AdaptiveEnsembleWeights, persisted across restarts."""
    global _shared_weights
    if _shared_weights is None:
        _shared_weights = AdaptiveEnsembleWeights(
            initial_accuracies={},
            ema_alpha=float(os.getenv("ELVIS_ADAPTIVE_ALPHA", "0.1")),
            state_path=os.getenv(
                "ELVIS_ADAPTIVE_STATE", "models/adaptive_ensemble_weights.json"
            ),
        )
    return _shared_weights


def record_entry(symbol: str, side: str, votes: Dict[str, str]) -> bool:
    """Record each model's vote for the position just opened."""
    directional = {m: v for m, v in votes.items() if v in ("BUY", "SELL", "HOLD")}
    if not directional:
        return False
    ok = _db.record_model_predictions(symbol, side, directional)
    if ok:
        logger.info(f"🧠 Recorded {len(directional)} model votes for {symbol} {side}")
    return ok


def score_closed_trades() -> int:
    """Score unscored votes against realized PnL; returns votes scored.

    Batches (same symbol + created_at) whose position has not closed yet are
    left unscored for a later cycle. pnl == 0 closes are marked scored without
    updating accuracies (no information).
    """
    rows = _db.get_unscored_predictions()
    if not rows:
        return 0

    weights = get_shared_weights()
    batches: Dict[tuple, list] = {}
    for row_id, created_at, symbol, side, model, vote in rows:
        batches.setdefault((symbol, created_at, side), []).append((row_id, model, vote))

    scored_votes = 0
    scored_ids = []
    updated = False
    for (symbol, created_at, side), members in batches.items():
        closing = _db.get_first_closing_trade_after(symbol, created_at)
        if closing is None:
            continue  # position still open; keep for a later cycle
        _, pnl = closing
        for row_id, model, vote in members:
            scored_ids.append(row_id)
            if vote == "HOLD" or pnl == 0:
                continue
            voted_executed_direction = vote == side
            correct = voted_executed_direction == (pnl > 0)
            weights.update(model, 1.0 if correct else 0.0)
            updated = True
            scored_votes += 1
            logger.debug(
                f"🧠 {model} vote {vote} on {symbol} ({side}, pnl {pnl:+.4f}): "
                f"{'right' if correct else 'wrong'}"
            )

    if updated:
        weights.save()
        logger.info(
            f"🧠 Adaptive ensemble scored {scored_votes} votes; "
            f"weights now {dict(sorted(weights.weights.items()))}"
        )
    if scored_ids:
        _db.mark_predictions_scored(scored_ids)
    return scored_votes
