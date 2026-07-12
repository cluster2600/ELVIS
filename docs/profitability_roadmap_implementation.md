# Profitability Roadmap — Implementation Status

Implementation of the 15 ideas in [`profitability_roadmap.md`](./profitability_roadmap.md).
Every mechanism below exists in code, is unit-tested, and (where it belongs in
the live loop) is wired into `main.py` behind an environment flag.

> **Honesty note:** the roadmap's win-rate and revenue projections
> (35% → 75%+, $1k → $25k/month) are the roadmap author's estimates, **not
> verified results**. What this implementation delivers is the *mechanisms*,
> each individually tested. Measure impact in paper trading before believing
> any number.

## Status by item

| # | Idea | Module / API | Wired | Flag (default) |
|---|------|--------------|-------|----------------|
| 1 | Market regime detector | `trading/analysis/market_regime_detector.MarketRegimeDetector` | signal loop (pre-existing) + regime cache feeds #5/#10 | always on with winrate filter |
| 2 | RSI overbought/oversold filter | `trading.signals.filters.rsi_gate` | signal gates | `ELVIS_ROADMAP_FILTERS` (on) |
| 3 | Volume-based trade sizing | `trading.risk.position_sizing.volume_multiplier` | sizing block | `ELVIS_VOLUME_SIZING` (on) |
| 4 | Trailing stop loss | `trading.execution.exits.TrailingStop` | position loop | `ELVIS_TRAILING_STOP` (on), `ELVIS_TRAIL_PCT` (0.02) |
| 5 | Fee optimization (all-in cost gate) | `trading.fees.fee_gate.is_trade_viable` | pre-execution | `ELVIS_FEE_GATE` (on) |
| 6 | Momentum confirmation | `trading.signals.filters.has_momentum` | signal gates | `ELVIS_ROADMAP_FILTERS` (on) |
| 7 | Bollinger Band squeeze | `trading.signals.filters.detect_bb_squeeze` | signal gates | `ELVIS_ROADMAP_FILTERS` (on) |
| 8 | Time-of-day filter | `trading.signals.filters.is_optimal_trading_hour` | signal gates | `ELVIS_ROADMAP_FILTERS` (on) |
| 9 | MACD histogram divergence | `trading.signals.filters.detect_macd_divergence` | signal gates (veto; override opt-in) | `ELVIS_ROADMAP_FILTERS` (on) |
| 10 | Dynamic take profit by regime | `trading.execution.exits.dynamic_take_profit` | position loop (replaces fixed $8) | `ELVIS_DYNAMIC_TP` (on) |
| 11 | Adaptive ML ensemble weights | `trading.signals.adaptive_ensemble` + `trading.signals.model_feedback` | ensemble voting + feedback loop (see below) | `ELVIS_ADAPTIVE_ENSEMBLE` (on) |
| 12 | Order flow analysis | `trading.signals.order_flow.confirm_signal_with_flow` | signal gates | `ELVIS_ORDER_FLOW` (**off**) |
| 13 | Kelly criterion sizing | `trading.risk.position_sizing.kelly_fraction` / `kelly_from_trades` | sizing block (caps size) | `ELVIS_KELLY_SIZING` (**off**) |
| 14 | Multi-timeframe analysis | `trading.signals.mtf.MTFAnalyzer` | signal gates | `ELVIS_MTF` (**off**) |
| 15 | Walk-forward optimization | `trading.optimization.walk_forward.WalkForwardOptimizer` | **offline tool** (by design) | — |

## How the live wiring works

All integration lives in `main.py`, mirroring the existing lazy-import pattern,
and every stage is wrapped in try/except so a gate failure logs and never kills
the loop.

1. **Signal gates** — after the existing high-win-rate filter, a
   `PROFITABILITY-ROADMAP SIGNAL GATES` block applies
   `apply_signal_filters` (RSI, momentum persistence, BB squeeze,
   trading hours, MACD divergence — each toggleable via its config key), then
   optional order-flow confirmation and MTF alignment. Any veto downgrades the
   signal to HOLD and logs the reason.
2. **Regime cache** — the regime detected for each symbol is cached on
   `main._last_regime` and reused by the dynamic-TP and fee-gate stages.
3. **Sizing** — `volume_multiplier(data)` scales the adaptive position size
   (0.5x–2.0x by volume vs its 20-bar mean); the optional Kelly stage derives
   f* from the last 200 paper trades' PnL and *caps* (never raises) the size.
4. **Fee gate** — before `execute_buy/sell`, the expected move to the
   regime's take-profit target is compared against all-in costs (entry+exit
   taker fees + funding); non-viable trades are skipped with a logged breakdown.
5. **Exits** — the position loop updates a per-position `TrailingStop`
   (2% giveback from the high-water mark, both sides) and, when the symbol's
   regime is known, replaces the fixed $8 take-profit with the regime target
   (TRENDING 5%, REVERSAL 1%, RANGING 0.25%, CHOPPY 0.1% — percentage-based;
   the roadmap's absolute dollar offsets don't transfer across price levels).

## Why three flags default off

- **`ELVIS_ORDER_FLOW`** — paper-mode order books are empty (`get_order_book`
  returns empty bids/asks), so the imbalance is always neutral; enable against
  a live/testnet book.
- **`ELVIS_KELLY_SIZING`** — Kelly needs a meaningful trade history
  (`kelly_from_trades` returns the 1% floor below 20 trades); enable once the
  paper DB has enough closed trades.
- **`ELVIS_MTF`** — multiplies kline API calls per signal (3 timeframes);
  enable deliberately.

## Item 11 (adaptive ensemble) — wired via a real feedback pipeline

The honest wiring this section previously called "the follow-up" now exists.
No assumed accuracies anywhere — the loop *earns* its weights:

1. **Entry** — when a trade executes, every ensemble member's own vote
   (Technical / Research / RL / Bonenkamp / Models) is recorded to the
   `np.model_predictions` table (`EnsembleStrategy.last_model_votes` →
   `model_feedback.record_entry`).
2. **Exit** — once per cycle, `model_feedback.score_closed_trades` matches
   each vote batch to the first subsequent trade of that symbol carrying
   realized PnL. A model that voted the executed direction was right iff
   `pnl > 0`; an opposite vote was right iff `pnl < 0`; HOLD votes and
   `pnl == 0` closes are not scored. Open positions stay queued.
3. **Learning** — each outcome EMA-updates `AdaptiveEnsembleWeights`
   (`ELVIS_ADAPTIVE_ALPHA`, default 0.1), persisted atomically to
   `models/adaptive_ensemble_weights.json` across restarts.
4. **Voting** — `EnsembleStrategy` modulates its hand-tuned static source
   weights by the learned weights (`combine_weights`). Weights start uniform,
   so behavior is identical to the static ensemble until real scored trades
   accumulate, and multipliers are clamped to [0.25x, 4x] so no model is
   silenced or dominates off a handful of trades.

Kill switch: `ELVIS_ADAPTIVE_ENSEMBLE=0` restores pure static weighting.

## Item 15 (walk-forward) — how to run

```python
from trading.optimization.walk_forward import (
    WalkForwardOptimizer, sma_crossover_backtest,
)
import pandas as pd

data = pd.read_csv("data/processed/training_data.csv")  # needs a close column
opt = WalkForwardOptimizer(
    backtest_fn=sma_crossover_backtest,
    param_grid={"sma_short": [10, 14, 20], "sma_long": [40, 50, 60]},
)
result = opt.optimize(data, train_window=2000, test_window=500)
print(result["best_params"], result["mean_test_metric"])
```

Run it weekly (e.g. cron) against fresh data; apply the resulting parameters
manually or from your own automation. It is deliberately not in the live loop.

## Tests

`tests/test_signal_filters.py`, `test_position_sizing.py`, `test_exits.py`,
`test_fee_gate.py`, `test_order_flow.py`, `test_mtf.py`,
`test_adaptive_ensemble.py`, `test_model_feedback.py`,
`test_walk_forward.py` — 328 tests covering the
happy paths, edge cases (empty/short/NaN data, empty order books, degenerate
Kelly inputs), and pinned math (Kelly formula, EMA weight updates, fee
arithmetic). All modules import without torch/talib/network, so they run in CI.
