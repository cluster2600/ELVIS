# Research strategy compatibility guide

> This is an experimental paper-only strategy retained in the compatibility
> runtime. It does not reproduce a published result, guarantee a binary trade,
> enable live trading, or advance V2 authority.

The strategy was inspired by the feature families discussed in Annelotte
Bonenkamp's 2021 Bitcoin-trading research: technical indicators, optional
social inputs, a Random Forest classifier, time-ordered evaluation, and rolling
training. The repository implementation and data are different, so figures
reported by the paper are not ELVIS results or targets.

## Feature contract

The financial feature set includes RSI, stochastic oscillator, rate of change,
EMA, MACD, CCI, OBV, ATR, and Williams %R. Optional Twitter/X sentiment and
Google Trends inputs require separately installed Python 3.14-compatible
dependencies and external access. If collectors are unavailable, the current
compatibility code may emit neutral placeholders; treat that as degraded input,
not real social evidence.

Model artefacts must carry the versioned feature manifest expected by the
consumer. Missing, reordered, non-finite, or synthetic inputs must be rejected
or explicitly quarantined before evaluation.

## Paper-only invocation

```bash
STRATEGY_MODE=research \
SOCIAL_DATA_ENABLED=false \
ROLLING_TRAINING_ENABLED=false \
python main.py --mode paper
```

Enable optional data or retraining only after reviewing its dependencies,
credentials, rate limits, causality, and output provenance. The main runtime
still enforces the global paper-only capability gate.

## Evaluation

Use time-series-aware validation with frozen input and model manifests. Include
fees, funding, slippage assumptions, rejected signals, neutral placeholders,
and drawdown. Compare against a frozen baseline and never tune on the final
window. A paper or backtest score is not evidence of venue behaviour or future
profitability.

Focused tests:

```bash
python3.14 -m pytest -q tests/test_research_strategy_features.py
python3.14 -m pytest -q tests/test_research_feature_schema.py
```

The V2 runtime does not consume this strategy as a new authority. See the
[migration roadmap](architecture_migration/04-migration-roadmap.md); `ACTIVE`
remains a **NO-GO**.
