# Paper-strategy mechanism backlog

> **No performance claim:** this page inventories experimental mechanisms. It
> contains no expected return, win rate, Sharpe ratio, leverage scenario, or
> revenue projection. Each mechanism must be measured independently on held-out
> paper data before any conclusion.

The compatibility runtime contains these strategy experiments:

1. market-regime classification;
2. fail-closed RSI, momentum, Bollinger-squeeze, trading-hours, and MACD gates;
3. volume-based position scaling;
4. trailing-stop and regime-aware take-profit candidates;
5. fee-viability checks;
6. adaptive ensemble feedback;
7. optional public order-book confirmation;
8. optional capped Kelly sizing;
9. optional multi-timeframe confirmation; and
10. offline walk-forward evaluation.

The exact modules, flags, defaults, and known wiring limitations are recorded
in [the implementation status](profitability_roadmap_implementation.md).

## Evaluation contract

- Use causal, versioned features and time-ordered train/validation/test splits.
- Include fees, funding, slippage assumptions, rejected signals, and drawdown.
- Compare against a frozen baseline and retain the input/model manifest.
- Do not tune on the final test window.
- Treat paper and backtest results as experimental evidence, never a promise of
  live behaviour or future profitability.
- Do not enable live trading; the executable runtime remains paper-only.

These experiments do not advance V2 runtime authority. The
[V2 roadmap](architecture_migration/04-migration-roadmap.md) remains the
authority for migration and `ACTIVE` remains a **NO-GO**.
