# Research-Based Trading Strategy Guide

## Overview

The Research-Based Strategy implements the exact methodology from the academic paper:
**"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"** by Annelotte Bonenkamp (2021)

This strategy is designed to solve the two main issues with your current bot:
1. **Not trading enough** (eliminates HOLD signals)
2. **Losing money** (follows the methodology the paper reports at a 14.9% annual
   return — a paper target, not a measured bot result)

## Key Features

### 🔬 Research Methodology
- **Pure Random Forest** classifier (600 trees, 10-fold cross-validation)
- **Binary classification** (BUY/SELL only, no HOLD signals)
- **5-minute trading frequency** as specified in research
- **Rolling 1-week training** windows for dynamic updates
- **Target performance**: 14.9% annualized return, 2.02 Sharpe ratio

### 📊 Technical Indicators (9 from research)
1. **RSI** - Relative Strength Index
2. **STOCH** - Stochastic Oscillator
3. **ROC** - Rate of Change
4. **EMA** - Exponential Moving Average
5. **MACD** - Moving Average Convergence-Divergence
6. **CCI** - Commodity Channel Index
7. **OBV** - On Balance Volume
8. **ATR** - Average True Range
9. **WILLR** - Williams %R

### 📱 Social Features (2 from research)
1. **Twitter Sentiment** - Lagged 'Price' sentiment analysis
2. **Google Trends** - 'Bitcoin' search volume (interpolated to 5-minute)

> **Constraint — social features need optional dependencies + credentials.**
> Real Twitter sentiment requires the optional [`tweepy`](https://www.tweepy.org/)
> library **and** a Twitter/X API bearer token in `TWITTER_BEARER_TOKEN`. Real
> Google Trends data requires the optional [`pytrends`](https://pypi.org/project/pytrends/)
> library (locale/timezone can be tuned with `GOOGLE_TRENDS_HL` /
> `GOOGLE_TRENDS_TZ`; no API key is needed). Neither library ships a Python 3.14
> wheel, so in the minimal/CI environment the collectors fall back to neutral
> constants (Twitter sentiment `0.0`, Google Trends `50.0`) and log the reason
> at debug level. Install the libraries and provide the token to enable live
> data; otherwise the two social features contribute their neutral baseline.

## Usage

### Quick Start

```bash
# Use research strategy with default settings
STRATEGY_MODE=research python main.py --mode paper

# Use research strategy with social features enabled
SOCIAL_DATA_ENABLED=true STRATEGY_MODE=research python main.py --mode paper

# Use research strategy with rolling training
ROLLING_TRAINING_ENABLED=true STRATEGY_MODE=research python main.py --mode paper

# Full research configuration
STRATEGY_MODE=research SOCIAL_DATA_ENABLED=true ROLLING_TRAINING_ENABLED=true python main.py --mode paper
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRATEGY_MODE` | `ensemble` | Strategy to use: `research` or `ensemble` |
| `SOCIAL_DATA_ENABLED` | `true` | Enable Twitter + Google Trends features |
| `ROLLING_TRAINING_ENABLED` | `true` | Enable 1-week rolling training windows |

## Loop Pacing & Retraining (how it works)

The strategy documents a **5-minute trading frequency** and **rolling/daily
retraining**. Two pieces in `main.py` make the live trading loop honour that:

### How it works

- **Pacing.** At the end of every loop iteration `main.py` calls
  `_strategy_loop_sleep_seconds(active_strategy)`. If the strategy exposes a
  positive `trading_frequency_minutes` (the research strategy sets `5`), the
  loop sleeps `trading_frequency_minutes * 60` seconds before the next
  iteration. If the attribute is missing, non-positive, or unparseable, the
  loop falls back to a **1-second default** — so strategies without an opinion
  (e.g. the ensemble) behave exactly as before. The sleep is sliced into
  1-second chunks so shutdown (SIGINT/SIGTERM) stays responsive.

- **Retraining.** At the start of every iteration `main.py` calls
  `_retrain_strategy_if_due(active_strategy, price_fetcher, logger)`. It invokes
  `active_strategy.should_retrain()` **only if that method exists**. When it
  returns truthy, fresh `5m` history is fetched and handed to
  `train_model(...)` (same path as initial training). The whole hook is wrapped
  in `try/except` and is **non-fatal**: any retrain error is logged and the loop
  keeps trading. The research strategy's `should_retrain()` returns `True` at
  most once per 24h (daily rolling-window updates) and only when
  `ROLLING_TRAINING_ENABLED` is on, so this is a cheap check per iteration.

### How to use

No new configuration is required — both behaviours are automatic:

```bash
# Research strategy: loop paces at 5-minute intervals and retrains daily
ROLLING_TRAINING_ENABLED=true STRATEGY_MODE=research python main.py --mode paper
```

- To change the cadence, set `trading_frequency_minutes` on the strategy.
- To disable daily retraining, set `ROLLING_TRAINING_ENABLED=false` (the hook
  then finds `should_retrain()` returns `False` and does nothing).
- Strategies that expose neither attribute keep the legacy 1-second loop and are
  never retrained by this hook — paper mode continues to run unchanged.

### Strategy Comparison

| Feature | Ensemble Strategy | Research Strategy |
|---------|------------------|-------------------|
| **Signal Types** | BUY, SELL, HOLD | BUY, SELL only |
| **Models** | Multiple (YDF, CoreML, DRL) | Pure Random Forest |
| **Training** | Static pre-trained | Rolling 1-week windows |
| **Social Data** | None | Twitter + Google Trends |
| **Research Basis** | Custom implementation | Academic paper (2021) |
| **Target Performance** | Unknown | 14.9% annual return |

## Expected Results

### 🎯 Performance Targets (from the source paper — NOT measured bot results)

> ⚠️ The figures below are the results **reported in Bonenkamp (2021)** for the
> paper's own backtest. They are the design targets this strategy aims to
> reproduce — they are **not** verified live or paper-trading results of this
> bot. Treat them as reference numbers from the paper, not a promise of bot
> performance.

- **Annualized Return**: 14.9% *(paper target)*
- **Annualized Sharpe Ratio**: 2.02 *(paper target)*
- **F1-Score**: 57.6% *(paper target)*
- **Accuracy**: 57.8% *(paper target)*

### 🚀 Trading Behavior Changes
- **More active trading** (no HOLD signals)
- **Binary decisions** (always BUY or SELL)
- **Higher confidence** signals (model probability drives confidence)
- **Aims to reproduce** the paper's methodology (past paper results are not a
  guarantee of future profitability)

## Testing

Run the end-to-end validation script (it prints a step-by-step report and can be
run directly):

```bash
python tests/test_research_strategy.py
```

This will test:
- ✅ Financial indicators calculation
- ✅ Social features collection
- ✅ Feature preparation and standardization
- ✅ Signal generation (BUY/SELL only)
- ✅ Model training with cross-validation
- ✅ Research methodology compliance

For fast, dependency-free assertions (no talib/torch/tweepy/pytrends required)
covering the binary-signal guarantee, time-series-aware cross-validation, and the
import-guarded social collectors, run the focused pytest suite:

```bash
pytest tests/test_research_strategy_features.py
```

## Implementation Details

### Financial Indicators Formulas

The strategy implements the exact mathematical formulas from the research:

1. **RSI**: `RSI = 100 - (100 / (1 + RS))` where `RS = Ave(Gains) / Ave(Losses)`
2. **STOCH**: `STOCH = 100 * (Ck - Lp) / (Hp - Lp)`
3. **ROC**: `ROC(p) = 100 * (Ck - Ck-p) / Ck-p`
4. **EMA**: `EMAk(p) = EMAk-1(p) + (2/(p+1)) * (Ck - EMAk-1(p))`
5. **MACD**: `MACD = EMA12 - EMA26`, Signal = `EMA9(MACD)`
6. **CCI**: `CCI(n) = (1/0.015) * (TPk - SMAn(TPk)) / σn(TPk)`
7. **OBV**: Volume-based momentum calculation
8. **ATR**: `ATR = max(Hk-Lk, |Hk-Ck-1|, |Lk-Ck-1|)`
9. **WILLR**: `WILLR = -100 * (Hp - Ck) / (Hp - Lp)`

### Feature Standardization

All features are standardized using the research formula:
```
x_standardized = (x - μ) / σ
```

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Trees**: 600 (exactly as specified)
- **Cross-Validation**: 10-fold
- **Classes**: Binary (0=SELL, 1=BUY)
- **Training Window**: 1 week of 5-minute data
- **Retraining**: Daily updates

### Binary Signals, Time-Series CV, and Feature Consistency

This section documents how three specific behaviors work and how to rely on
them.

#### How it works

- **Binary BUY/SELL signals (no HOLD).** `generate_signals()` never emits a
  `HOLD`. Every branch resolves to a side:
  - Trained model: `BUY` when `buy_prob > 0.5`, otherwise `SELL`.
  - Untrained model / prediction failure: RSI-based fallback — `RSI < 35 → BUY`,
    `RSI > 65 → SELL`, and a **neutral RSI (35–65) picks a side** rather than
    holding (`RSI < 50 → SELL`, else `BUY`).
  - No/empty market data or an unhandled error: conservative `SELL` with
    `confidence 0.0`.
- **Time-series-aware cross-validation.** `train_model()` uses
  `sklearn.model_selection.TimeSeriesSplit(n_splits=10)` (no shuffle) for its
  10-fold CV. Because financial samples are time-ordered, this trains only on
  past data and validates on future data — unlike the default (Stratified)`KFold`
  that `cross_val_score` would otherwise select for an integer `cv` value.
- **Feature-count consistency.** At predict time the feature vector length now
  respects the social flag: **11 features** (9 financial + 2 social) when
  `social_data_enabled=True`, and **9 features** when it is disabled. This
  matches the training-time vector, so the model no longer silently truncates
  the two social features to 9.

#### How to use

- Run the strategy as usual (see [Usage](#usage)); the binary-signal behavior is
  automatic and needs no configuration.
- Keep `SOCIAL_DATA_ENABLED` consistent between training and inference so the
  9-vs-11 feature count matches. If you train with social data enabled, run
  inference with it enabled too (and vice versa).
- No action is required to get time-series CV — it is the default for
  `train_model()`.

## Troubleshooting

### Common Issues

1. **Model not training**
   - Ensure sufficient historical data (>100 samples)
   - Check data quality and missing values
   - Verify feature calculation

2. **Social features failing**
   - Set `SOCIAL_DATA_ENABLED=false` to disable
   - Install the optional libs: `pip install tweepy pytrends` (no 3.14 wheels —
     use a compatible interpreter)
   - Set `TWITTER_BEARER_TOKEN` for Twitter sentiment (Google Trends needs none)
   - Without the libs/token the collectors return neutral constants and log the
     reason at debug level — this is expected, not an error
   - Review network connectivity

3. **Low trading activity**
   - This should NOT happen with research strategy
   - Binary classification guarantees BUY or SELL
   - Check logs for signal generation

### Logs to Monitor

```
🔬 Research-based strategy active - targeting 14.9% annual returns
📊 Binary classification: BUY/SELL only (no HOLD signals)
🎯 Following Bonenkamp (2021) research methodology
🎯 Research signal: BUY with 0.742 confidence
📊 Probabilities: BUY=0.742, SELL=0.258
```

## Next Steps

1. **Start with financial indicators only**:
   ```bash
   STRATEGY_MODE=research SOCIAL_DATA_ENABLED=false python main.py --mode paper
   ```

2. **Monitor trading activity** - should be much more active than ensemble

3. **Add social features** once basic strategy works well

4. **Enable rolling training** for dynamic model updates

5. **Compare performance** with ensemble strategy over time

This research-based strategy directly addresses your trading issues by following an academic methodology that the source paper reports at a 14.9% return, using active binary trading decisions. That 14.9% is the paper's reported figure, not verified performance of this bot.
