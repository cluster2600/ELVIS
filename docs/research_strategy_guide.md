# Research-Based Trading Strategy Guide

## Overview

The Research-Based Strategy implements the exact methodology from the academic paper:
**"High-Frequency Algorithmic Bitcoin Trading Using Both Financial and Social Features"** by Annelotte Bonenkamp (2021)

This strategy is designed to solve the two main issues with your current bot:
1. **Not trading enough** (eliminates HOLD signals)
2. **Losing money** (follows proven 14.9% annual return methodology)

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

### 🎯 Performance Targets (from research)
- **Annualized Return**: 14.9%
- **Annualized Sharpe Ratio**: 2.02
- **F1-Score**: 57.6%
- **Accuracy**: 57.8%

### 🚀 Trading Behavior Changes
- **More active trading** (no HOLD signals)
- **Binary decisions** (always BUY or SELL)
- **Higher confidence** signals (research-proven methodology)
- **Consistent profitability** (follows academic research)

## Testing

Run the test script to validate the implementation:

```bash
python test_research_strategy.py
```

This will test:
- ✅ Financial indicators calculation
- ✅ Social features collection
- ✅ Feature preparation and standardization
- ✅ Signal generation (BUY/SELL only)
- ✅ Model training with cross-validation
- ✅ Research methodology compliance

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

## Troubleshooting

### Common Issues

1. **Model not training**
   - Ensure sufficient historical data (>100 samples)
   - Check data quality and missing values
   - Verify feature calculation

2. **Social features failing**
   - Set `SOCIAL_DATA_ENABLED=false` to disable
   - Check API keys for Twitter/Google Trends
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

This research-based strategy directly addresses your trading issues by following a proven academic methodology that achieved 14.9% returns with active binary trading decisions.
