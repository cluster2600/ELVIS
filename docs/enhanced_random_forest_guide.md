# Enhanced Random Forest Implementation Guide

## Overview

The Enhanced Random Forest is a scikit-learn `RandomForestClassifier` wrapped in
`core/models/enhanced_random_forest_model.py` with optional MLOps add-ons:
hyperparameter search (Optuna), explainability (SHAP), Prometheus metrics, and a
comprehensive feature-engineering pipeline. It sits alongside the plain
`RandomForestModel` (`core/models/random_forest_model.py`) rather than replacing
it.

Every "enhanced" capability that depends on a third-party package is **optional
and guarded** — if the package is not installed, the feature is silently
disabled and the model still trains and predicts with sklearn + joblib.

> Python-3.14 note: `optuna` and `shap` are **not** pinned in `requirements.txt`
> and have no reliable 3.14 wheels yet. On a stock 3.14 environment they are
> absent, so hyperparameter optimization and SHAP explanations are disabled by
> default. The model falls back to hand-tuned default hyperparameters. This
> model is **sklearn**, not TensorFlow Decision Forests; the unrelated synthetic
> YDF Ensemble placeholder was retired.

## What the code actually provides

### Core

- **RandomForestClassifier core** — trained with `model.fit`, persisted with
  `joblib.dump` to `rf_model.pkl` plus a `rf_metadata.json` sidecar.
- **Optuna hyperparameter search** *(optional; needs `optuna`)* — a `trial`
  object is passed into `train()`, or a full study is run via
  `HyperparameterOptimizer`. Disabled automatically when Optuna is missing.
- **SHAP explanations** *(optional; needs `shap`)* — `shap.TreeExplainer` built
  on a background sample at train time; `explain_prediction()` returns SHAP
  values. Disabled automatically when SHAP is missing.
- **Feature engineering** — `TradingFeaturePipeline` generates 50+ OHLCV-derived
  features using the pure-Python `ta` library (not the TA-Lib C extension).
- **Prometheus metrics** *(optional; `enable_monitoring=True`)* — four metrics
  are registered and pushed to a Pushgateway (see the monitoring section for the
  important caveat).
- **Incremental "learning"** — `partial_fit()` buffers samples and periodically
  retrains from scratch on the buffer. sklearn's RandomForest has no true
  online update, so this is a full retrain, not incremental fitting.

### Not implemented (despite older claims)

- **Redis caching of predictions** — there is no Redis or cache code in the
  model or the integration layer. The `predict()` docstring mentions "caching"
  but nothing caches.
- **Drift-detection metrics/alerts** — `train()` stores per-feature
  mean/std/min/max in `training_data_stats` for future drift work, but no drift
  score is ever computed, exported, or alerted on.
- **A dedicated Grafana RF dashboard** — none of the provisioned dashboards in
  `grafana/dashboards/` reference the `rf_*` metrics. You would have to build
  panels yourself.

## Architecture Components

### 1. Enhanced Random Forest Model (`core/models/enhanced_random_forest_model.py`)

Class `EnhancedRandomForestModel(BaseModel)`.

```python
import logging
from core.models.enhanced_random_forest_model import EnhancedRandomForestModel

logger = logging.getLogger(__name__)

# enable_optuna / enable_shap are auto-downgraded to False if the package
# is not importable, regardless of what you pass.
model = EnhancedRandomForestModel(
    logger=logger,
    model_path="models/enhanced_rf",   # default is "models/rf_enhanced"
    enable_optuna=True,
    enable_shap=True,
    enable_monitoring=True,
    prometheus_gateway="localhost:9091",  # Pushgateway address
)

# train() returns validation metrics only when validation_data is given,
# otherwise an empty dict. Pass a live optuna Trial as `trial=` to use
# suggested hyperparameters; otherwise hand-tuned defaults are used.
validation_metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))

predictions = model.predict(X_test)
# explain_prediction returns an np.ndarray of SHAP values, or None if SHAP
# is unavailable/uninitialized.
explanations = model.explain_prediction(X_test, max_samples=10)
```

Default hyperparameters when no Optuna trial is supplied (from
`_get_default_hyperparameters`): `n_estimators=150, max_depth=12,
min_samples_split=5, min_samples_leaf=2, max_features="sqrt", bootstrap=True,
class_weight="balanced", n_jobs=-1`.

### 2. Trading Feature Pipeline (`core/models/features/trading_feature_pipeline.py`)

Class `TradingFeaturePipeline`. `transform(df)` takes an OHLCV DataFrame
(`open, high, low, close, volume`, optional `timestamp`) and returns a cleaned,
all-`float64` feature DataFrame. Fewer than 50 input rows falls back to
`_generate_minimal_features`.

Feature groups (populated in `self.feature_groups`, retrievable via
`get_feature_names()`):

- **price** — `price_change`, `high_low_ratio`, `close_to_high/low`, candlestick
  body/shadow, `is_hammer`, `is_doji`, SMA/EMA for windows 5/10/20/50 plus
  `price_to_sma_*`/`price_to_ema_*` and slopes, `price_position_14/28`, gap
  features.
- **volume** — `volume_change`, `volume_sma_*`, `volume_ratio`,
  `price_volume_trend`, `obv` (+ `obv_sma_10`, `obv_ratio`), `vwap`,
  `price_to_vwap`, `relative_volume_5/20`.
- **volatility** — `true_range`, `atr_14/28`, `volatility_14/28`,
  `normalized_volatility_*`, `gk_volatility` (Garman–Klass),
  `volatility_ratio`, `volatility_breakout`.
- **momentum** — `roc_5/10/20` (+ MAs), `momentum_10/20`,
  `price_acceleration`, `williams_r_14/21`, `stoch_k_*`, `stoch_d_*`.
- **technical** — with the `ta` library: `rsi_14/21`, `macd`/`macd_signal`/
  `macd_histogram`, Bollinger `bb_upper/lower/middle/width/position`, `adx`/
  `di_plus`/`di_minus`, `cci`, `mfi`, `psar`/`psar_signal`. Without `ta`, a basic
  fallback computes `rsi_14`, MACD, and Bollinger Bands only.
- **market_structure** — `support_20/50`, `resistance_20/50`, distances,
  `trend_strength`, `local_high/low` fractals, `regime_volatility`,
  `regime_trend`.
- **time** — hour/day/month, cyclical `hour_sin/cos`, `day_sin/cos`, session
  flags (`us_session`, `europe_session`, `asia_session`), `is_weekend`.

`transform()` also adds a handful of interaction features
(`price_volume_momentum`, `rsi_bb_combo`, `macd_rsi_combo`, `trend_confirmation`,
etc.). `get_feature_count()` and `validate_features(df)` are also available.

### 3. Hyperparameter Optimizer (`core/models/optimization/hyperparameter_optimizer.py`)

Class `HyperparameterOptimizer`. **Requires Optuna** — the constructor raises
`ImportError` if `optuna` is not importable.

```python
from core.models.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from core.models.enhanced_random_forest_model import EnhancedRandomForestModel

optimizer = HyperparameterOptimizer(
    model_class=EnhancedRandomForestModel,
    logger=logger,
    study_name="rf_optimization",
    storage_path="optimization_studies",  # results/trials JSON written here
)

results = optimizer.optimize(
    X_train, y_train,
    X_val=None, y_val=None,   # if given, uses a val split instead of CV
    n_trials=100,
    timeout=3600,             # seconds
    cv_folds=5,               # TimeSeriesSplit
    scoring_metric="f1_weighted",
    direction="maximize",
)

print(results["best_value"], results["best_params"])
```

Uses `TPESampler(seed=42)` and `MedianPruner`, TimeSeriesSplit CV, and Optuna's
built-in parameter-importance and a convergence/recommendation report saved to
`optimization_studies/`. Additional methods: `optimize_multiple_metrics(...)`,
`load_best_parameters(study_name=None)`, `get_optimization_summary()`.

### 4. Integration Layer (`core/models/integration/enhanced_rf_integration.py`)

Class `EnhancedRFIntegration`. Note `logger` is a **required first positional
argument**.

```python
from core.models.integration.enhanced_rf_integration import EnhancedRFIntegration

rf_integration = EnhancedRFIntegration(
    logger,                              # required, positional
    model_path="models/enhanced_rf",
    enable_optimization=True,            # builds a HyperparameterOptimizer (needs optuna)
    enable_monitoring=True,
    auto_retrain=True,
)

# market_data: an OHLCV dict or DataFrame; trading_features: extra columns.
prediction, metadata = rf_integration.get_prediction(market_data, trading_features)

# prediction is a length-3 np.ndarray of probabilities ordered [SELL, HOLD, BUY],
# or None on failure. Read weights/confidence from metadata:
ensemble_weight = metadata["ensemble_weights"]["final_weight"]
confidence = metadata["confidence"]
shap_explanation = metadata.get("shap_explanation", {})
```

Behavior worth knowing:

- On construction it tries to `load()` an existing model from
  `<model_path>/rf_model.pkl`; if none exists it initializes a fresh (untrained)
  model. `get_prediction` returns `(None, {"error": ...})` if the model is
  unavailable/untrained.
- Ensemble weight is dynamic: base `ensemble_weight` (default 0.3) scaled by a
  confidence multiplier and a prediction-entropy "certainty" score, then clipped
  to `[0.1, 0.5]`.
- `auto_retrain` only *logs an alert* when recent confidence degrades — it does
  not retrain automatically.
- `train_model(training_data, labels, optimize_hyperparameters=True)` trains and
  saves the model; `get_model_performance()`, `export_model_insights(...)`, and
  `update_configuration(config)` are also available.

## Quick Start

### Step 1: Dependencies

The core model works with what is already in `requirements.txt` (`scikit-learn`,
`joblib`, `pandas`, `numpy`, `ta`, `prometheus-client`). The optional extras are
**not** pinned and may not install on Python 3.14:

```bash
# Optional — only if wheels are available for your Python version:
pip install optuna shap
```

Do **not** try to `pip install ta-lib` / `brew install ta-lib` for this pipeline:
it uses the pure-Python `ta` package (already in `requirements.txt`), not the
TA-Lib C extension. Without `ta`, the pipeline still runs with a reduced
technical-indicator set.

### Step 2: Train

```bash
python scripts/train_enhanced_rf.py \
    --data-source database \      # database | csv | synthetic
    --optimize \                  # required flag to turn on Optuna search
    --trials 100 \                # default 50
    --samples 2000 \              # min samples; default 2000
    --model-path models/enhanced_rf
```

Data sources: `database` pulls paper trades via `utils.paper_trade_db.get_all_trades`
and labels them by realized PnL (BUY=2 / HOLD=1 / SELL=0); `csv` reads
`data/training_data.csv`; `synthetic` generates a random-walk OHLCV series. If a
source yields fewer than `--samples` rows it falls back to synthetic data. The
script also runs an integration smoke test and writes a JSON report under
`reports/`.

### Step 3: Use inside a strategy

```python
from core.models.integration.enhanced_rf_integration import EnhancedRFIntegration

class EnhancedEnsembleStrategy(BaseStrategy):
    def __init__(self, ...):
        self.enhanced_rf = EnhancedRFIntegration(
            self.logger,                       # positional
            model_path="models/enhanced_rf",
            enable_monitoring=True,
        )

    def generate_signal(self, market_data):
        rf_probs, rf_meta = self.enhanced_rf.get_prediction(
            market_data, self.get_trading_features()
        )
        if rf_probs is None:
            return None  # model unavailable / prediction failed
        rf_weight = rf_meta["ensemble_weights"]["final_weight"]
        # rf_probs is [SELL, HOLD, BUY]; combine with your other models here.
```

## Monitoring

### Prometheus metrics

When `enable_monitoring=True`, the model registers these metrics (see
`_setup_prometheus_metrics`):

```
rf_predictions_total                      # Counter: total predictions
rf_current_accuracy                       # Gauge:   last evaluate() accuracy
rf_feature_importance{feature_name=...}   # Gauge:   per-feature importance
rf_last_training_duration_seconds         # Gauge:   last train() duration
```

Important caveat: these are **pushed** to a Prometheus **Pushgateway** at
`prometheus_gateway` (default `localhost:9091`). The bundled `docker-compose.yml`
runs Prometheus on `9090` and **does not run a Pushgateway**, so by default the
push fails and is swallowed (logged at debug level). To actually collect these
metrics you must run a Pushgateway on `9091` and add it as a scrape target in
`observability/prometheus.yml`. The four metrics above are the only ones exported — there is no
drift or confidence metric.

### Grafana

Grafana is exposed on host port **3001** (`http://localhost:3001`, container
port 3000). None of the dashboards provisioned in `grafana/dashboards/` reference
the `rf_*` metrics, so there is no ready-made RF panel; you would build panels
for the metrics above yourself once a Pushgateway is wired up.

## Configuration

### Model constructor knobs

`EnhancedRandomForestModel(logger, model_path, enable_optuna, enable_shap,
enable_monitoring, prometheus_gateway)`. There is no dict-based `model_config`
API; configure via these constructor arguments. Optuna search space and pruning
live in `HyperparameterOptimizer` (`n_trials`, `timeout`, `cv_folds`,
`scoring_metric`, `direction`).

### Integration runtime config

`EnhancedRFIntegration.update_configuration(config)` accepts only these keys
(anything else is ignored):

```python
rf_integration.update_configuration({
    "ensemble_weight": 0.35,               # base weight (clipped to [0.1, 0.5] at use)
    "confidence_threshold": 0.6,           # stored on the instance
    "auto_retrain": True,                  # toggles the retrain-alert check
    "retrain_check_interval_hours": 6,     # how often the check runs
})
```

There are no `caching_enabled`, `drift_threshold`, or `explanation_export`
options — those do not exist in the code.

## Troubleshooting

**Optuna not installed / search disabled** — expected on Python 3.14. The model
logs a warning and uses default hyperparameters. Construct with
`enable_optuna=False` to silence the warning, or install `optuna` if a wheel is
available.

**SHAP not installed / explanations disabled** — same story; `explain_prediction`
returns `None`. Install `shap` if a wheel is available, else run with
`enable_shap=False`.

**Technical indicators missing** — if `import ta` fails, the pipeline logs
"TA-Lib not available - using basic indicators only" and falls back to a reduced
indicator set (RSI/MACD/Bollinger only). Ensure `ta` (pure-Python) is installed;
do not install the TA-Lib C extension for this path.

**Prometheus metrics not appearing** — see the monitoring caveat: no Pushgateway
runs by default. Check the scrape target that *does* exist with
`curl http://localhost:9090/api/v1/targets`, and restart Prometheus with
`docker restart elvis-prometheus` if needed. To get the `rf_*` metrics you must
add a Pushgateway on `9091`.

**Low model performance** — inspect feature quality and increase training data:

```python
validation = feature_pipeline.validate_features(features_df)
print(f"Data quality score: {validation['data_quality_score']:.3f}")

features_df, labels = trainer.prepare_training_data(min_samples=5000)
trainer.train_model(features_df, labels, optimize_hyperparameters=True)
```

## Migration from the plain Random Forest

The plain model lives at `core/models/random_forest_model.py`
(`class RandomForestModel(BaseModel)`). To switch:

1. **Back up** the current model artifacts.
2. **(Optional)** install `optuna`/`shap` if wheels are available for your Python.
3. **Update imports:**
   ```python
   # from core.models.random_forest_model import RandomForestModel
   from core.models.enhanced_random_forest_model import EnhancedRandomForestModel
   ```
4. **Update initialization:**
   ```python
   model = EnhancedRandomForestModel(
       logger=logger,
       enable_optuna=True,   # auto-disabled if optuna missing
       enable_shap=True,     # auto-disabled if shap missing
       enable_monitoring=True,
   )
   ```
5. **Retrain through the feature pipeline:**
   ```python
   pipeline = TradingFeaturePipeline(logger=logger)
   features = pipeline.transform(raw_ohlcv_df)
   model.train(features, labels, validation_data=(X_val, y_val))
   ```
6. **Wire the integration layer** into your ensemble as shown above.

## API Reference

### `EnhancedRandomForestModel(BaseModel)`

```python
def __init__(self, logger=None, model_path="models/rf_enhanced",
             enable_optuna=True, enable_shap=True, enable_monitoring=True,
             prometheus_gateway="localhost:9091")

def train(self, X_train, y_train, trial=None, validation_data=None) -> dict
def predict(self, X_test) -> np.ndarray
def predict_proba(self, X_test) -> np.ndarray
def evaluate(self, X_test, y_test) -> dict
def cross_validate(self, X, y, cv_folds=5, scoring=None) -> dict
def get_feature_importance(self) -> pd.DataFrame
def explain_prediction(self, X, max_samples=10) -> Optional[np.ndarray]
def partial_fit(self, X_new, y_new) -> None   # buffers then full-retrains
def save(self, path=None) -> None
@classmethod
def load(cls, path, logger=None) -> "EnhancedRandomForestModel"
def get_params(self) -> dict
def set_params(self, **params) -> "EnhancedRandomForestModel"
def get_model_stats(self) -> dict
```

### `TradingFeaturePipeline`

```python
def __init__(self, logger=None)
def transform(self, df) -> pd.DataFrame
def get_feature_names(self) -> dict     # feature groups
def get_feature_count(self) -> int
def validate_features(self, df) -> dict
```

### `HyperparameterOptimizer`  *(requires optuna)*

```python
def __init__(self, model_class, logger=None, study_name="rf_optimization",
             storage_path="optimization_studies", n_jobs=-1)
def optimize(self, X_train, y_train, X_val=None, y_val=None, n_trials=100,
             timeout=3600, cv_folds=5, scoring_metric="f1_weighted",
             direction="maximize") -> dict
def optimize_multiple_metrics(self, X_train, y_train, metrics=None, n_trials=50) -> dict
def load_best_parameters(self, study_name=None) -> Optional[dict]
def get_optimization_summary(self) -> dict
```

### `EnhancedRFIntegration`

```python
def __init__(self, logger, model_path="models/enhanced_rf",
             enable_optimization=True, enable_monitoring=True, auto_retrain=True)
def get_prediction(self, market_data, trading_features) -> tuple[np.ndarray|None, dict]
def train_model(self, training_data, labels, optimize_hyperparameters=True) -> dict
def get_model_performance(self) -> dict
def export_model_insights(self, output_path="model_insights") -> dict
def update_configuration(self, config) -> None
```

---

**Status**: sklearn RandomForest with optional (guarded) Optuna/SHAP/Prometheus
add-ons. Verified against the source under `core/models/`.
