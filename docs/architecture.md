# ELVIS Architecture (verified 2026-07-02)

Diagrams below reflect the **actual** code after the doc-audit fixes — every
module, class, and path shown exists and is import-verified.

## System / runtime flow

```mermaid
flowchart TD
    A["main.py (paper or live mode)"] --> B["core.bootstrap.bootstrap_application"]
    B --> C["core.di container"]
    C --> S["trading.strategies.EnsembleStrategy"]
    C --> X["trading.execution.BinanceExecutor"]
    C --> P["utils.price_fetcher.PriceFetcher"]
    C --> R["trading.risk.AdvancedRiskManager"]
    P -->|real Binance klines| S
    S -->|BUY / SELL / HOLD| X
    X -->|paper fills + PnL| DB[("Postgres: np.trades / np.open_positions")]
    DB --> API["trading.utils.trade_history_api Flask :5050"]
    X -.->|balance = deposit + realized PnL| DB
```

## Configuration & secrets

```mermaid
flowchart LR
    subgraph Config
      CC[config.config.TRADING_CONFIG]
      TY[trading_config.yaml] --> LT[config.trading_config.load_trading_config]
    end
    subgraph Secrets
      SM[utils.secrets_manager.EnhancedSecretsManager]
      VC[utils.vault_client mount=secrets]
      SM --> VC
      VC -->|secrets/binance api_key,secret_key| OB[(OpenBao :8200)]
    end
    CC --> BOOT[bootstrap]
    LT --> BOOT
    SM --> BOOT
    BOOT -.->|DEFAULT_LEVERAGE / OVERRIDE_HIGH_LEVERAGE| GUARD[validate_leverage_config]
```

## RandomForest model capabilities

```mermaid
flowchart TD
    RF[core.models.random_forest_model.RandomForestModel<br/>sklearn RandomForestClassifier] --> T[train]
    RF --> PR[predict]
    RF --> CV[cross_validate k-fold]
    RF --> EX[explain_predictions<br/>SHAP if installed, else feature_importances_]
    RF --> TU[tune_hyperparameters<br/>Optuna if installed, else GridSearchCV]
    RF --> SV[save/load via joblib]
    CV --> PM[push_cv_metrics_to_prometheus]
    EX --> EU[core.viz.export_utils<br/>export_to_csv / export_shap_summary]
    PM --> PG[(Prometheus Pushgateway)]
```

## Notes on optional dependencies

- `shap` and `optuna` have no Python 3.14 wheels yet; `explain_predictions` and
  `tune_hyperparameters` **degrade gracefully** (feature-importance export and
  `GridSearchCV` respectively) so the documented interface works everywhere.
- `ydf` / `tensorflow` are likewise absent on 3.14; the ensemble skips those
  members via import guards. Install the `[ml]` extra on Python 3.13 for the
  full stack.
