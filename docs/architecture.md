# ELVIS compatibility-runtime architecture (verified 2026-07-12)

Diagrams below reflect the **actual** code after the doc-audit fixes and the
2026-07 root reorganization — every module, class, and path shown exists and
is import-verified.

> This is the audited legacy/runtime-compatibility topology, retained while
> ELVIS V2 is migrated in reversible slices. It is not the V2 target and does
> not prove V2 activation. See the [V2 architecture](V2_ARCHITECTURE.md) for the
> new approach and the
> [migration roadmap](architecture_migration/04-migration-roadmap.md) for
> current authority. `ACTIVE` remains a **NO-GO**.

## System / runtime flow

```mermaid
flowchart TD
    A["main.py (paper-only capability gate)"] --> B["core.bootstrap.bootstrap_application"]
    B --> C["core.di container"]
    C --> S["trading.strategies.EnsembleStrategy"]
    C --> X["trading.execution.BinanceExecutor"]
    C --> P["utils.price_fetcher.PriceFetcher"]
    C --> R["trading.risk.AdvancedRiskManager"]
    P -->|real Binance klines| S
    S -->|BUY / SELL / HOLD| G["signal-quality gates: regime detector + winrate filter + trading.signals (RSI, momentum, BB squeeze, hours, MACD, order flow, MTF)"]
    G -->|approved signal| SZ["sizing + fee gate: trading.risk.position_sizing + trading.fees.fee_gate"]
    G -->|vetoed| H["HOLD"]
    SZ -->|viable trade| I["typed Signal + OrderIntent"]
    I --> OS["OrderService: one call, no retry"]
    OS --> LA["LegacyPaperExecutionAdapter"]
    LA --> X
    X -->|paper fills + PnL| DB[("Postgres: np.trades / np.open_positions")]
    X -.->|"exits: trading.execution.exits (trailing stop, regime take-profit)"| X
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

- `shap` and `optuna` are omitted from the minimal install; `explain_predictions` and
  `tune_hyperparameters` **degrade gracefully** (feature-importance export and
  `GridSearchCV` respectively) so the documented interface works everywhere.
- The synthetic YDF and CoreML ensemble members were retired: neither had a
  deployable, provenance-backed model. The tracked YDF placeholder scored 37.1%
  OOB accuracy, below its 39.3% majority-class baseline. The `[ml]` extra now
  contains only supported 3.14-compatible add-ons.
- Python 3.14 is the only supported runtime. Optional model integrations without
  compatible wheels remain import-guarded and are omitted rather than moved to
  an older interpreter.
