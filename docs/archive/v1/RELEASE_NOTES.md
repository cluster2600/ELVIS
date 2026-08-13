# ELVIS v0.2.0

> **Historical V1 document.** These v0.2.0 notes are retained for provenance,
> not as current release or operational guidance. The
> [ELVIS V2 architecture programme](../../../CHANGELOG.md#elvis-v2-architecture-programme-in-progress)
> is not deployed or activated. `ACTIVE` remains a **NO-GO**; see the
> [V2 overview](../../V2_ARCHITECTURE.md) for current context.

Documentation-audit release: verified 864 documented claims against the code,
fixed real bugs the audit surfaced, and **built the code to match the docs**
for the concrete capabilities they promised — each with tests and mermaid
diagrams of the real architecture.

## Docker image

```bash
docker pull ghcr.io/cluster2600/elvis:0.2.0   # or :latest
```

## What's in this release

**Code bugs fixed (were crashing on import)**
- `scripts/setup_vault.py` and `scripts/vault_admin.py` imported a nonexistent
  `utils.secrets_manager_enhanced` → fixed to `utils.secrets_manager`.
- `trading/api/app.py` imported a nonexistent `PaperTradeDB` class → added a thin
  `PaperTradeDB` wrapper in `utils/paper_trade_db.py`; the trading API imports again.

**Documented capabilities now implemented + tested**
- `trading/risk/advanced_risk_manager.py` and `core/features/feature_pipeline.py`
  — the import paths the docs reference now resolve to the real classes.
- `core/viz/export_utils.py` — CSV / Prometheus / SHAP export helpers.
- `trading_config.yaml` + `config.trading_config.load_trading_config()` — the
  unified config the docs describe (trading / risk_management / execution / monitoring).
- `RandomForestModel.explain_predictions()` and `.tune_hyperparameters()` — the
  documented explainability/tuning, using SHAP/Optuna when installed and degrading
  to `feature_importances_` / `GridSearchCV` otherwise (no 3.14 wheels for those).
- New `tests/test_documented_features.py` (7 tests) covers all of the above.

**Docs**
- Corrected the Vault instructions to the real `secrets` mount + `secrets/binance`
  (`api_key`/`secret_key`) layout, the cooldown claim, and the RandomForest
  (scikit-learn, not TFDF/SHAP) description.
- Added a "partially outdated" banner to five deep-dive docs that still describe a
  fictional architecture and need full rewrites.
- New `docs/architecture.md` with mermaid diagrams of the **verified** system.

## Known limitations

- `shap`, `optuna`, `ydf`, `tensorflow` have no Python 3.14 wheels; the affected
  features degrade gracefully via guards. Use the `[ml]` extra on Python 3.13 for
  the full stack.
- Five deep-dive docs remain substantially outdated (flagged with banners) and
  need rewrites — not fixed in this release.
