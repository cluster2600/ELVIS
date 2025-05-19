## Random Forest Model Documentation

### Usage

- Hyperparameter optimization can be triggered via the training pipeline with configurable parameters.
- Feature engineering pipelines are modular and configurable to allow easy extension.
- SHAP-based explainability tools are accessible via model methods for analysis.
- Cross-validation can be executed via the `cross_validate()` method, with integrated Optuna trial support.
- Visual insights across folds are available through automatic plotting of metrics.
- Execution time for core methods (`train`, `evaluate`, `predict`, etc.) is automatically logged via decorators.

### Latest Changes

- Added robust handling of evaluation metrics including default values for missing keys such as `'loss'`.
- Enhanced feature importance extraction to try multiple importance types for better coverage.
- Improved error handling and logging for model loading, saving, training, prediction, and evaluation.
- Integrated Optuna hyperparameter tuning with support for dynamic parameter suggestion during training.
- Added `cross_validate()` method with k-fold support, metric aggregation, and ROC-AUC calculation (for binary tasks).
- Introduced visual feedback for cross-validation: matplotlib plots of per-fold metric performance.
- Added `@log_time` decorators to core methods to track and log runtime duration.
- Simplified and modularized class structure for improved readability and testability.

### Future Work

- Full implementation of online and incremental learning capabilities.
- Enhanced visualization dashboards for model interpretability.
- Continuous integration of new market features and data sources.
- Support for stratified or grouped k-fold cross-validation.
- Export of visual artifacts (plots, architecture graphs) for documentation pipelines.

---

This document will be updated iteratively as improvements are developed and integrated.