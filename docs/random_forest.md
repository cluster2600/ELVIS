## Usage

- Hyperparameter optimization can be triggered via the training pipeline with configurable parameters.
- Feature engineering pipelines are modular and configurable to allow easy extension.
- SHAP-based explainability tools are accessible via model methods for analysis.

## Latest Changes

- Added robust handling of evaluation metrics including default values for missing keys such as 'loss'.
- Enhanced feature importance extraction to try multiple importance types for better coverage.
- Improved error handling and logging for model loading, saving, training, prediction, and evaluation.
- Integrated Optuna hyperparameter tuning with support for dynamic parameter suggestion during training.

## Future Work

- Full implementation of online and incremental learning capabilities.
- Enhanced visualization dashboards for model interpretability.
- Continuous integration of new market features and data sources.

---

This document will be updated iteratively as improvements are developed and integrated.
