# Changelog

## [Unreleased]

### eureka
- Fixed import error for `TFKerasPruningCallback` in `model_trainer.py` by adding fallback handling for missing `optuna_integration` module.
- Corrected `DataProcessor` class definition and imports in `trading/data/data_processor.py` to resolve import errors.
- Fixed import path for `setup_logger` in `trading/scripts/train_models.py`.
- Added `xgboost` package to `requirements.txt` to resolve missing module error during training.
- Verified training script runs without previous import errors.
- Fixed data validation in TradingEnvironment to handle cases where data is not an array, preventing TypeError.
