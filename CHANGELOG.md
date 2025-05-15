# Changelog

## [Unreleased]

### eureka
- Fixed import error for `TFKerasPruningCallback` in `model_trainer.py` by adding fallback handling for missing `optuna_integration` module.
- Corrected `DataProcessor` class definition and imports in `trading/data/data_processor.py` to resolve import errors.
- Fixed import path for `setup_logger` in `trading/scripts/train_models.py`.
- Added `xgboost` package to `requirements.txt` to resolve missing module error during training.
- Verified training script runs without previous import errors.
- Fixed data validation in TradingEnvironment to handle cases where data is not an array, preventing TypeError.
- Integrated meta-learning into training/learner.py for advanced RL extensions.
- Implemented multi-agent support in training/worker.py for advanced RL extensions.
- Implemented transfer learning in training/models/evaluator.py for two-stage training loop.
- Implemented automated hyperparameter optimization in training/train_models.py for advanced RL extensions.
- Reviewed and ensured compliance with training.md documentation for all changes.
- Fixed import error in training/train_models.py for Evaluator module by correcting the import path.
- Verified that the corrected import resolves the ModuleNotFoundError in training/train_models.py.
- Added MPS device detection in training/config.py for Apple Silicon compatibility.
- Added configuration toggles for two-stage training (pretrain and finetune) in training/config.py.
- Implemented two-stage training logic in training/train_models.py to use the new toggles.
- Verified that the two-stage training logic functions correctly with the toggles in training/train_models.py.
- Fixed Evaluator initialization error in training/train_models.py by providing required arguments.
- Verified that the new toggles are correctly implemented and functional in training/config.py.
