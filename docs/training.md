# ELVIS Trading System - Model Training Documentation

## Overview

This document provides an overview and detailed explanation of the model training pipeline for the ELVIS trading system. It covers the key components, data preparation, model training, evaluation, explanation, and configuration.

---

## Components

### 1. Training Pipeline (`training/train_models.py`)

- **Purpose:** Orchestrates the entire training process including data loading, model training, evaluation, and explanation generation.
- **Key Classes and Functions:**
  - `TrainingPipeline`: Main class managing the training workflow.
  - `parse_args()`: Parses CLI arguments for configuration, data paths, and options.
  - `main()`: Entry point to run the training pipeline.

### 2. Model Trainer (`training/models/model_trainer.py`)

- **Purpose:** Handles model-specific logic including data preparation, training, validation, ensemble training, evaluation, and explanation.
- **Key Features:**
  - Supports PyTorch-based model training with a simple feedforward network example.
  - Manages ensemble models: stacking, weighted, and neural ensembles.
  - Provides model explanation capabilities using SHAP and LIME.
  - Includes methods for saving and loading models.

### 3. Evaluator (`training/models/evaluator.py`)

- **Purpose:** Monitors and evaluates agent performance during training.
- **Features:**
  - Records metrics such as rewards and steps.
  - Saves models when performance improves.
  - Plots learning curves.
  - Handles error logging for robustness.

### 4. Data Processing

- **Data Source:** OHLCV data downloaded from Binance API (`training/data/data_downloader.py`).
- **Data Format:** Processed data stored in `data/processed/training_data.csv` with features like `feature1`, `feature2`, `feature3`, and target `price`.
- **Data Processor:** Abstract base class defines interface for data processing (`core/data/processors/base_processor.py`).

---

## Configuration (`training/config/model_config.yaml`)

- **Feature Configuration:** Defines features used for training (`feature1`, `feature2`, `feature3`) and normalization method.
- **Model Parameters:** Includes transformer model hyperparameters and RL agent settings.
- **Training Parameters:** Batch size, checkpoint frequency, epochs, learning rates.
- **Output Paths:** Directories for models, logs, and checkpoints.

---

## Training Workflow

1. **Setup:**
   - Signal handlers for graceful interruption.
   - Logging initialization.
   - Configuration loading.
   - Distributed training setup (optional).
   - Directory creation for outputs.
   - Component initialization (data processor, model trainer, monitor, checkpoint manager, tensorboard writer).

2. **Data Loading and Preparation:**
   - Load CSV or Parquet data.
   - Extract features and target arrays based on configuration.
   - Validate data shapes.

3. **Data Loaders:**
   - Create PyTorch DataLoaders with time-series split for training and validation.

4. **Model Training:**
   - Train model for configured epochs.
   - Log training and validation metrics.
   - Save checkpoints periodically.
   - Monitor for early stopping.

5. **Reinforcement Learning Agents:**
   - Train multi-agent RL system if configured.

6. **Evaluation:**
   - Evaluate ensemble models and RL agents.
   - Save evaluation metrics.

7. **Explanation Generation:**
   - Generate model explanations using SHAP or LIME for transformer models.
   - Skip explanations for RL agents due to incompatibility.
   - Save explanations as JSON files.

---

## Known Issues and Warnings

- TensorFlow warnings related to version compatibility in SHAP explainers.
- RL agent explanation is currently unsupported and skipped.
- Device mismatch warnings handled by ensuring model and data are on the same device.

---

## Next Steps

- Implement advanced model training logic replacing placeholder methods.
- Enhance RL agent explanation support.
- Improve data processing and feature engineering.
- Address any remaining warnings and optimize performance.

---

## References

- `training/train_models.py`
- `training/models/model_trainer.py`
- `training/models/evaluator.py`
- `training/config/model_config.yaml`
- `training/data/data_downloader.py`
- `core/data/processors/base_processor.py`
