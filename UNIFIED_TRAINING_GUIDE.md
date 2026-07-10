# 🚀 ELVIS Unified Training System - Guide

`run_training.sh` is a single entry point that dispatches to the various
training scripts in the repo root, with method auto-detection, dependency
checks, optional Vault setup, and runtime fallback. This guide describes what
the script actually does; `./run_training.sh --help` is the authoritative
reference for the full flag list.

## 🎯 **Single Command Training**

### **Quick Start (Recommended)**
```bash
# Auto-detect best method and run training
./run_training.sh

# Quick test run (100 trades, 5 epochs with debug)
./run_training.sh --quick

# Production training (5000 trades, 50 epochs)
./run_training.sh --production
```

### **Method-Specific Training**
```bash
# PostgreSQL method (best for real data)
./run_training.sh --method postgres --limit 2000 --epochs 30

# Enhanced method with Vault support
./run_training.sh --method enhanced --debug

# No-Vault method (testing/development)
./run_training.sh --method no-vault --limit 500

# Research-strategy method (Bonenkamp 2021 methodology)
./run_training.sh --method research --social
```

## 📊 **Available Options**

These are the flags accepted by `run_training.sh` (see `--help` for the full,
authoritative list — the script's `print_usage` is the source of truth).

### **Training Methods**
- `--method postgres` - **Recommended**: trains against the PostgreSQL trade DB
- `--method enhanced` - Runs `train_with_trades.py` (Vault fixes + fallbacks)
- `--method no-vault` - Complete Vault bypass for testing
- `--method research` - Research-based strategy training (Bonenkamp 2021); runs
  with Vault disabled and paper-mode credentials
- `--method auto` - **Default**: auto-detects the best available method

### **Parameters**
- `--limit NUM` - Maximum trades to use (default: `0` = all available)
- `--epochs NUM` - Training epochs (default: `20`)
- `--horizon NUM` - Prediction horizon in trades ahead (default: `5`)
- `--debug` - Enable debug mode with detailed logging
- `--no-vault` - Disable Vault (equivalent to `--method no-vault`)

### **Research-Strategy Options**
- `--social` / `--no-social` - Toggle social-data features (Twitter + Google Trends)
- `--rolling` / `--no-rolling` - Toggle rolling training (1-week windows)
- `--live` - Prepare models for live deployment (training only; noted in logs)

### **Enhanced / Phase-2 Options** (optional, feature-gated)
These flags set environment variables and check for the corresponding module
before activating; if the module is missing, the feature is skipped with a
warning:
- `--automl` / `--automl-trials N` - AutoML hyperparameter search
  (`training/automl/hyperparameter_optimizer.py`, requires `optuna`)
- `--dashboard` / `--dashboard-log F` - Real-time console dashboard
  (`utils/dashboard/console_dashboard.py`)
- `--llm` / `--llm-provider P` / `--llm-model M` / `--llm-url URL` - LLM market
  analysis (`core/ai/llm_market_analyzer.py`; provider defaults to `local`,
  model `openai/gpt-oss-20b`, URL `http://localhost:1234/v1`)
- `--continuous` - Continuous learning pipeline
  (`core/streaming/continuous_learning_pipeline.py`)
- `--multimodal` - Multi-modal learning (`core/multimodal/multimodal_learning_system.py`)
- `--nas` / `--nas-generations N` - Neural architecture search
  (`core/nas/neural_architecture_search.py`)

### **Quick Options**
- `--quick` - Quick test: sets `--limit 100 --epochs 5` and enables debug mode
- `--production` - Production: sets `--limit 5000 --epochs 50`
- `--research` - Research strategy mode (also enables `--social` and `--rolling`)
- `--check` - Run system diagnostics only (runs `debug_training.py`)
- `--help` - Show detailed help

## 🎯 **Auto-Detection Logic**

When `--method auto` (the default) is used, `auto_detect_method()` selects the
method in this order:

1. **Research mode** - If `--research` was passed, uses the `research` method
2. **PostgreSQL** - Otherwise, if PostgreSQL is reachable with a `trades` table,
   uses the `postgres` method
3. **Fallback** - If PostgreSQL is unavailable, falls back to the `no-vault` method

Separately from auto-detection, the script also:

- **Validates dependencies** up front (`pandas`, `numpy`, `torch`, `psycopg2`
  are mandatory; `tensorflow` is a soft, optional warning)
- **Sets up Vault** later, only for non-`no-vault`/non-`research` methods, and
  degrades gracefully to Vault-off if the dev server can't be started
- **Falls back at runtime** - if the `postgres` method is chosen but PostgreSQL
  turns out to be unreachable, it switches to the `no-vault` method

> Note: auto-detection does **not** probe or "enable Vault features". Vault is
> configured separately and only affects the postgres/enhanced paths.

## 🧩 **Optional ML Dependencies (optuna / tensorflow / shap / lime)**

Some advanced training and interpretability features rely on libraries that are
**optional**: they have no wheels on newer Python versions and are absent in CI
and minimal environments. The core PyTorch + ensemble training path does **not**
need them, so the training modules are designed to import and run without them.

### How it works

- Each optional import is wrapped in `try/except ImportError` and exposed via an
  `*_AVAILABLE` flag:
  - `training/models/model_trainer.py` → `OPTUNA_AVAILABLE`, `TENSORFLOW_AVAILABLE`
  - `training/automl/hyperparameter_optimizer.py` → `OPTUNA_AVAILABLE`
    (plus the existing `TF_AVAILABLE` / `TORCH_AVAILABLE`)
  - `training/models/explainable_ai.py` → `SHAP_AVAILABLE`, `LIME_AVAILABLE`,
    `PLOTLY_AVAILABLE`
- Importing these modules always succeeds, even when `optuna`, `tensorflow`,
  `keras`, `shap`, or `lime` are not installed. Test collection therefore works
  in a minimal environment.
- Explanation generation degrades gracefully. `generate_explanations()` in
  `training/models/explainable_ai.py` logs a warning and returns `{}` (a no-op)
  when the required backend (`shap` for tensor models, `lime` for sklearn-style
  models) is missing, instead of crashing the pipeline.
- Directly instantiating `SHAPExplainer` / `LIMEExplainer`, calling their plotly
  visualizer, or calling `HyperparameterOptimizer.optimize_model()` without the
  matching library raises a clear `ImportError` telling you what to `pip install`.
- `run_training.sh`'s dependency check treats **TensorFlow as optional**: it is a
  soft warning (`⚠️  TensorFlow not installed (optional)`), not a hard failure.
  Only `pandas`, `numpy`, `torch`, and `psycopg2` are mandatory.

### How to use

- **Minimal / CI run** — do nothing. Training runs on PyTorch + ensembles;
  optuna AutoML, TensorFlow model variants, and SHAP/LIME explanations are simply
  skipped (with warnings in the logs).
- **Enable a feature** — install the corresponding extra and rerun:
  ```bash
  venv314/bin/pip install optuna       # AutoML hyperparameter search
  venv314/bin/pip install tensorflow   # TensorFlow / keras model paths
  venv314/bin/pip install shap lime    # model explanations
  ```
  With the library present, the matching `*_AVAILABLE` flag flips to `True` and
  the feature activates automatically — no code changes needed.

## 📈 **What Gets Produced**

A completed run writes trained model artifacts plus processed feature/target
data. The processed-data snapshot in this repo (`data/processed/trade_history/
trade_data_metadata.json`) describes a full extraction of:

- **35 engineered features** per sample (`side`, `price`, `quantity`, `pnl`,
  `fee`, time features, rolling price/volume stats, win/loss streaks, etc. —
  see `feature_names` in the metadata)
- **6 prediction targets** per sample
- **4,995 valid samples** (full dataset — a `--quick` run caps input at 100
  trades, so it processes far fewer)

## 🛠️ **What Actually Runs Under Each Method**

`run_training.sh` builds a command by probing which training scripts exist. The
selection precedence (see `run_training_postgres` / `run_training_no_vault` /
`run_training_research` in the script) is important because it determines which
model type you get:

### **`postgres` / `auto` method**
- If `--llm` is set and `train_all_trades_patient_llm.py` exists → runs that
  ("patient" LLM path for slow LLMs)
- **Otherwise, the usual path**: `train_all_paper_trades.py --method auto` — this
  trains **scikit-learn RandomForest models** (a `RandomForestClassifier`, a
  `RandomForestRegressor`, and a `StandardScaler`), persisted with **joblib** to
  `models/all_trades_*_<timestamp>.joblib`. There is no TensorFlow / TFDF here.
- Only if `train_all_paper_trades.py` is absent does it fall back to
  `train_with_postgres.py`, which delegates to `training/train_models.py` and
  trains a **PyTorch `FinancialTransformer` plus RL agents**, writing
  `models/transformer_evaluation.json` and `models/rl_agents_evaluation.json`.

### **`enhanced` method**
- Runs `train_with_trades.py` (Vault fixes; add `--no-vault` to bypass Vault).

### **`no-vault` method**
- Runs `train_no_vault.py` (or the LLM script when `--llm` is set).

### **`research` method**
- Prefers `train_all_paper_trades.py --method all_trades` (all available trades),
  falling back through the LLM / postgres / trades / no-vault scripts if that
  file is missing. Runs with Vault disabled and paper-mode credentials.

## 🛠️ **Integration Components**

### **Scripts Integrated** (all present in the repo root)
1. `train_all_paper_trades.py` - Default data-maximizing path (RandomForest + joblib)
2. `train_all_trades_patient_llm.py` - Patient LLM path for slow local LLMs
3. `train_with_postgres.py` - Delegates to `training/train_models.py` (Transformer + RL)
4. `train_with_trades.py` - Enhanced original with Vault fixes
5. `train_no_vault.py` - Complete Vault bypass
6. `debug_training.py` - System diagnostics (invoked by `--check`)

### **Features Integrated**
- ✅ **Automatic Method Detection** (research → postgres → no-vault)
- ✅ **PostgreSQL Database Integration**
- ✅ **Vault Authentication Bypass** (`no-vault`/`research`)
- ✅ **Runtime Fallback** (postgres → no-vault if DB unreachable)
- ✅ **Real-time Progress Reporting**
- ✅ **Dependency Validation** (hard: pandas/numpy/torch/psycopg2; soft: tensorflow)
- ✅ **Optional Phase-2 AI hooks** (AutoML, LLM, continuous, multimodal, NAS)
- ✅ **Colored Output for Clarity**

## 🎯 **Usage Examples**

### **Development Workflow**
```bash
# 1. Quick test to verify everything works
./run_training.sh --quick

# 2. Development training with debug
./run_training.sh --debug --limit 1000

# 3. Validation with more data  
./run_training.sh --limit 3000 --epochs 25

# 4. Production deployment
./run_training.sh --production
```

### **Troubleshooting**
```bash
# System diagnostics
./run_training.sh --check

# Force specific method
./run_training.sh --method no-vault --debug

# Test with minimal data
./run_training.sh --limit 50 --epochs 3 --debug
```

### **Batch Operations**
```bash
# Test all methods
./run_training.sh --method postgres --quick
./run_training.sh --method enhanced --quick  
./run_training.sh --method no-vault --quick

# Performance comparison
./run_training.sh --limit 1000 --epochs 10
./run_training.sh --limit 2000 --epochs 10
./run_training.sh --limit 5000 --epochs 10
```

## 📊 **Output Structure**

### **Generated Files**
The exact artifacts depend on which method/script ran (see "What Actually Runs"
above). Common locations:
```
models/
├── checkpoints/                          # PyTorch training checkpoints (.pt)
├── logs/                                 # Training logs
├── all_trades_classifier_<timestamp>.joblib   # RandomForest (default path)
├── all_trades_regressor_<timestamp>.joblib
├── all_trades_scaler_<timestamp>.joblib
├── transformer_evaluation.json           # Transformer path (train_models.py)
├── rl_agents_evaluation.json             # RL path (train_models.py)
└── ...

data/processed/
└── trade_history/
    ├── trade_data_metadata.json
    ├── trade_features.csv
    └── trade_targets.csv
```
> Whether you get `*.joblib` RandomForest artifacts or `*_evaluation.json`
> Transformer/RL outputs depends on the script precedence, not on a config flag.

### **Training Logs**
- Real-time progress display with colors
- Feature engineering logs
- Model training metrics
- PostgreSQL connection status
- Error messages with hints

## 🚀 **Notes on Scale**

- **PostgreSQL source**: the research/all-trades path is written to consume all
  available trades (`--trades 0`); comments in the script reference an expected
  order of ~25k+ trades, but the actual count depends on your database.
- **Processed snapshot**: the checked-in `trade_history` extraction contains
  **4,995 samples** with **35 features** and **6 targets**.
- **Checkpointing**: the Transformer path (`training/train_models.py`) saves and
  can resume from `models/checkpoints/*.pt` via `CheckpointManager` (see
  `--resume`).
- **Runtime** scales with `--limit`/`--epochs` and the chosen model type; the
  script prints total wall-clock duration on completion.

## 🏆 **Key Points**

1. ✅ **Unified Interface**: single `run_training.sh` entry point for all methods
2. ✅ **Auto-Detection**: research → postgres → no-vault selection
3. ✅ **Runtime Fallback**: postgres → no-vault if the DB is unreachable
4. ✅ **PostgreSQL Integration**: trains against the local `trades` table
5. ✅ **Vault Independence**: `no-vault`/`research` run without Vault
6. ✅ **Real Data Processing**: 35 features from actual trading history
7. ✅ **Two Model Families**: sklearn RandomForest (default) or PyTorch
   Transformer + RL agents (fallback)
8. ✅ **Optional Phase-2 AI**: AutoML / LLM / continuous / multimodal / NAS hooks
9. ✅ **Flexible Parameters**: easy customization for different needs
10. ✅ **Colored, Diagnostic Logging**: progress + `--check` diagnostics

## 🎯 **Recommended Usage**

### **For Development**
```bash
./run_training.sh --quick --debug
```

### **For Testing New Features**
```bash
./run_training.sh --method no-vault --limit 500
```

### **For Production Models**
```bash
./run_training.sh --production
```

### **For Continuous Integration**
```bash
./run_training.sh --method auto --limit 1000 --epochs 15
```

## 🎉 **Conclusion**

The unified training system provides a **single, reliable command** for all ELVIS training needs:

- **Automatically detects** the training method (research → postgres → no-vault)
- **Falls back gracefully** (postgres → no-vault) when the DB is unreachable
- **Integrates with** the local PostgreSQL `trades` table
- **Bypasses Vault** on the `no-vault` and `research` methods
- **Processes real trade data** with 35-feature / 6-target engineering
- **Trains** either sklearn RandomForest models (default `train_all_paper_trades.py`
  path, saved as joblib) or a PyTorch Transformer + RL agents (via
  `train_with_postgres.py` → `training/train_models.py`)

Use `./run_training.sh --help` for the complete, authoritative option list, or
`./run_training.sh --quick` to get started immediately.