# ELVIS Trading Bot - Project Cleanup Analysis

## Files and Directories to Delete/Clean

### 1. Duplicate Files
- **FUTURE_IMPROVEMENTS.md** (root) - Duplicate of docs/future_improvements.md
- **ensemble_models.py** (root) - Functionality exists in core/models/ensemble_model.py
- **training.py** (root) - Duplicate of training/train_models.py
- **trading/models/model_trainer.py** - Duplicate of training/models/model_trainer.py
- **trading/rl_agents.py** - Duplicate of training/models/rl_agents.py
- **trading/performance_monitor.py** - Duplicate of core/metrics/performance_monitor.py
- **utils/price_fetcher.py** - Duplicate of trading/data/price_fetcher.py

### 2. macOS System Files (.DS_Store)
- **trading/.DS_Store**
- **trading/strategies/.DS_Store**
- **training/.DS_Store**
- **training/models/.DS_Store**
- **training/training_results/.DS_Store**

### 3. Virtual Environment Directories (Should be in .gitignore)
- **venv310/** - Old Python 3.10 virtual environment
- **env-coreml/** - CoreML specific environment
- **env-ydf/** - YDF specific environment
- **venv/** - Another virtual environment

### 4. Build/Package Artifacts
- **elvis.egg-info/** - Python package build artifacts
- **tensorflow/** - Large TensorFlow source (if not needed)

### 5. Obsolete/Unused Files
- **Miniforge3-MacOSX-arm64.sh** - Installation script, not needed in repo
- **test_*.py** (root level) - Should be in tests/ directory
- **function_*.py** (root level) - Standalone functions, should be organized
- **create_coreml_model.py** - One-time script, can be archived
- **predict_with_ydf.py** - Standalone script, should be in scripts/

### 6. Old Training Results (Archive or Clean)
- **training/training_results/res_2025-02-23__*/** - Old training runs (keep latest only)
- **model_rf.ydf/** (root) - Should be in models/ directory
- **nn_model.h5** (root) - Should be in models/ directory
- **NNModel.mlpackage/** (root) - Should be in models/ directory

### 7. Configuration Duplicates
- **trading/config/** - Duplicate configs exist in config/
- Multiple **model_config.yaml** files in different locations

### 8. Empty or Minimal Directories
- **services/** - Check if contains useful content
- **examples/** - Check if contains useful content
- **images/** - Check if contains useful content

### 9. Log and Data Files
- **logs/** - Old log files (keep structure, clean old logs)
- **data/queries.active** - Temporary file
- **paper_trades.db** - Development database
- **profile-metrics.json** - Profiling output

## Recommended Actions

### Phase 1: Remove System Files and Build Artifacts
1. Delete all .DS_Store files
2. Remove virtual environment directories
3. Remove build artifacts (elvis.egg-info)
4. Clean up installation scripts

### Phase 2: Consolidate Duplicate Files
1. Remove duplicate model trainers
2. Consolidate ensemble models
3. Remove duplicate RL agents
4. Consolidate performance monitors
5. Remove duplicate price fetchers

### Phase 3: Organize Standalone Files
1. Move test files to tests/ directory
2. Move function files to appropriate modules
3. Move model files to models/ directory
4. Move scripts to scripts/ directory

### Phase 4: Clean Training Results
1. Archive old training results
2. Keep only latest successful runs
3. Organize model artifacts

### Phase 5: Update Documentation
1. Update README.md with new structure
2. Update CHANGELOG.md with cleanup actions
3. Ensure all documentation reflects new organization

## Files to Keep and Organize Better
- Core functionality in core/
- Trading system in trading/
- Training pipeline in training/
- Utilities in utils/
- Documentation in docs/
- Tests in tests/
- Configuration in config/
