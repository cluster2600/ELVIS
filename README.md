# Project Overview

![Project Image](./images/elvis.png)

This project includes a trading application with machine learning models, feature pipelines, and visualization tools.

<!-- Removed non-existent architecture_links.svg image reference -->


## Core Models

- `core/models/random_forest_model.py`: Contains the RandomForestModel class and related utilities. Recently updated to include Prometheus metrics integration for cross-validation metrics, enabling monitoring of model evaluation results.

## Feature Engineering

- `core/features/feature_pipeline.py`: New module responsible for feature extraction and transformation pipelines used in model training and evaluation.

## Visualization

- `core/viz/streamlit_dashboard.py`: New Streamlit-based dashboard for interactive visualization of model results and trading metrics.
- `core/viz/export_utils.py`: New utility functions to support exporting visualizations and data from the dashboard.

## Documentation

- `docs/plots/`: Directory containing SHAP and cross-validation plots in PNG and SVG formats for model interpretability and evaluation.
- `docs/bot_architecture_mermaid.md`: Mermaid diagram file documenting the bot architecture and data flow within the project.

## Change Log

Please refer to `CHANGELOG.md` for detailed updates and version history.
