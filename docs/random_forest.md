# Random Forest Model for Trading – ELVIS Project

![Random Forest Overview](ELVIS/images/random_forest.png)

---

## 📘 What is a Random Forest?

Random Forest is a **supervised machine learning algorithm** that is widely used for both **classification** and **regression** tasks. It belongs to the family of **ensemble learning methods**, which means it builds multiple models (in this case, decision trees) and combines their results to improve overall performance and robustness.

The core idea behind Random Forest is:
- Build many decision trees.
- Each tree is trained on a random subset of the data.
- During prediction, all trees vote (classification) or average their predictions (regression).

This strategy helps to **reduce overfitting** and improve **generalization** compared to single decision trees.

### 🌲 Why Use a Forest Instead of One Tree?

- Single decision trees are prone to overfitting and high variance.
- Random Forest introduces randomness in:
  - Data (bootstrap sampling)
  - Features (random subsets of features at each split)
- The aggregation of results leads to more stable and accurate predictions.

## 🎯 Use in the ELVIS Trading System

In our ELVIS trading platform, the `RandomForestModel` is a production-grade implementation leveraging **TensorFlow Decision Forests** (TFDF). It includes modern ML practices:

- Integrated with **Optuna** for automated hyperparameter tuning
- Supports **cross-validation** and **metric tracking**
- Offers **explainability** through SHAP values and feature importance
- Designed for **streaming updates** with a `partial_fit` interface
- Integrated with **Prometheus** and **Grafana** for MLOps

---

## 📦 Core Features

### ✅ Model Architecture
- Based on TFDF's `RandomForestModel`
- Uses `num_trees`, `max_depth`, and `min_examples` as primary hyperparameters
- Saved and loaded via `.ydf` format

### 🧠 Training
- Accepts pandas DataFrames (`X_train`, `y_train`)
- Optionally optimized using an **Optuna trial**
- Automatically saved after training

### 🔁 Cross-Validation
- Standard K-Fold, StratifiedKFold, and GroupKFold strategies
- Saves fold-wise metric plots
- Pushes average results to **Prometheus** for Grafana monitoring

### 📊 Evaluation
- Returns a dictionary with:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Loss
  - ROC AUC

### 📤 Prediction
- Uses TFDF's batch prediction API
- Returns flattened NumPy array

### 📈 Explainability
- SHAP value summary
- TFDF's built-in feature importance extractors:
  - `MEAN_DECREASE_IN_ACCURACY`
  - `NUM_AS_ROOT`
  - `SUM_SCORE`

---

## ⚙️ Advanced Integrations

### 🧪 Optuna Hyperparameter Optimization
- Automatically suggests `num_trees`, `max_depth`, and `min_examples`
- Integrated with cross-validation
- Can be extended to optimize learning rate, class weights, etc.

### 📡 Prometheus & Grafana
- After each `cross_validate()`:
  - Pushes average metrics to Pushgateway
  - Metrics are exposed as `rf_accuracy`, `rf_loss`, etc.
  - CSV version saved for external ingestion

### 📁 Visual Artifact Export
- Saves plots from `cross_validate()` as `.png` and `.svg`
- SHAP summary plot saved to `docs/plots/`
- Mermaid `.mmd` model architecture exported and rendered

---

## 🔁 Incremental Learning (Simulated)

TensorFlow Decision Forests does not natively support online learning. As a workaround:

- `partial_fit()` accumulates data batches
- Retrains on all previously seen data
- Simulates streaming adaptability

This ensures the model can continue learning from new market data.

---

## 🔌 Feature Pipeline

The model is powered by a modular feature pipeline supporting:
- OHLCV-based features (e.g., high-low ratio, rolling means)
- Custom financial indicators
- Rolling statistics
- Future integrations for sentiment and blockchain metrics

Each version of the feature set is hashed and tracked for reproducibility.

---

## 📁 File Structure

```
core/
├── models/
│   └── random_forest_model.py     # Main model implementation
├── features/
│   └── feature_pipeline.py        # Modular feature engineering
├── viz/
│   ├── export_utils.py            # SHAP, Prometheus, CSV exports

metrics/
├── model_metrics.csv
├── shap_summary.csv

prometheus/
├── pushgateway_config.yml

docs/
├── plots/
│   ├── shap_summary.png
│   └── cv_metrics.png
└── architecture_links.mmd
```

---

## 🛣️ Future Enhancements

- Full support for streaming ML frameworks like `river`
- Real-time alerts from Prometheus thresholds (e.g. F1 score dip)
- Grafana dashboards for CV history, SHAP trends, and live predictions
- Automated model drift detection and retraining triggers

---

## 📚 References

- [IBM Random Forest Explanation](https://www.ibm.com/think/topics/random-forest)
- [TensorFlow Decision Forests Documentation](https://www.tensorflow.org/decision_forests)
- [Optuna: Hyperparameter Optimization Framework](https://optuna.org/)
- [Prometheus Client for Python](https://github.com/prometheus/client_python)
- [SHAP: Explainable AI](https://shap.readthedocs.io/en/latest/)

---

> This document will be updated iteratively as new capabilities are deployed.
