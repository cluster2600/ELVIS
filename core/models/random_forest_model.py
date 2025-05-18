"""
Random Forest model for the ELVIS project.
Implements a Random Forest using TensorFlow Decision Forests,
including training, evaluation, interpretability, and cross-validation with Optuna support.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from core.models.base_model import BaseModel
import config

# Define FILE_PATHS here if not defined in config
if not hasattr(config, 'FILE_PATHS'):
    FILE_PATHS = {
        'TRAIN_RESULTS_DIR': 'data/models'
    }
else:
    FILE_PATHS = config.FILE_PATHS

# --------------------
# Decorators
# --------------------

def log_time(func):
    """Decorator to log execution time of methods."""
    def wrapper(self, *args, **kwargs):
        start = time.time()
        self.logger.info(f"Starting '{func.__name__}'...")
        result = func(self, *args, **kwargs)
        duration = time.time() - start
        self.logger.info(f"Finished '{func.__name__}' in {duration:.2f} seconds")
        return result
    return wrapper

# --------------------
# RandomForestModel
# --------------------

class RandomForestModel(BaseModel):
    """
    Random Forest model for trading using TensorFlow Decision Forests.
    Includes training, evaluation, interpretability, and CV with Optuna.
    """

    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.num_trees = kwargs.get('num_trees', 100)
        self.max_depth = kwargs.get('max_depth', 20)
        self.min_examples = kwargs.get('min_examples', 5)
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'model_rf.ydf'))
        self.model = None

    @log_time
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, trial: Optional[optuna.trial.Trial] = None) -> None:
        if trial:
            self.num_trees = trial.suggest_int('num_trees', 50, 300)
            self.max_depth = trial.suggest_int('max_depth', 5, 30)
            self.min_examples = trial.suggest_int('min_examples', 1, 20)
            self.logger.info(f"Optuna trial: num_trees={self.num_trees}, max_depth={self.max_depth}, min_examples={self.min_examples}")

        label = 'target_label'
        df = X_train.copy()
        df[label] = y_train
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label)

        self.model = tfdf.keras.RandomForestModel(
            num_trees=self.num_trees,
            max_depth=self.max_depth,
            min_examples=self.min_examples,
            verbose=2
        )
        self.model.fit(ds)
        self.save()

    @log_time
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            self.load_model()
            if self.model is None:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': 0.0}

        df = X_test.copy()
        df['target_label'] = y_test
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label='target_label')
        eval_res = self.model.evaluate(ds, return_dict=True)

        return {
            'accuracy': eval_res.get('accuracy', 0.0),
            'loss': eval_res.get('loss', 0.0),
            'precision': eval_res.get('precision_at_1', 0.0),
            'recall': eval_res.get('recall_at_1', 0.0),
            'f1': eval_res.get('f1_score', 0.0)
        }

    @log_time
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load_model()
            if self.model is None:
                return np.zeros(len(X))

        ds = tfdf.keras.pd_dataframe_to_tf_dataset(X, label=None)
        return self.model.predict(ds).flatten()

    @log_time
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, trial: Optional[optuna.trial.Trial] = None) -> Dict[str, float]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'loss', 'roc_auc']}

        for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(X), total=n_splits, desc="Cross-validation")):
            self.logger.info(f"Fold {fold + 1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.train(X_train, y_train, trial=trial)
            metrics = self.evaluate(X_test, y_test)
            y_pred = self.predict(X_test)

            for k in ['accuracy', 'precision', 'recall', 'f1', 'loss']:
                scores[k].append(metrics.get(k, 0.0))

            try:
                if len(np.unique(y_test)) == 2:
                    scores['roc_auc'].append(roc_auc_score(y_test, y_pred))
                else:
                    self.logger.warning("ROC AUC not supported for multiclass")
                    scores['roc_auc'].append(0.0)
            except Exception as e:
                self.logger.warning(f"ROC AUC error: {e}")
                scores['roc_auc'].append(0.0)

        avg = {k: float(np.mean(v)) for k, v in scores.items()}
        self.logger.info(f"CV Average: {avg}")

        plt.figure(figsize=(10, 6))
        for k, v in scores.items():
            plt.plot(range(1, n_splits + 1), v, label=k)
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.title("Cross-validation Metrics per Fold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return avg

    @log_time
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            self.load_model()
            if self.model is None:
                return pd.DataFrame()

        inspector = self.model.make_inspector()
        importance = inspector.variable_importances()

        for key in ['MEAN_DECREASE_IN_ACCURACY', 'NUM_AS_ROOT', 'SUM_SCORE']:
            if key in importance:
                df = pd.DataFrame(importance[key], columns=['feature', 'importance'])
                return df.sort_values('importance', ascending=False)

        return pd.DataFrame()

    def save(self, path: Optional[str] = None) -> None:
        if self.model is None:
            self.logger.warning("No model to save")
            return
        save_path = path if path else self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model not found at {self.model_path}")
            return
        self.model = tf.keras.models.load_model(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")

    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        logger = logging.getLogger('RandomForestModel')
        model_instance = cls(logger)
        model_instance.model_path = path
        model_instance.load_model()
        return model_instance

    def get_params(self) -> Dict[str, Any]:
        return {
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'min_examples': self.min_examples
        }

    def set_params(self, **params) -> None:
        self.num_trees = params.get('num_trees', self.num_trees)
        self.max_depth = params.get('max_depth', self.max_depth)
        self.min_examples = params.get('min_examples', self.min_examples)
