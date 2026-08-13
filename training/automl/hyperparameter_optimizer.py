#!/usr/bin/env python3.14
"""
AutoML Hyperparameter Optimization for ELVIS Trading Models
Integrates Optuna for intelligent hyperparameter tuning with advanced features
"""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Optuna is optional and is not part of the canonical ELVIS dependency sets.
# Guard the import so this module stays importable; optimizer methods that call
# into Optuna run only when an operator installs it separately.
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

# PyTorch is an optional dependency of this module and part of the ``ml`` extra.
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""

    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    n_trials: int
    optimization_time: float
    model_type: str
    study_name: str
    timestamp: datetime

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration for hyperparameter spaces"""

    name: str
    param_space: Dict[str, Any]
    scorer: str = "accuracy"
    cv_folds: int = 5
    early_stopping: bool = True


class HyperparameterOptimizer:
    """Advanced hyperparameter optimizer using Optuna"""

    def __init__(
        self,
        study_name: str = "elvis_optimization",
        storage_url: Optional[str] = None,
        optimization_direction: str = "maximize",
        pruner_type: str = "median",
        sampler_type: str = "tpe",
    ):

        self.study_name = study_name
        self.storage_url = storage_url or f"sqlite:///optuna_{study_name}.db"
        self.optimization_direction = optimization_direction

        # Initialize Optuna components
        self.pruner = self._create_pruner(pruner_type)
        self.sampler = self._create_sampler(sampler_type)

        # Results storage
        self.results_history = []
        self.best_models = {}

        # Supported model configurations
        self.model_configs = self._setup_model_configs()

    def _create_pruner(self, pruner_type: str):
        """Create Optuna pruner"""
        pruner_map = {
            "median": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            "percentile": optuna.pruners.PercentilePruner(25.0),
            "hyperband": optuna.pruners.HyperbandPruner(),
            "nop": optuna.pruners.NopPruner(),
        }
        return pruner_map.get(pruner_type, optuna.pruners.MedianPruner())

    def _create_sampler(self, sampler_type: str):
        """Create Optuna sampler"""
        sampler_map = {
            "tpe": optuna.samplers.TPESampler(seed=42),
            "random": optuna.samplers.RandomSampler(seed=42),
            "cmaes": optuna.samplers.CmaEsSampler(seed=42),
            "grid": optuna.samplers.GridSampler(),
        }
        return sampler_map.get(sampler_type, optuna.samplers.TPESampler(seed=42))

    def _setup_model_configs(self) -> Dict[str, ModelConfig]:
        """Setup hyperparameter spaces for different models"""

        configs = {
            "random_forest": ModelConfig(
                name="random_forest",
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "max_depth": ("int", 3, 20),
                    "min_samples_split": ("int", 2, 20),
                    "min_samples_leaf": ("int", 1, 10),
                    "max_features": ("categorical", ["auto", "sqrt", "log2"]),
                    "bootstrap": ("categorical", [True, False]),
                    "criterion": ("categorical", ["gini", "entropy"]),
                },
                scorer="f1_weighted",
                cv_folds=5,
            ),
            "gradient_boosting": ModelConfig(
                name="gradient_boosting",
                param_space={
                    "n_estimators": ("int", 50, 300),
                    "learning_rate": ("float", 0.01, 0.3),
                    "max_depth": ("int", 3, 15),
                    "min_samples_split": ("int", 2, 20),
                    "min_samples_leaf": ("int", 1, 10),
                    "subsample": ("float", 0.6, 1.0),
                    "max_features": ("categorical", ["auto", "sqrt", "log2"]),
                },
                scorer="f1_weighted",
                cv_folds=5,
            ),
            "neural_network": ModelConfig(
                name="neural_network",
                param_space={
                    "hidden_layer_sizes": (
                        "categorical",
                        [
                            (64,),
                            (128,),
                            (256,),
                            (64, 32),
                            (128, 64),
                            (256, 128),
                            (128, 64, 32),
                            (256, 128, 64),
                        ],
                    ),
                    "activation": ("categorical", ["relu", "tanh", "logistic"]),
                    "alpha": ("float", 1e-5, 1e-1, True),  # log=True
                    "learning_rate": (
                        "categorical",
                        ["constant", "invscaling", "adaptive"],
                    ),
                    "learning_rate_init": ("float", 1e-4, 1e-1, True),
                    "max_iter": ("int", 100, 500),
                    "early_stopping": ("categorical", [True]),
                    "validation_fraction": ("float", 0.1, 0.3),
                },
                scorer="accuracy",
                cv_folds=3,  # Reduced for neural networks
            ),
            "svm": ModelConfig(
                name="svm",
                param_space={
                    "C": ("float", 0.1, 100, True),
                    "kernel": ("categorical", ["rbf", "poly", "sigmoid"]),
                    "degree": ("int", 2, 5),  # for poly kernel
                    "gamma": ("categorical", ["scale", "auto"]),
                    "probability": ("categorical", [True]),  # for trading confidence
                },
                scorer="accuracy",
                cv_folds=3,  # SVM can be slow
            ),
        }

        # Add the supported deep-learning config when PyTorch is available.
        if TORCH_AVAILABLE:
            configs["pytorch_nn"] = self._get_pytorch_config()

        return configs

    def _get_pytorch_config(self) -> ModelConfig:
        """PyTorch neural network configuration"""
        return ModelConfig(
            name="pytorch_nn",
            param_space={
                "hidden_dims": (
                    "categorical",
                    [[64], [128], [256], [64, 32], [128, 64], [256, 128, 64]],
                ),
                "dropout_rate": ("float", 0.0, 0.5),
                "learning_rate": ("float", 1e-5, 1e-2, True),
                "batch_size": ("categorical", [16, 32, 64, 128]),
                "epochs": ("int", 50, 200),
                "weight_decay": ("float", 1e-6, 1e-2, True),
            },
            scorer="accuracy",
            cv_folds=3,
        )

    def optimize_model(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        custom_objective: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize hyperparameters for a specific model type

        Args:
            model_type: Type of model to optimize
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            custom_objective: Custom objective function
            **kwargs: Additional arguments

        Returns:
            OptimizationResult containing best parameters and metrics
        """

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Hyperparameter optimization requires the 'optuna' package, "
                "which is not installed. Install it with `pip install optuna`."
            )

        if model_type not in self.model_configs:
            raise ValueError(
                f"Model type '{model_type}' not supported. Available: {list(self.model_configs.keys())}"
            )

        config = self.model_configs[model_type]

        # Create or load study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_type}",
            storage=self.storage_url,
            direction=self.optimization_direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,
        )

        # Define objective function
        objective = custom_objective or self._create_objective_function(
            config, X, y, **kwargs
        )

        # Run optimization
        start_time = time.time()
        logger.info(f"🚀 Starting hyperparameter optimization for {model_type}")
        logger.info(
            f"   Trials: {n_trials}, Timeout: {timeout}s, Direction: {self.optimization_direction}"
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._trial_callback],
        )

        optimization_time = time.time() - start_time

        # Extract results
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=study.best_trial.number,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            model_type=model_type,
            study_name=study.study_name,
            timestamp=datetime.now(),
        )

        self.results_history.append(result)

        # Train and store best model
        best_model = self._train_best_model(model_type, result.best_params, X, y)
        self.best_models[model_type] = best_model

        logger.info(f"✅ Optimization complete for {model_type}")
        logger.info(f"   Best score: {result.best_score:.4f}")
        logger.info(f"   Best params: {result.best_params}")
        logger.info(f"   Time taken: {optimization_time:.2f}s")

        return result

    def _create_objective_function(
        self, config: ModelConfig, X: np.ndarray, y: np.ndarray, **kwargs
    ):
        """Create objective function for optimization"""

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in config.param_space.items():
                param_type = param_config[0]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config[1], param_config[2]
                    )
                elif param_type == "float":
                    log = len(param_config) > 3 and param_config[3]
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[1], param_config[2], log=log
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config[1]
                    )

            # Create and train model
            try:
                model = self._create_model(config.name, params)

                # Time series cross-validation for trading data
                cv = TimeSeriesSplit(n_splits=config.cv_folds)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=config.scorer, n_jobs=-1
                )

                # Handle potential issues
                if np.isnan(scores).any() or len(scores) == 0:
                    return 0.0

                score = np.mean(scores)

                # Report intermediate result for pruning
                trial.report(score, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        return objective

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters"""

        if model_type == "random_forest":
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(**params, random_state=42)

        elif model_type == "neural_network":
            return MLPClassifier(**params, random_state=42)

        elif model_type == "svm":
            return SVC(**params, random_state=42)

        elif model_type == "pytorch_nn" and TORCH_AVAILABLE:
            return self._create_pytorch_model(params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_pytorch_model(self, params: Dict[str, Any]):
        """Create PyTorch model with parameters"""
        # This would be implemented based on specific PyTorch architecture needs
        # Placeholder for now
        pass

    def _train_best_model(
        self, model_type: str, best_params: Dict[str, Any], X: np.ndarray, y: np.ndarray
    ):
        """Train the best model with optimized parameters"""
        model = self._create_model(model_type, best_params)
        model.fit(X, y)
        return model

    def _trial_callback(self, study, trial):
        """Callback function for trial completion"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number}: {trial.value:.4f} - {trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.debug(f"Trial {trial.number}: Pruned")

    def optimize_multiple_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: List[str],
        n_trials_per_model: int = 50,
        timeout_per_model: Optional[int] = None,
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize multiple models and return comparison results

        Args:
            X: Feature matrix
            y: Target vector
            model_types: List of model types to optimize
            n_trials_per_model: Trials per model
            timeout_per_model: Timeout per model in seconds

        Returns:
            Dictionary mapping model types to optimization results
        """

        results = {}
        total_start_time = time.time()

        logger.info(f"🚀 Starting multi-model optimization")
        logger.info(f"   Models: {model_types}")
        logger.info(f"   Trials per model: {n_trials_per_model}")

        for i, model_type in enumerate(model_types, 1):
            logger.info(f"\n📊 Optimizing model {i}/{len(model_types)}: {model_type}")

            try:
                result = self.optimize_model(
                    model_type=model_type,
                    X=X,
                    y=y,
                    n_trials=n_trials_per_model,
                    timeout=timeout_per_model,
                )
                results[model_type] = result

            except Exception as e:
                logger.error(f"❌ Failed to optimize {model_type}: {e}")
                continue

        total_time = time.time() - total_start_time

        # Generate comparison report
        self._generate_comparison_report(results, total_time)

        return results

    def _generate_comparison_report(
        self, results: Dict[str, OptimizationResult], total_time: float
    ):
        """Generate comparison report for multiple model optimization"""

        if not results:
            logger.warning("No successful optimizations to compare")
            return

        logger.info(f"\n🏆 Model Comparison Report")
        logger.info(f"{'='*50}")

        # Sort by best score
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].best_score, reverse=True
        )

        for rank, (model_type, result) in enumerate(sorted_results, 1):
            logger.info(f"{rank}. {model_type.upper()}")
            logger.info(f"   Score: {result.best_score:.4f}")
            logger.info(f"   Time: {result.optimization_time:.2f}s")
            logger.info(f"   Trials: {result.n_trials}")

        best_model = sorted_results[0]
        logger.info(
            f"\n🥇 Best Model: {best_model[0]} (Score: {best_model[1].best_score:.4f})"
        )
        logger.info(f"⏱️  Total Time: {total_time:.2f}s")

    def get_study_statistics(self, model_type: str) -> Dict[str, Any]:
        """Get detailed statistics for a study"""

        study = optuna.load_study(
            study_name=f"{self.study_name}_{model_type}", storage=self.storage_url
        )

        stats = {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "datetime_start": min(
                t.datetime_start for t in study.trials if t.datetime_start
            ),
            "datetime_complete": max(
                t.datetime_complete for t in study.trials if t.datetime_complete
            ),
            "pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "complete_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "failed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        }

        return stats

    def save_results(self, output_path: Optional[Path] = None):
        """Save optimization results to file"""

        if output_path is None:
            timestamp = int(time.time())
            output_path = Path(f"optimization_results_{timestamp}.json")

        # Prepare data for serialization
        results_data = {
            "study_name": self.study_name,
            "optimization_timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in self.results_history],
            "configuration": {
                "optimization_direction": self.optimization_direction,
                "pruner": str(type(self.pruner).__name__),
                "sampler": str(type(self.sampler).__name__),
                "storage_url": self.storage_url,
            },
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"💾 Results saved to: {output_path}")
        return output_path

    def load_best_model(self, model_type: str):
        """Load the best trained model for a given type"""
        return self.best_models.get(model_type)


# Usage example and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate sample trading-like data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    print("🚀 Testing AutoML Hyperparameter Optimizer")

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name="elvis_automl_test", optimization_direction="maximize"
    )

    # Test single model optimization
    print("\n📊 Single Model Optimization (Random Forest)")
    result = optimizer.optimize_model(model_type="random_forest", X=X, y=y, n_trials=20)

    print(f"✅ Best score: {result.best_score:.4f}")
    print(f"⏱️  Time taken: {result.optimization_time:.2f}s")

    # Test multi-model optimization
    print("\n📊 Multi-Model Optimization")
    models_to_test = ["random_forest", "gradient_boosting", "neural_network"]
    results = optimizer.optimize_multiple_models(
        X=X, y=y, model_types=models_to_test, n_trials_per_model=15
    )

    # Save results
    results_path = optimizer.save_results()
    print(f"💾 Results saved to: {results_path}")

    print("\n🎉 AutoML testing complete!")
