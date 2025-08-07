"""
Hyperparameter Optimization for ELVIS Random Forest
Advanced optimization with Optuna integration and cross-validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for Random Forest trading models.
    
    Features:
    - Optuna-based Bayesian optimization
    - Time-series aware cross-validation
    - Multi-objective optimization
    - Early stopping and pruning
    - Optimization history tracking
    """
    
    def __init__(self, 
                 model_class: Any,
                 logger: logging.Logger = None,
                 study_name: str = "rf_optimization",
                 storage_path: str = "optimization_studies",
                 n_jobs: int = -1):
        
        self.model_class = model_class
        self.logger = logger or logging.getLogger(__name__)
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_jobs = n_jobs
        
        # Optimization settings
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna not available - hyperparameter optimization disabled")
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.logger.info(f"🔧 Hyperparameter Optimizer initialized: {study_name}")
    
    def optimize(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_val: Optional[pd.DataFrame] = None,
                 y_val: Optional[pd.Series] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = 3600,  # 1 hour
                 cv_folds: int = 5,
                 scoring_metric: str = 'f1_weighted',
                 direction: str = 'maximize') -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize
            direction: 'maximize' or 'minimize'
            
        Returns:
            Dictionary with optimization results
        """
        
        self.logger.info(f"🚀 Starting hyperparameter optimization with {n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(
                trial, X_train, y_train, X_val, y_val, 
                cv_folds, scoring_metric
            )
        
        try:
            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,  # Use single job to avoid conflicts
                show_progress_bar=True
            )
            
            # Store results
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            # Save optimization results
            results = self._save_optimization_results(study, X_train, y_train)
            
            self.logger.info(f"✅ Optimization completed!")
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Best params: {self.best_params}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return {'error': str(e)}
    
    def _objective_function(self,
                           trial: 'optuna.Trial',
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: Optional[pd.DataFrame],
                           y_val: Optional[pd.Series],
                           cv_folds: int,
                           scoring_metric: str) -> float:
        """Objective function for Optuna optimization."""
        
        try:
            # Get hyperparameters for this trial
            params = self._suggest_hyperparameters(trial)
            
            # Create model with suggested parameters
            model = self.model_class(
                logger=self.logger,
                enable_optuna=False,  # Disable nested optimization
                enable_shap=False,    # Disable SHAP during optimization
                enable_monitoring=False  # Disable monitoring during optimization
            )
            
            # Set model parameters
            model.model = model.model.__class__(**params, random_state=42)
            
            # Evaluate using cross-validation or validation set
            if X_val is not None and y_val is not None:
                # Use validation set
                model.model.fit(X_train, y_train)
                y_pred = model.model.predict(X_val)
                score = self._calculate_score(y_val, y_pred, scoring_metric)
            else:
                # Use cross-validation
                score = self._cross_validate_model(
                    model.model, X_train, y_train, cv_folds, scoring_metric
                )
            
            # Report intermediate value for pruning
            trial.report(score, step=1)
            
            # Prune if performance is poor
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -np.inf if scoring_metric in ['accuracy', 'f1', 'precision', 'recall'] else np.inf
    
    def _suggest_hyperparameters(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """Suggest hyperparameters for Random Forest."""
        
        return {
            # Core RF parameters
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            
            # Class balancing
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            
            # Tree quality parameters
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1),
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 50, 100, 200, 500]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
            
            # Performance parameters
            'n_jobs': self.n_jobs,
            'random_state': 42
        }
    
    def _cross_validate_model(self,
                             model: Any,
                             X: pd.DataFrame,
                             y: pd.Series,
                             cv_folds: int,
                             scoring_metric: str) -> float:
        """Perform cross-validation with time-series awareness."""
        
        # Use TimeSeriesSplit for time-aware validation
        cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
        
        # Get scorer
        scorer = self._get_scorer(scoring_metric)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv_splitter,
            scoring=scorer,
            n_jobs=1,  # Use single job to avoid conflicts
            error_score='raise'
        )
        
        return cv_scores.mean()
    
    def _get_scorer(self, scoring_metric: str):
        """Get sklearn scorer for the specified metric."""
        
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted')
        }
        
        return scorers.get(scoring_metric, scorers['f1_weighted'])
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, scoring_metric: str) -> float:
        """Calculate score for a specific metric."""
        
        if scoring_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif scoring_metric == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted')
        elif scoring_metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro')
        elif scoring_metric == 'precision_weighted':
            return precision_score(y_true, y_pred, average='weighted')
        elif scoring_metric == 'recall_weighted':
            return recall_score(y_true, y_pred, average='weighted')
        else:
            return accuracy_score(y_true, y_pred)
    
    def _save_optimization_results(self, study: 'optuna.Study', X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Save optimization results and generate report."""
        
        results = {
            'study_name': study.study_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'n_features': len(X_train.columns),
            'n_classes': len(np.unique(y_train))
        }
        
        # Save detailed results
        results_file = os.path.join(self.storage_path, f"{study.study_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trial history
        trial_data = []
        for trial in study.trials:
            trial_info = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            trial_data.append(trial_info)
        
        trials_file = os.path.join(self.storage_path, f"{study.study_name}_trials.json")
        with open(trials_file, 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Generate optimization report
        results['optimization_report'] = self._generate_optimization_report(study, trial_data)
        
        self.logger.info(f"📊 Optimization results saved to {results_file}")
        
        return results
    
    def _generate_optimization_report(self, study: 'optuna.Study', trial_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        completed_trials = [t for t in trial_data if t['state'] == 'COMPLETE']
        
        if not completed_trials:
            return {'error': 'No completed trials'}
        
        values = [t['value'] for t in completed_trials]
        
        report = {
            'summary': {
                'total_trials': len(trial_data),
                'completed_trials': len(completed_trials),
                'pruned_trials': len([t for t in trial_data if t['state'] == 'PRUNED']),
                'failed_trials': len([t for t in trial_data if t['state'] == 'FAIL']),
                'best_value': study.best_value,
                'best_trial_number': study.best_trial.number,
            },
            'performance_stats': {
                'mean_score': np.mean(values),
                'std_score': np.std(values),
                'min_score': np.min(values),
                'max_score': np.max(values),
                'median_score': np.median(values),
                'score_improvement': study.best_value - np.mean(values[:10]) if len(values) >= 10 else 0
            },
            'parameter_importance': self._calculate_parameter_importance(study),
            'convergence_analysis': self._analyze_convergence(values),
            'recommendations': self._generate_recommendations(study, values)
        }
        
        return report
    
    def _calculate_parameter_importance(self, study: 'optuna.Study') -> Dict[str, float]:
        """Calculate parameter importance using Optuna's built-in method."""
        
        try:
            importance = optuna.importance.get_param_importances(study)
            return {k: float(v) for k, v in importance.items()}
        except Exception as e:
            self.logger.warning(f"Parameter importance calculation failed: {e}")
            return {}
    
    def _analyze_convergence(self, values: List[float]) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        
        if len(values) < 10:
            return {'status': 'insufficient_data'}
        
        # Calculate rolling best
        rolling_best = []
        current_best = values[0]
        for value in values:
            if value > current_best:  # Assuming maximization
                current_best = value
            rolling_best.append(current_best)
        
        # Check for convergence
        recent_improvement = rolling_best[-1] - rolling_best[-10]
        improvement_rate = recent_improvement / 10
        
        convergence_status = 'converged' if improvement_rate < 0.001 else 'improving'
        
        return {
            'status': convergence_status,
            'improvement_rate': improvement_rate,
            'trials_without_improvement': len(values) - np.argmax(values) - 1,
            'final_best_score': rolling_best[-1],
            'improvement_from_start': rolling_best[-1] - values[0]
        }
    
    def _generate_recommendations(self, study: 'optuna.Study', values: List[float]) -> Dict[str, str]:
        """Generate optimization recommendations."""
        
        recommendations = {}
        
        # Performance recommendations
        if len(values) > 0:
            if study.best_value > np.mean(values) + 2 * np.std(values):
                recommendations['performance'] = "Excellent optimization results achieved"
            elif study.best_value > np.mean(values):
                recommendations['performance'] = "Good optimization results, consider more trials"
            else:
                recommendations['performance'] = "Limited improvement found, review feature engineering"
        
        # Parameter recommendations
        best_params = study.best_params
        
        if best_params.get('n_estimators', 0) >= 400:
            recommendations['n_estimators'] = "Consider reducing n_estimators for faster training"
        elif best_params.get('n_estimators', 0) <= 75:
            recommendations['n_estimators'] = "Consider increasing n_estimators for better performance"
        
        if best_params.get('max_depth', 0) >= 20:
            recommendations['max_depth'] = "High max_depth may cause overfitting"
        elif best_params.get('max_depth', 0) <= 5:
            recommendations['max_depth'] = "Low max_depth may cause underfitting"
        
        # Convergence recommendations
        convergence = self._analyze_convergence(values)
        if convergence.get('status') == 'improving':
            recommendations['trials'] = "Run more trials for better optimization"
        elif convergence.get('trials_without_improvement', 0) > 20:
            recommendations['exploration'] = "Consider different parameter spaces"
        
        return recommendations
    
    def optimize_multiple_metrics(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 metrics: List[str] = None,
                                 n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize for multiple metrics using multiple studies.
        
        Args:
            X_train: Training features
            y_train: Training labels
            metrics: List of metrics to optimize
            n_trials: Number of trials per metric
            
        Returns:
            Combined optimization results
        """
        
        if metrics is None:
            metrics = ['f1_weighted', 'accuracy', 'precision_weighted']
        
        self.logger.info(f"🎯 Multi-metric optimization for: {metrics}")
        
        results = {}
        
        for metric in metrics:
            self.logger.info(f"Optimizing for {metric}...")
            
            metric_results = self.optimize(
                X_train, y_train,
                n_trials=n_trials,
                scoring_metric=metric,
                direction='maximize'
            )
            
            results[metric] = metric_results
        
        # Find best overall parameters (weighted combination)
        best_overall = self._combine_metric_results(results)
        results['best_overall'] = best_overall
        
        return results
    
    def _combine_metric_results(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple metric optimization results."""
        
        # Simple approach: use the parameters from the best F1 score
        # More sophisticated approaches could include Pareto optimization
        
        best_metric = 'f1_weighted'
        if best_metric in metric_results:
            return {
                'params': metric_results[best_metric]['best_params'],
                'combined_score': metric_results[best_metric]['best_value'],
                'selection_method': f'best_{best_metric}'
            }
        
        # Fallback to first available metric
        first_metric = list(metric_results.keys())[0]
        return {
            'params': metric_results[first_metric]['best_params'],
            'combined_score': metric_results[first_metric]['best_value'],
            'selection_method': f'fallback_{first_metric}'
        }
    
    def load_best_parameters(self, study_name: str = None) -> Optional[Dict[str, Any]]:
        """Load best parameters from a previous optimization."""
        
        if study_name is None:
            # Find most recent study
            study_files = [f for f in os.listdir(self.storage_path) if f.endswith('_results.json')]
            if not study_files:
                return None
            study_files.sort(reverse=True)
            results_file = os.path.join(self.storage_path, study_files[0])
        else:
            results_file = os.path.join(self.storage_path, f"{study_name}_results.json")
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            self.best_params = results.get('best_params')
            self.best_score = results.get('best_value')
            
            self.logger.info(f"✅ Loaded best parameters from {results_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {e}")
            return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization studies."""
        
        summary = {
            'studies': [],
            'total_trials': 0,
            'best_overall_score': None,
            'best_overall_params': None
        }
        
        try:
            result_files = [f for f in os.listdir(self.storage_path) if f.endswith('_results.json')]
            
            for file in result_files:
                file_path = os.path.join(self.storage_path, file)
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                study_info = {
                    'study_name': results.get('study_name'),
                    'best_value': results.get('best_value'),
                    'n_trials': results.get('n_trials'),
                    'optimization_time': results.get('optimization_time')
                }
                
                summary['studies'].append(study_info)
                summary['total_trials'] += results.get('n_trials', 0)
                
                # Track best overall
                if (summary['best_overall_score'] is None or 
                    results.get('best_value', 0) > summary['best_overall_score']):
                    summary['best_overall_score'] = results.get('best_value')
                    summary['best_overall_params'] = results.get('best_params')
            
            summary['n_studies'] = len(summary['studies'])
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization summary: {e}")
        
        return summary