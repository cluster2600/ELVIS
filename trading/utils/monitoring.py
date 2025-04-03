"""
Training monitoring utilities for tracking progress and metrics.
"""

import time
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path

class TrainingMonitor:
    """Monitors and visualizes training progress and metrics."""
    
    def __init__(self, config: Dict):
        """
        Initialize the training monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_time = time.time()
        self.metrics = {
            'train': {},
            'val': {}
        }
        self.best_metrics = {}
        self.best_epoch = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.early_stopping_metric = config.get('early_stopping_metric', 'val_loss')
        self.early_stopping_mode = config.get('early_stopping_mode', 'min')
        
        # Create visualization directory
        self.viz_dir = Path(config['log_dir']) / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
    def update_metrics(self, phase: str, metrics: Dict[str, float]):
        """
        Update metrics for a given phase (train/val).
        
        Args:
            phase: 'train' or 'val'
            metrics: Dictionary of metric names and values
        """
        for metric, value in metrics.items():
            if metric not in self.metrics[phase]:
                self.metrics[phase][metric] = []
            self.metrics[phase][metric].append(value)
            
        # Update best metrics
        if phase == 'val':
            self._update_best_metrics(metrics)
            
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics and check for early stopping."""
        if not self.best_metrics:
            self.best_metrics = metrics.copy()
            return
            
        current_value = metrics.get(self.early_stopping_metric)
        best_value = self.best_metrics.get(self.early_stopping_metric)
        
        if current_value is None or best_value is None:
            return
            
        if (self.early_stopping_mode == 'min' and current_value < best_value) or \
           (self.early_stopping_mode == 'max' and current_value > best_value):
            self.best_metrics = metrics.copy()
            self.best_epoch = len(self.metrics['val'][self.early_stopping_metric]) - 1
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
    def should_stop(self) -> bool:
        """Check if training should stop based on early stopping criteria."""
        return self.early_stopping_counter >= self.early_stopping_patience
        
    def get_metrics(self) -> Dict:
        """Get all recorded metrics."""
        return self.metrics
        
    def get_best_metrics(self) -> Dict:
        """Get best validation metrics."""
        return self.best_metrics
        
    def get_best_epoch(self) -> int:
        """Get epoch with best validation metrics."""
        return self.best_epoch
        
    def get_training_time(self) -> float:
        """Get total training time in seconds."""
        return time.time() - self.start_time
        
    def display_progress(self, epoch: int):
        """Display training progress with metrics."""
        # Create progress bar
        pbar = tqdm(total=len(self.metrics['train']), desc=f'Epoch {epoch+1}')
        pbar.update(1)
        
        # Display metrics
        train_metrics = {k: v[-1] for k, v in self.metrics['train'].items()}
        val_metrics = {k: v[-1] for k, v in self.metrics['val'].items()}
        
        metrics_str = " | ".join([
            f"train {k}: {v:.4f}" for k, v in train_metrics.items()
        ] + [
            f"val {k}: {v:.4f}" for k, v in val_metrics.items()
        ])
        
        pbar.set_postfix_str(metrics_str)
        pbar.close()
        
    def plot_metrics(self, save: bool = True):
        """Plot training and validation metrics."""
        plt.figure(figsize=(12, 8))
        
        for phase in ['train', 'val']:
            for metric, values in self.metrics[phase].items():
                plt.plot(values, label=f'{phase} {metric}')
                
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(self.viz_dir / 'metrics.png')
            plt.close()
        else:
            plt.show()
            
    def plot_learning_curves(self, save: bool = True):
        """Plot learning curves for different metrics."""
        n_metrics = len(self.metrics['train'])
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for i, (metric, train_values) in enumerate(self.metrics['train'].items()):
            val_values = self.metrics['val'].get(metric, [])
            
            axes[i].plot(train_values, label='train')
            if val_values:
                axes[i].plot(val_values, label='val')
                
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Learning Curve')
            axes[i].legend()
            axes[i].grid(True)
            
        plt.tight_layout()
        
        if save:
            plt.savefig(self.viz_dir / 'learning_curves.png')
            plt.close()
        else:
            plt.show()
            
    def save_summary(self):
        """Save training summary to JSON file."""
        summary = {
            'config': self.config,
            'metrics': self.metrics,
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch,
            'training_time': self.get_training_time(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.viz_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2) 