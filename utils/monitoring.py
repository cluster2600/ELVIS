"""
Basic TrainingMonitor implementation for ELVIS Trading Bot.
Tracks training and validation metrics, supports early stopping, and displays progress.
"""

import time
import logging
from prometheus_client import Counter, Histogram, Gauge

try:
    from prometheus_client.gateway import push_to_gateway
    HAS_PUSHGATEWAY = True
except ImportError:
    HAS_PUSHGATEWAY = False

class TrainingMonitor:
    def __init__(self, config=None):
        self.config = config or {}
        self.metrics = {'train': [], 'val': []}
        self.start_time = time.time()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.epochs_no_improve = 0
        self.logger = logging.getLogger('TrainingMonitor')

    def update_metrics(self, phase, metrics_dict):
        """
        Update metrics for a given phase ('train' or 'val').
        """
        if phase not in self.metrics:
            self.metrics[phase] = []
        self.metrics[phase].append(metrics_dict)
        if phase == 'val' and 'val_loss' in metrics_dict:
            val_loss = metrics_dict['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = len(self.metrics['val']) - 1
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

    def should_stop(self):
        """
        Check if early stopping should be triggered.
        """
        return self.epochs_no_improve >= self.early_stopping_patience

    def display_progress(self, epoch):
        """
        Display training progress.
        """
        train_metrics = self.metrics['train'][-1] if self.metrics['train'] else {}
        val_metrics = self.metrics['val'][-1] if self.metrics['val'] else {}
        msg = f"[Epoch {epoch+1}] Train: {train_metrics} | Val: {val_metrics}"
        print(msg)
        self.logger.info(msg)

    def get_metrics(self):
        """
        Get all recorded metrics.
        """
        return self.metrics

    def get_training_time(self):
        """
        Get elapsed training time in seconds.
        """
        return time.time() - self.start_time

    def get_best_epoch(self):
        """
        Get the epoch with the best validation loss.
        """
        return self.best_epoch


def push_metric_to_prometheus(metric_name, value, job_name='elvis_trading', gateway='localhost:9091', labels=None):
    """
    Push a metric to Prometheus pushgateway.
    
    Args:
        metric_name (str): Name of the metric
        value (float): Value of the metric
        job_name (str): Job name for Prometheus
        gateway (str): Pushgateway address
        labels (dict): Optional labels for the metric
    """
    if not HAS_PUSHGATEWAY:
        logging.getLogger(__name__).debug(f"Prometheus pushgateway not available, skipping metric {metric_name}={value}")
        return
        
    try:
        # Create gauge with labels if provided
        if labels:
            label_names = list(labels.keys())
            gauge = Gauge(metric_name, f'ELVIS Trading Bot metric: {metric_name}', labelnames=label_names)
            gauge.labels(**labels).set(value)
        else:
            gauge = Gauge(metric_name, f'ELVIS Trading Bot metric: {metric_name}')
            gauge.set(value)
        
        push_to_gateway(gateway, job=job_name, registry=gauge.registry)
        logging.getLogger(__name__).debug(f"Pushed metric {metric_name}={value} to {gateway}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to push metric to Prometheus: {e}")
