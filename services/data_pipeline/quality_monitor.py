"""
Data Quality Monitor for tracking and alerting on data quality issues.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataQualityMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.alerts = []
        self.quality_thresholds = {
            'latency_max': config.get('latency_max_seconds', 1.0),
            'throughput_min': config.get('throughput_min_messages', 10),
            'error_rate_max': config.get('error_rate_max_percent', 1.0)
        }

    def update_metrics(self, metrics: Dict):
        """Update quality metrics and check for issues."""
        timestamp = datetime.now()
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Clean old metrics
        self._clean_old_metrics()
        
        # Check for quality issues
        self._check_quality_issues(metrics, timestamp)

    def _clean_old_metrics(self):
        """Remove metrics older than the retention period."""
        retention_period = self.config.get('metrics_retention_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=retention_period)
        
        self.metrics_history = [
            entry for entry in self.metrics_history
            if entry['timestamp'] > cutoff_time
        ]

    def _check_quality_issues(self, metrics: Dict, timestamp: datetime):
        """Check for data quality issues and generate alerts."""
        issues = []
        
        # Check latency
        if metrics['latency_avg'] > self.quality_thresholds['latency_max']:
            issues.append(f"High latency detected: {metrics['latency_avg']:.2f}s")
            
        # Check throughput
        if metrics['throughput'] < self.quality_thresholds['throughput_min']:
            issues.append(f"Low throughput detected: {metrics['throughput']} messages")
            
        # Check error rate
        error_rate = (metrics['error_rate'] / metrics['throughput']) * 100 if metrics['throughput'] > 0 else 0
        if error_rate > self.quality_thresholds['error_rate_max']:
            issues.append(f"High error rate detected: {error_rate:.2f}%")
            
        if issues:
            alert = {
                'timestamp': timestamp,
                'issues': issues,
                'metrics': metrics
            }
            self.alerts.append(alert)
            self.logger.warning(f"Data quality issues detected: {', '.join(issues)}")

    def get_quality_report(self) -> Dict:
        """Generate a quality report with statistics and alerts."""
        if not self.metrics_history:
            return {
                'status': 'no_data',
                'message': 'No metrics available'
            }
            
        # Calculate statistics
        metrics_df = pd.DataFrame([
            entry['metrics'] for entry in self.metrics_history
        ])
        
        return {
            'status': 'ok',
            'statistics': {
                'latency_avg': metrics_df['latency_avg'].mean(),
                'latency_max': metrics_df['latency_avg'].max(),
                'throughput_avg': metrics_df['throughput'].mean(),
                'error_rate_avg': metrics_df['error_rate'].mean() / metrics_df['throughput'].mean() * 100
            },
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'total_alerts': len(self.alerts)
        }

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_time
        ]

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = [] 