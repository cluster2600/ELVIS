#!/usr/bin/env python3
"""
Script to run the model training pipeline for the ELVIS trading system.
Optimized version: Modularized code, removed redundancies, and fixed syntax errors.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
import json
import signal
from typing import Dict, Optional, Union
from types import SimpleNamespace

# Handle TensorBoard import with NumPy compatibility
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    print(f"Warning: TensorBoard import failed: {e}")
    # Mock SummaryWriter for compatibility
    class MockSummaryWriter:
        def __init__(self, log_dir=None):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
    SummaryWriter = MockSummaryWriter

# Add necessary paths
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
sys.path.append(str(Path(__file__).parent / 'data'))
sys.path.append(str(Path(__file__).parent / 'utils'))

from training.models.model_trainer import ModelTrainer
from trading.data.data_processor import DataProcessor
from utils.logging_utils import setup_logger
from utils.monitoring import TrainingMonitor
from trading.utils.checkpoint import CheckpointManager
from training.models.evaluator import Evaluator  # Import for reference
from training.data.trade_history_processor import TradeHistoryProcessor  # Import trade history processor

class TrainingInterrupt(Exception):
    pass

def signal_handler(signum, frame):
    raise TrainingInterrupt("Training interrupted by user")

def parse_args():
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--config', type=str, default='training/config/model_config.yaml', help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/processed/training_data.csv', help='Path to training data')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--include-trade-history', action='store_true', help='Include trade history from database')
    return parser.parse_args()

class TrainingPipeline:
    def __init__(self, args):
        self.args = args
        self.config = None
        self.logger = None
        self.is_distributed = False
        self.data_processor = None
        self.model_trainer = None
        self.monitor = None
        self.checkpoint_manager = None
        self.writer = None
        self.data = None
        self.X = None
        self.y = None
        self.train_loader = None
        self.val_loader = None
        self.start_epoch = 0

    def setup(self):
        self._setup_signal_handlers()
        self._setup_logging()
        self._load_config()
        self._setup_distributed()
        self._setup_training_environment()
        self._initialize_components()
        self._load_and_prepare_data()
        self._create_data_loaders()
        self._resume_training_if_needed()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        raise KeyboardInterrupt("Training interrupted")

    def _setup_logging(self):
        log_dir = Path(self.args.output) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.logger = setup_logger('model_training', str(log_file))
        self.logger.info("Logging initialized")

    def _load_config(self):
        self.config = yaml.safe_load(open(self.args.config, 'r'))
        self.logger.info("Configuration loaded")

    def _setup_distributed(self):
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.args.local_rank)
            self.is_distributed = True
            self.logger.info(f"Distributed training enabled. Local rank: {self.args.local_rank}")

    def _setup_training_environment(self):
        output_dir = Path(self.args.output)
        for dir_name in ['models', 'logs', 'checkpoints']:
            (output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        self.config.update({
            'model_dir': str(output_dir / 'models'),
            'log_dir': str(output_dir / 'logs'),
            'checkpoint_dir': str(output_dir / 'checkpoints')
        })

    def _initialize_components(self):
        self.data_processor = DataProcessor(
            exchange=None,
            feature_config=self.config.get('feature_config', {}),
            quality_config=self.config.get('quality_config', {}),
            logger=self.logger
        )
        self.model_trainer = ModelTrainer(self.config)
        self.monitor = TrainingMonitor(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.writer = SummaryWriter(log_dir=str(Path(self.config['log_dir']) / 'tensorboard'))

    def _load_and_prepare_data(self):
        # Load original market data
        self.data = pd.read_csv(self.args.data) if self.args.data.endswith('.csv') else pd.read_parquet(self.args.data)
        self.logger.info(f"Loaded market data with shape: {self.data.shape}")
        
        # Load and integrate trade history if requested
        if getattr(self.args, 'include_trade_history', False):
            self.logger.info("📊 Integrating trade history from database...")
            try:
                trade_processor = TradeHistoryProcessor(self.logger)
                trade_features, trade_targets = trade_processor.process_for_training(limit=5000)
                
                if not trade_features.empty:
                    self.logger.info(f"✅ Loaded {len(trade_features)} trade history samples")
                    self.logger.info(f"📈 Trade features: {len(trade_features.columns)} columns")
                    self.logger.info(f"🎯 Trade targets: {len(trade_targets.columns)} columns")
                    
                    # Save trade data for reference
                    trade_processor.save_processed_data(trade_features, trade_targets, 'data/processed/trade_history')
                    
                    # Add trade history features to config for model training
                    self.config['trade_history_features'] = list(trade_features.columns)
                    self.config['trade_history_targets'] = list(trade_targets.columns)
                    self.config['trade_history_samples'] = len(trade_features)
                    
                    # Store trade data for model training
                    self.trade_features = trade_features
                    self.trade_targets = trade_targets
                    
                    self.logger.info("✅ Trade history integration completed successfully")
                else:
                    self.logger.warning("⚠️ No trade history data available")
                    self.trade_features = pd.DataFrame()
                    self.trade_targets = pd.DataFrame()
                    
            except Exception as e:
                self.logger.error(f"❌ Error integrating trade history: {e}")
                self.trade_features = pd.DataFrame()
                self.trade_targets = pd.DataFrame()
        else:
            self.logger.info("📊 Skipping trade history integration (not requested)")
            self.trade_features = pd.DataFrame()
            self.trade_targets = pd.DataFrame()
        
        # Prepare data for model training
        self.X, self.y = self.model_trainer.prepare_data(self.data)
        
        # Validate shapes of X and y
        if self.X.size == 0 or self.y.size == 0:
            raise ValueError("Feature matrix X or target vector y is empty after data preparation.")
        self.logger.info(f"Feature matrix X shape: {self.X.shape}, Target vector y shape: {self.y.shape}")
        
        # Log trade history integration status
        if not self.trade_features.empty:
            self.logger.info(f"🔗 Trade history features available: {self.trade_features.shape}")
            self.logger.info(f"🔗 Trade history targets available: {self.trade_targets.shape}")
        else:
            self.logger.info("🔗 No trade history features integrated")

    def _create_data_loaders(self):
        self.train_loader, self.val_loader = self.model_trainer.create_data_loaders(self.X, self.y, self.config['batch_size'])

    def _resume_training_if_needed(self):
        if self.args.resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(self.args.resume)
            if checkpoint:
                self.start_epoch = checkpoint['epoch']
                self.model_trainer.load_state_dict(checkpoint['model_state'])
                self.logger.info(f"Resuming from epoch {self.start_epoch}")

    def train(self):
        for epoch in range(self.start_epoch, int(self.config['transformer'].get('epochs', 0))):
            self.logger.info(f"Epoch {epoch+1}/{self.config['transformer']['epochs']}")
            train_metrics = self.model_trainer.train_epoch(self.train_loader, epoch)
            self.monitor.update_metrics('train', train_metrics)
            val_metrics = self.model_trainer.validate(self.val_loader)
            self.monitor.update_metrics('val', val_metrics)
            for metric, value in {**train_metrics, **val_metrics}.items():
                self.writer.add_scalar(metric, value, epoch)
            if (epoch + 1) % self.config.get('checkpoint_frequency', 5) == 0:
                self.checkpoint_manager.save_checkpoint({'epoch': epoch + 1, 'model_state': self.model_trainer.state_dict(), 'metrics': self.monitor.get_metrics()})
            if self.monitor.should_stop():
                break
            self.monitor.display_progress(epoch)

    def train_rl_agents(self):
        from training.models.rl_agents import MultiAgentTradingSystem
        rl_config = self.config.get('rl', {})
        # Pass only expected arguments to avoid TypeError
        expected_args = {
            'total_timesteps': rl_config.get('total_timesteps', 10000),
            'eval_freq': rl_config.get('eval_freq', 1000),
            'n_eval_episodes': rl_config.get('n_eval_episodes', 10)
        }
        rl_agents = MultiAgentTradingSystem(rl_config.get('env', {}), rl_config.get('agent', {}).get('n_agents', 1))
        rl_agents.train(**expected_args)  # Use only expected args
        return rl_agents

    def evaluate_models(self, rl_agents):
        from training.models.evaluator import Evaluator
        # Fix: Pass all required arguments to Evaluator
        args_namespace = SimpleNamespace(env=None, eval_gap=500, eval_times=10, target_return=200, cwd=self.args.output)
        evaluator = Evaluator(cwd=self.args.output, agent_id=0, eval_env=None, args=args_namespace)  # Using placeholder for agent_id and eval_env
        
        try:
            ensemble_models = self.model_trainer.load_ensemble()
            self.logger.info(f"Loaded {len(ensemble_models)} ensemble models for evaluation")
            
            transformer_metrics = {}
            for name, model in ensemble_models.items():
                try:
                    metrics = evaluator.evaluate(model, self.X, self.y)
                    transformer_metrics[name] = metrics
                    self.logger.info(f"Evaluated model {name}: {metrics}")
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate model {name}: {e}")
                    transformer_metrics[name] = {"error": str(e)}
                    
        except Exception as e:
            self.logger.warning(f"Failed to load ensemble models: {e}")
            transformer_metrics = {"error": "Failed to load ensemble models", "details": str(e)}
        
        # Evaluate RL agents
        try:
            rl_metrics = rl_agents.evaluate(self.X, self.y) if hasattr(rl_agents, 'evaluate') else {"status": "no_evaluate_method"}
        except Exception as e:
            self.logger.warning(f"Failed to evaluate RL agents: {e}")
            rl_metrics = {"error": "RL evaluation failed", "details": str(e)}
        
        # Save results with fallback content
        if not transformer_metrics:
            transformer_metrics = {"status": "no_models_evaluated", "timestamp": str(datetime.now())}
        if not rl_metrics:
            rl_metrics = {"status": "no_rl_evaluation", "timestamp": str(datetime.now())}
            
        evaluator.save_results(transformer_metrics, 'transformer_evaluation.json')
        evaluator.save_results(rl_metrics, 'rl_agents_evaluation.json')
        return transformer_metrics, rl_metrics

    def generate_explanations(self, rl_agents):
        import numpy as np

        target_column = self.config.get('target', 'price')
        if target_column in self.data.columns:
            feature_names = self.data.drop(columns=[target_column]).columns.tolist()
        else:
            feature_names = self.data.columns.tolist()

        transformer_explanations = self.model_trainer.explain_model(self.model_trainer.model, self.X, feature_names)

        # Convert numpy arrays in explanations to lists for JSON serialization
        def convert_np(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(i) for i in obj]
            return obj

        transformer_explanations = convert_np(transformer_explanations)

        # Skip RL agent explanations as they are not supported
        rl_explanations = {}

        explanations_dir = Path(self.config['log_dir']) / 'explanations'
        explanations_dir.mkdir(exist_ok=True)
        with open(explanations_dir / 'transformer_explanations.json', 'w') as f:
            json.dump(transformer_explanations, f, indent=2)
        with open(explanations_dir / 'rl_explanations.json', 'w') as f:
            json.dump(rl_explanations, f, indent=2)

def main():
    args = parse_args()
    pipeline = TrainingPipeline(args)
    pipeline.setup()
    pipeline.train()
    rl_agents = pipeline.train_rl_agents()
    pipeline.evaluate_models(rl_agents)
    pipeline.generate_explanations(rl_agents)

if __name__ == "__main__":
    main()
