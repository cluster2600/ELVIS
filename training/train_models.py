#!/usr/bin/env python3
"""
Script to run the model training pipeline for the ELVIS trading system.
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import signal
from typing import Dict, Optional, Union

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

print("DEBUG: sys.path =", sys.path)
print("DEBUG: Current working directory =", os.getcwd())

# Add the training/data directory to the Python path to fix import errors
training_data_dir = str(Path(__file__).parent / 'data')
sys.path.append(training_data_dir)

# Add the training/utils directory to the Python path to fix import errors
training_utils_dir = str(Path(__file__).parent / 'utils')
sys.path.append(training_utils_dir)

# ✅ CORRECTED imports
from training.models.model_trainer import ModelTrainer
from trading.data.data_processor import DataProcessor
from utils.logging_utils import setup_logger
from utils.monitoring import TrainingMonitor
from trading.utils.checkpoint import CheckpointManager

class TrainingInterrupt(Exception):
    pass

def signal_handler(signum, frame):
    raise TrainingInterrupt("Training interrupted by user")

def parse_args():
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--config', type=str, default='training/config/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/processed/training_data.csv',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
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
        raise KeyboardInterrupt("Training interrupted by user")

    def _setup_logging(self):
        log_dir = Path(self.args.output) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.logger = setup_logger('model_training', str(log_file))
        self.logger.info("Logging initialized")

    def _load_config(self):
        self.config = yaml.safe_load(open(self.args.config, 'r'))
        self.logger.info("Configuration loaded successfully")

    def _setup_distributed(self):
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.args.local_rank)
            self.is_distributed = True
            self.logger.info(f"Distributed training enabled. Local rank: {self.args.local_rank}")

    def _setup_training_environment(self):
        output_dir = Path(self.args.output)
        model_dir = output_dir / 'models'
        log_dir = output_dir / 'logs'
        checkpoint_dir = output_dir / 'checkpoints'
        for directory in [model_dir, log_dir, checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.config['model_dir'] = str(model_dir)
        self.config['log_dir'] = str(log_dir)
        self.config['checkpoint_dir'] = str(checkpoint_dir)

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
        if self.args.data.endswith('.csv'):
            self.data = pd.read_csv(self.args.data)
        elif self.args.data.endswith('.parquet'):
            self.data = pd.read_parquet(self.args.data)
        else:
            raise ValueError(f"Unsupported data format: {self.args.data}")
        self.logger.info(f"Loaded training data with shape: {self.data.shape}")
        self.X, self.y = self.model_trainer.prepare_data(self.data)
        self.logger.info("Data prepared for training")

    def _create_data_loaders(self):
        self.train_loader, self.val_loader = self.model_trainer.create_data_loaders(
            self.X, self.y, self.config['batch_size'])

    def _resume_training_if_needed(self):
        self.start_epoch = 0
        if self.args.resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(self.args.resume)
            if checkpoint:
                self.start_epoch = checkpoint['epoch']
                self.model_trainer.load_state_dict(checkpoint['model_state'])
                self.logger.info(f"Resuming training from epoch {self.start_epoch}")

    def train(self):
        try:
            for epoch in range(self.start_epoch, self.config['transformer']['epochs']):
                self.logger.info(f"Starting epoch {epoch+1}/{self.config['transformer']['epochs']}")
                train_metrics = self.model_trainer.train_epoch(self.train_loader, epoch)
                self.monitor.update_metrics('train', train_metrics)
                val_metrics = self.model_trainer.validate(self.val_loader)
                self.monitor.update_metrics('val', val_metrics)
                for metric, value in {**train_metrics, **val_metrics}.items():
                    self.writer.add_scalar(metric, value, epoch)
                if (epoch + 1) % self.config.get('checkpoint_frequency', 5) == 0:
                    self.checkpoint_manager.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state': self.model_trainer.state_dict(),
                        'metrics': self.monitor.get_metrics()
                    })
                if self.monitor.should_stop():
                    self.logger.info("Early stopping triggered")
                    break
                if not self.is_distributed or self.args.local_rank == 0:
                    self.monitor.display_progress(epoch)
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            self.checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state': self.model_trainer.state_dict(),
                'metrics': self.monitor.get_metrics()
            }, is_final=True)
            summary = {
                'config': self.config,
                'metrics': self.monitor.get_metrics(),
                'training_time': self.monitor.get_training_time(),
                'best_epoch': self.monitor.get_best_epoch()
            }
            with open(Path(self.config['log_dir']) / 'training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            self.writer.close()

    def train_rl_agents(self):
        self.logger.error("train_rl_agents method is not implemented in ModelTrainer")
        raise NotImplementedError("train_rl_agents method is not implemented in ModelTrainer")

    def evaluate_models(self, rl_agents):
        try:
            transformer_metrics = self.model_trainer.evaluate_model(self.model_trainer.model, self.X, self.y)
            rl_metrics = self.model_trainer.evaluate_model(rl_agents, self.X, self.y)
            self.logger.info("Transformer Model Metrics:")
            for metric, value in transformer_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            self.logger.info("RL Agents Metrics:")
            for metric, value in rl_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise

    def generate_explanations(self, rl_agents):
        try:
            feature_names = self.data.drop(columns=['target']).columns.tolist()
            transformer_explanations = self.model_trainer.explain_model(self.model_trainer.model, self.X, feature_names)
            self.logger.info("Transformer model explanations generated")
            rl_explanations = self.model_trainer.explain_model(rl_agents, self.X, feature_names)
            self.logger.info("RL agents explanations generated")
            explanations_dir = Path(self.config['log_dir']) / 'explanations'
            explanations_dir.mkdir(exist_ok=True)
            with open(explanations_dir / 'transformer_explanations.json', 'w') as f:
                json.dump(transformer_explanations, f, indent=2)
            with open(explanations_dir / 'rl_explanations.json', 'w') as f:
                json.dump(rl_explanations, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error generating model explanations: {str(e)}")
            raise

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
