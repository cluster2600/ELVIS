"""
Main training module for the ELVIS trading system.
Orchestrates the training process using model trainer, monitoring, and checkpointing utilities.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import signal
import atexit
from typing import Dict, Optional, Union, Tuple

from trading.models.model_trainer import ModelTrainer
from trading.data.data_processor import DataProcessor
from trading.utils.monitoring import TrainingMonitor
from trading.utils.checkpoint import CheckpointManager
from trading.utils.logger import setup_logger

class TrainingInterrupt(Exception):
    """Custom exception for training interruption."""
    pass

def signal_handler(signum, frame):
    """Handle training interruption signals."""
    raise TrainingInterrupt("Training interrupted by user")

class TrainingManager:
    """Manages the training process for the ELVIS trading system."""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str):
        """
        Initialize the training manager.
        
        Args:
            config_path: Path to configuration file
            data_path: Path to training data
            output_dir: Directory for output files
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / 'models'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        
        for directory in [self.model_dir, self.log_dir, self.checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Update config with paths
        self.config['model_dir'] = str(self.model_dir)
        self.config['log_dir'] = str(self.log_dir)
        self.config['checkpoint_dir'] = str(self.checkpoint_dir)
        
        # Setup logging
        self.logger = setup_logger('training', str(self.log_dir / 'training.log'))
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.monitor = TrainingMonitor(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate training data."""
        try:
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
            
            # Validate data
            required_columns = self.config.get('required_columns', [])
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.logger.info(f"Loaded training data with shape: {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Prepare data for training."""
        try:
            # Process data
            X, y = self.model_trainer.prepare_data(self.data)
            
            # Create data loaders
            train_loader, val_loader = self.model_trainer.create_data_loaders(
                X, y, self.config['batch_size']
            )
            
            self.logger.info("Data prepared for training")
            return train_loader, val_loader
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def train(self, resume_checkpoint: Optional[str] = None):
        """Run the training process."""
        try:
            # Prepare data
            train_loader, val_loader = self.prepare_data()
            
            # Resume training if checkpoint provided
            start_epoch = 0
            if resume_checkpoint:
                checkpoint = self.checkpoint_manager.load_checkpoint(resume_checkpoint)
                if checkpoint:
                    start_epoch = checkpoint['epoch']
                    self.model_trainer.load_state_dict(checkpoint['model_state'])
                    self.logger.info(f"Resuming training from epoch {start_epoch}")
            
            # Training loop
            for epoch in range(start_epoch, self.config['transformer']['epochs']):
                self.logger.info(f"Starting epoch {epoch+1}/{self.config['transformer']['epochs']}")
                
                # Train step
                train_metrics = self.model_trainer.train_epoch(train_loader, epoch)
                self.monitor.update_metrics('train', train_metrics)
                
                # Validation step
                val_metrics = self.model_trainer.validate(val_loader)
                self.monitor.update_metrics('val', val_metrics)
                
                # Log metrics
                for metric, value in {**train_metrics, **val_metrics}.items():
                    self.writer.add_scalar(metric, value, epoch)
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('checkpoint_frequency', 5) == 0:
                    self.checkpoint_manager.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state': self.model_trainer.state_dict(),
                        'metrics': self.monitor.get_metrics()
                    })
                
                # Early stopping check
                if self.monitor.should_stop():
                    self.logger.info("Early stopping triggered")
                    break
                
                # Display progress
                self.monitor.display_progress(epoch)
                
        except TrainingInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            # Save final model and metrics
            self.checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state': self.model_trainer.state_dict(),
                'metrics': self.monitor.get_metrics()
            }, is_final=True)
            
            # Save training summary
            self._save_training_summary()
            
            # Cleanup
            self.cleanup()
            
    def train_rl_agents(self):
        """Train reinforcement learning agents."""
        try:
            self.logger.info("Starting RL agent training")
            rl_agents = self.model_trainer.train_rl_agents(
                self.config['rl']['env'], self.config['rl']['agent']
            )
            self.logger.info("RL agent training completed")
            return rl_agents
        except Exception as e:
            self.logger.error(f"Error training RL agents: {str(e)}")
            raise
            
    def evaluate_models(self, transformer_model, rl_agents):
        """Evaluate trained models."""
        try:
            self.logger.info("Evaluating models")
            
            # Prepare data for evaluation
            X, y = self.model_trainer.prepare_data(self.data)
            
            # Evaluate transformer model
            transformer_metrics = self.model_trainer.evaluate_model(transformer_model, X, y)
            self.logger.info("Transformer Model Metrics:")
            for metric, value in transformer_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
                
            # Evaluate RL agents
            rl_metrics = self.model_trainer.evaluate_model(rl_agents, X, y)
            self.logger.info("RL Agents Metrics:")
            for metric, value in rl_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
                
            return transformer_metrics, rl_metrics
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise
            
    def generate_explanations(self, transformer_model, rl_agents):
        """Generate model explanations."""
        try:
            self.logger.info("Generating model explanations")
            
            # Prepare data for explanations
            X, _ = self.model_trainer.prepare_data(self.data)
            feature_names = self.data.drop(columns=['target']).columns.tolist()
            
            # Generate explanations
            transformer_explanations = self.model_trainer.explain_model(
                transformer_model, X, feature_names
            )
            self.logger.info("Transformer model explanations generated")
            
            rl_explanations = self.model_trainer.explain_model(
                rl_agents, X, feature_names
            )
            self.logger.info("RL agents explanations generated")
            
            # Save explanations
            explanations_dir = self.log_dir / 'explanations'
            explanations_dir.mkdir(exist_ok=True)
            
            with open(explanations_dir / 'transformer_explanations.json', 'w') as f:
                json.dump(transformer_explanations, f, indent=2)
            
            with open(explanations_dir / 'rl_explanations.json', 'w') as f:
                json.dump(rl_explanations, f, indent=2)
                
            return transformer_explanations, rl_explanations
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            raise
            
    def _save_training_summary(self):
        """Save training summary to JSON file."""
        summary = {
            'config': self.config,
            'metrics': self.monitor.get_metrics(),
            'best_metrics': self.monitor.get_best_metrics(),
            'best_epoch': self.monitor.get_best_epoch(),
            'training_time': self.monitor.get_training_time(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Close TensorBoard writer
            self.writer.close()
            
            # Cleanup old checkpoints
            self.checkpoint_manager.cleanup_old_checkpoints(
                keep_last_n=self.config.get('keep_checkpoints', 5)
            )
            
            # Create checkpoint backup
            backup_dir = self.checkpoint_dir / 'backups' / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_manager.backup_checkpoints(str(backup_dir))
            
            self.logger.info("Training cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

def main():
    """Main entry point for training."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train ELVIS trading models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    # Initialize training manager
    trainer = TrainingManager(args.config, args.data, args.output)
    
    try:
        # Train transformer model
        trainer.train(args.resume)
        
        # Train RL agents
        rl_agents = trainer.train_rl_agents()
        
        # Evaluate models
        transformer_metrics, rl_metrics = trainer.evaluate_models(
            trainer.model_trainer.model, rl_agents
        )
        
        # Generate explanations
        trainer.generate_explanations(
            trainer.model_trainer.model, rl_agents
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 