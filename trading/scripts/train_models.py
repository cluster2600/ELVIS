#!/usr/bin/env python3
"""
Script to run the model training pipeline for the ELVIS trading system.
Implements distributed training, checkpointing, and advanced monitoring.
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
import atexit
from typing import Dict, Optional, Union

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from trading.models.model_trainer import ModelTrainer
from trading.data.data_processor import DataProcessor
from trading.utils.logger import setup_logger
from trading.utils.monitoring import TrainingMonitor
from trading.utils.checkpoint import CheckpointManager

class TrainingInterrupt(Exception):
    """Custom exception for training interruption."""
    pass

def signal_handler(signum, frame):
    """Handle training interruption signals."""
    raise TrainingInterrupt("Training interrupted by user")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--config', type=str, default='trading/config/model_config.yaml',
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

def setup_distributed(args):
    """Setup distributed training environment."""
    if args.distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        return True
    return False

def load_data(data_path: str, config: Dict) -> pd.DataFrame:
    """Load and validate training data."""
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Validate data
        required_columns = config.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def setup_training_environment(args, config: Dict) -> Dict:
    """Setup training environment and directories."""
    # Create output directories
    output_dir = Path(args.output)
    model_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'
    
    for directory in [model_dir, log_dir, checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Update config with paths
    config['model_dir'] = str(model_dir)
    config['log_dir'] = str(log_dir)
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging
    log_dir = Path(args.output) / 'logs'
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger('model_training', str(log_file))
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Setup distributed training if enabled
        is_distributed = setup_distributed(args)
        if is_distributed:
            logger.info(f"Distributed training enabled. Local rank: {args.local_rank}")
        
        # Setup training environment
        config = setup_training_environment(args, config)
        
        # Initialize components
        data_processor = DataProcessor(config)
        model_trainer = ModelTrainer(config)
        monitor = TrainingMonitor(config)
        checkpoint_manager = CheckpointManager(config)
        
        # Setup TensorBoard
        writer = SummaryWriter(log_dir=str(Path(config['log_dir']) / 'tensorboard'))
        
        # Load and prepare data
        data = load_data(args.data, config)
        logger.info(f"Loaded training data with shape: {data.shape}")
        
        # Prepare data
        X, y = model_trainer.prepare_data(data)
        logger.info("Data prepared for training")
        
        # Create data loaders
        train_loader, val_loader = model_trainer.create_data_loaders(X, y, config['batch_size'])
        
        # Resume training if checkpoint provided
        start_epoch = 0
        if args.resume:
            checkpoint = checkpoint_manager.load_checkpoint(args.resume)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                model_trainer.load_state_dict(checkpoint['model_state'])
                logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        try:
            for epoch in range(start_epoch, config['transformer']['epochs']):
                # Train transformer model
                logger.info(f"Starting epoch {epoch+1}/{config['transformer']['epochs']}")
                
                # Train step
                train_metrics = model_trainer.train_epoch(train_loader, epoch)
                monitor.update_metrics('train', train_metrics)
                
                # Validation step
                val_metrics = model_trainer.validate(val_loader)
                monitor.update_metrics('val', val_metrics)
                
                # Log metrics
                for metric, value in {**train_metrics, **val_metrics}.items():
                    writer.add_scalar(metric, value, epoch)
                
                # Save checkpoint
                if (epoch + 1) % config.get('checkpoint_frequency', 5) == 0:
                    checkpoint_manager.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state': model_trainer.state_dict(),
                        'metrics': monitor.get_metrics()
                    })
                
                # Early stopping check
                if monitor.should_stop():
                    logger.info("Early stopping triggered")
                    break
                
                # Progress bar
                if not is_distributed or args.local_rank == 0:
                    monitor.display_progress(epoch)
        
        except TrainingInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            # Save final model and metrics
            checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state': model_trainer.state_dict(),
                'metrics': monitor.get_metrics()
            }, is_final=True)
            
            # Save training summary
            summary = {
                'config': config,
                'metrics': monitor.get_metrics(),
                'training_time': monitor.get_training_time(),
                'best_epoch': monitor.get_best_epoch()
            }
            with open(Path(config['log_dir']) / 'training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            writer.close()
        
        # Train RL agents
        if not is_distributed or args.local_rank == 0:
            logger.info("Starting RL agent training")
            rl_agents = model_trainer.train_rl_agents(
                config['rl']['env'], config['rl']['agent']
            )
            logger.info("RL agent training completed")
            
            # Evaluate models
            logger.info("Evaluating models")
            transformer_metrics = model_trainer.evaluate_model(model_trainer.model, X, y)
            rl_metrics = model_trainer.evaluate_model(rl_agents, X, y)
            
            # Log evaluation results
            logger.info("Transformer Model Metrics:")
            for metric, value in transformer_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info("RL Agents Metrics:")
            for metric, value in rl_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Generate model explanations
            logger.info("Generating model explanations")
            feature_names = data.drop(columns=['target']).columns.tolist()
            
            transformer_explanations = model_trainer.explain_model(
                model_trainer.model, X, feature_names
            )
            logger.info("Transformer model explanations generated")
            
            rl_explanations = model_trainer.explain_model(
                rl_agents, X, feature_names
            )
            logger.info("RL agents explanations generated")
            
            # Save explanations
            explanations_dir = Path(config['log_dir']) / 'explanations'
            explanations_dir.mkdir(exist_ok=True)
            
            with open(explanations_dir / 'transformer_explanations.json', 'w') as f:
                json.dump(transformer_explanations, f, indent=2)
            
            with open(explanations_dir / 'rl_explanations.json', 'w') as f:
                json.dump(rl_explanations, f, indent=2)
        
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise
    finally:
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 