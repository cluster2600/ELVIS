#!/usr/bin/env python3
"""
Enhanced Training Script with PostgreSQL Trade History Integration
Trains ELVIS models using both market data and actual trade history from the database.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train ELVIS models with integrated trade history')
    parser.add_argument('--limit', type=int, default=5000, help='Maximum number of trades to include (default: 5000)')
    parser.add_argument('--prediction-horizon', type=int, default=5, help='Number of trades to look ahead for targets (default: 5)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--config', type=str, default='training/config/model_config.yaml', help='Path to model config')
    parser.add_argument('--output', type=str, default='models', help='Output directory for models')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    args = parser.parse_args()
    
    print("🚀 ELVIS Enhanced Training with Trade History Integration")
    print("=" * 60)
    print(f"📊 Trade limit: {args.limit}")
    print(f"🎯 Prediction horizon: {args.prediction_horizon} trades")
    print(f"⏱️  Training epochs: {args.epochs}")
    print(f"📁 Output directory: {args.output}")
    print("=" * 60)
    
    # Build command for main training script
    cmd_parts = [
        sys.executable,
        'training/train_models.py',
        '--include-trade-history',  # Enable trade history integration
        '--config', args.config,
        '--output', args.output,
    ]
    
    if args.debug:
        cmd_parts.append('--debug')
    
    if args.distributed:
        cmd_parts.append('--distributed')
    
    # Set environment variables for trade processing
    os.environ['TRADE_LIMIT'] = str(args.limit)
    os.environ['PREDICTION_HORIZON'] = str(args.prediction_horizon)
    
    print("🔄 Starting training with trade history integration...")
    print(f"📋 Command: {' '.join(cmd_parts)}")
    print("")
    
    # Execute training
    import subprocess
    try:
        result = subprocess.run(cmd_parts, check=True)
        print("\n✅ Training completed successfully!")
        print("📈 Models have been trained with integrated trade history")
        print(f"💾 Check {args.output}/ for trained models and logs")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()