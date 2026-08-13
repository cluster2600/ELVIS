#!/usr/bin/env python3.14
"""
Training script for the Reinforcement Learning trading model
This script trains a DQN agent on historical trading data to learn optimal trading patterns
"""

# Make the repo root importable no matter where this script is run from
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import sys
from datetime import datetime

from analyze_trades import analyze_recent_trades, get_all_trades_for_rl

from core.models.trading_rl_model import TradingRLModel
from utils.paper_trade_db import get_all_trades, get_conn

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train RL model on trading data")
    parser.add_argument(
        "--trades", type=int, default=500, help="Number of trades to train on"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trading_rl_model.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Force retrain even if model exists"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze recent trades first"
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    try:
        # Analyze recent trades if requested
        if args.analyze:
            logger.info("Analyzing recent trades...")
            analyze_recent_trades()
            print("\n" + "=" * 60 + "\n")

        # Check if we can connect to database
        conn = get_conn()
        if not conn:
            logger.error(
                "Cannot connect to database. Please check your database configuration."
            )
            sys.exit(1)
        conn.close()

        # Initialize RL model
        logger.info("Initializing RL model...")
        rl_model = TradingRLModel(logger, args.model_path)

        # Check if model exists and we're not forcing retrain
        if os.path.exists(args.model_path) and not args.retrain:
            logger.info(f"Model already exists at {args.model_path}")
            if input("Do you want to retrain? (y/n): ").lower() != "y":
                logger.info("Loading existing model...")
                rl_model.load_model()

                # Show training stats
                stats = rl_model.get_training_stats()
                logger.info(f"Existing model stats: {stats}")
                return

        # Train the model
        logger.info(f"Training RL model on {args.trades} trades...")
        success = rl_model.train_on_historical_data(limit=args.trades)

        if success:
            logger.info("Training completed successfully!")

            # Show training statistics
            stats = rl_model.get_training_stats()
            logger.info("Training Statistics:")
            logger.info(f"  Total episodes: {stats.get('total_episodes', 0)}")
            logger.info(f"  Average reward: {stats.get('avg_reward', 0):.3f}")
            logger.info(f"  Latest reward: {stats.get('latest_reward', 0):.3f}")
            logger.info(f"  Best reward: {stats.get('best_reward', 0):.3f}")
            logger.info(f"  Worst reward: {stats.get('worst_reward', 0):.3f}")
            logger.info(f"  Exploration rate: {stats.get('epsilon', 0):.3f}")
            logger.info(f"  Memory size: {stats.get('memory_size', 0)}")

            # Test the model on recent data
            logger.info("\nTesting model on recent market data...")
            test_model_predictions(rl_model)

        else:
            logger.error("Training failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)


def test_model_predictions(rl_model):
    """Test the model on recent market data"""
    try:
        # Get recent trades for testing
        recent_trades = get_all_trades_for_rl()

        if not recent_trades:
            logger.warning("No recent trades found for testing")
            return

        # Test predictions on recent data
        logger.info("Testing model predictions...")

        # Get current BTC price from recent trades
        current_price = 107000.0  # Default
        if recent_trades:
            current_price = recent_trades[-1][4]  # Latest trade price

        test_cases = [
            {
                "price": current_price * 0.99,
                "volume": 1000.0,
                "rsi": 30.0,
                "macd": 0.5,
            },  # Oversold
            {
                "price": current_price * 1.01,
                "volume": 1200.0,
                "rsi": 70.0,
                "macd": -0.3,
            },  # Overbought
            {
                "price": current_price,
                "volume": 900.0,
                "rsi": 50.0,
                "macd": 0.1,
            },  # Neutral
            {
                "price": current_price * 1.02,
                "volume": 1500.0,
                "rsi": 45.0,
                "macd": 0.8,
            },  # Bullish
            {
                "price": current_price * 0.98,
                "volume": 800.0,
                "rsi": 35.0,
                "macd": -0.5,
            },  # Bearish
        ]

        for i, test_case in enumerate(test_cases):
            signal, confidence = rl_model.predict_action(test_case, recent_trades)
            logger.info(
                f"Test case {i+1}: {signal} (confidence: {confidence:.3f}) - {test_case}"
            )

    except Exception as e:
        logger.error(f"Error testing model: {e}")


if __name__ == "__main__":
    main()
