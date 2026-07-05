"""
Reinforcement Learning Trading Strategy
Uses a DQN agent trained on historical trading data to make trading decisions
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.models.trading_rl_model import TradingRLModel
from trading.strategies.base_strategy import BaseStrategy
from utils.paper_trade_db import get_all_trades


class RLStrategy(BaseStrategy):
    """
    Reinforcement Learning Strategy that uses a DQN agent to make trading decisions
    based on learned patterns from historical trading data.
    """

    def __init__(
        self,
        logger: logging.Logger,
        symbols: List[str] = ["BTCUSDT"],
        model_path: str = "models/trading_rl_model.pth",
        retrain_interval: int = 100,  # Retrain every 100 trades
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize the RL trading strategy

        Args:
            logger: Logger instance
            symbols: List of trading symbols
            model_path: Path to save/load RL model
            retrain_interval: How often to retrain the model
            confidence_threshold: Minimum confidence for trade signals
        """
        super().__init__(logger)
        self.symbols = symbols
        self.model_path = model_path
        self.retrain_interval = retrain_interval
        self.confidence_threshold = confidence_threshold

        # Initialize RL model
        self.rl_model = TradingRLModel(logger, model_path)

        # Training state
        self.trades_since_retrain = 0
        self.last_trade_result = None

        # Performance tracking
        self.predictions = []
        self.actual_results = []

        # Initial training
        self._initial_training()

    def _initial_training(self):
        """Perform initial training on historical data"""
        try:
            self.logger.info("Starting initial RL training...")

            # Train on historical data
            success = self.rl_model.train_on_historical_data(limit=500)

            if success:
                self.logger.info("✅ Initial RL training completed successfully")
                stats = self.rl_model.get_training_stats()
                self.logger.info(f"📊 Training stats: {stats}")
            else:
                self.logger.warning(
                    "⚠️ Initial RL training failed, model may not perform optimally"
                )

        except Exception as e:
            self.logger.error(f"Error in initial training: {e}")

    def generate_signals(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals using the RL model

        Args:
            data: Dictionary of market data for each symbol

        Returns:
            Dictionary of signals for each symbol
        """
        signals = {}

        for symbol in self.symbols:
            try:
                if symbol not in data or data[symbol].empty:
                    signals[symbol] = {"signal": "HOLD", "confidence": 0.0}
                    continue

                # Get current market data
                df = data[symbol]
                market_data = self._extract_market_data(df)

                # Get RL prediction
                signal, confidence = self.rl_model.predict_action(market_data)

                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    self.logger.info(
                        f"RL confidence {confidence:.3f} below threshold {self.confidence_threshold}, using HOLD"
                    )
                    signal = "HOLD"
                    confidence = 0.5

                signals[symbol] = {
                    "signal": signal,
                    "confidence": confidence,
                    "source": "RL_DQN",
                }

                self.logger.info(
                    f"RL signal for {symbol}: {signal} (confidence: {confidence:.3f})"
                )

            except Exception as e:
                self.logger.error(f"Error generating RL signal for {symbol}: {e}")
                signals[symbol] = {"signal": "HOLD", "confidence": 0.0}

        return signals

    def generate_signal(self, symbol: str, market_data: dict) -> Tuple[str, float]:
        """
        Generate a single trading signal for the main trading loop

        Args:
            symbol: Trading symbol
            market_data: Current market data

        Returns:
            (signal, confidence) tuple
        """
        try:
            self.logger.info(f"Generating RL signal for {symbol}")

            # Get RL prediction
            signal, confidence = self.rl_model.predict_action(market_data)

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                self.logger.info(
                    f"RL confidence {confidence:.3f} below threshold, using HOLD"
                )
                signal = "HOLD"
                confidence = 0.5

            # Log prediction for tracking
            self.predictions.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "price": market_data.get("price", 0.0),
                }
            )

            # Check if we need to retrain
            self._check_retrain()

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error generating RL signal: {e}")
            return "HOLD", 0.5

    def update_with_trade_result(self, trade_result: Dict):
        """
        Update the RL model with actual trade results for online learning

        Args:
            trade_result: Dictionary containing trade results
        """
        try:
            self.logger.info("Updating RL model with trade result")

            # Update model with trade result
            self.rl_model.update_with_trade_result(trade_result)

            # Track performance
            self.actual_results.append(trade_result)
            self.trades_since_retrain += 1

            # Save updated model periodically
            if self.trades_since_retrain % 10 == 0:
                self.rl_model.save_model()

        except Exception as e:
            self.logger.error(f"Error updating RL model with trade result: {e}")

    def _check_retrain(self):
        """Check if model needs retraining"""
        try:
            if self.trades_since_retrain >= self.retrain_interval:
                self.logger.info("Retraining RL model with recent data...")

                # Retrain on recent data
                success = self.rl_model.train_on_historical_data(limit=200)

                if success:
                    self.logger.info("RL model retrained successfully")
                    stats = self.rl_model.get_training_stats()
                    self.logger.info(f"Updated training stats: {stats}")

                self.trades_since_retrain = 0

        except Exception as e:
            self.logger.error(f"Error retraining RL model: {e}")

    def _extract_market_data(self, df: pd.DataFrame) -> Dict:
        """Extract market data from DataFrame"""
        try:
            latest = df.iloc[-1]

            market_data = {
                "price": latest.get("close", 0.0),
                "volume": latest.get("volume", 0.0),
                "high": latest.get("high", 0.0),
                "low": latest.get("low", 0.0),
                "rsi": latest.get("rsi", 50.0),
                "macd": latest.get("macd", 0.0),
                "macd_signal": latest.get("macd_signal", 0.0),
                "bb_upper": latest.get("bb_upper", 0.0),
                "bb_lower": latest.get("bb_lower", 0.0),
                "bb_middle": latest.get("bb_middle", 0.0),
                "atr": latest.get("atr", 0.0),
                "adx": latest.get("adx", 0.0),
                "sma_20": latest.get("sma_20", 0.0),
                "sma_50": latest.get("sma_50", 0.0),
            }

            return market_data

        except Exception as e:
            self.logger.error(f"Error extracting market data: {e}")
            return {"price": 0.0, "volume": 0.0}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the RL strategy"""
        try:
            if not self.predictions or not self.actual_results:
                return {}

            # Calculate accuracy
            correct_predictions = 0
            total_predictions = min(len(self.predictions), len(self.actual_results))

            for i in range(total_predictions):
                pred = self.predictions[i]
                result = self.actual_results[i]

                # Check if prediction was correct
                if pred["signal"] == "BUY" and result.get("pnl", 0) > 0:
                    correct_predictions += 1
                elif pred["signal"] == "SELL" and result.get("pnl", 0) > 0:
                    correct_predictions += 1
                elif pred["signal"] == "HOLD":
                    correct_predictions += 1  # HOLD is always "correct"

            accuracy = (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            )

            # Get RL training stats
            training_stats = self.rl_model.get_training_stats()

            return {
                "accuracy": accuracy,
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "avg_confidence": sum(p["confidence"] for p in self.predictions)
                / len(self.predictions),
                "training_stats": training_stats,
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def reset_performance_tracking(self):
        """Reset performance tracking metrics"""
        self.predictions = []
        self.actual_results = []
        self.trades_since_retrain = 0

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the RL model"""
        try:
            training_stats = self.rl_model.get_training_stats()

            return {
                "model_type": "DQN",
                "state_size": (
                    self.rl_model.feature_extractor.state_size
                    if hasattr(self.rl_model.feature_extractor, "state_size")
                    else 16
                ),
                "action_size": 3,
                "confidence_threshold": self.confidence_threshold,
                "retrain_interval": self.retrain_interval,
                "trades_since_retrain": self.trades_since_retrain,
                "training_stats": training_stats,
            }

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}

    def force_retrain(self):
        """Force retraining of the model"""
        try:
            self.logger.info("Forcing RL model retraining...")
            success = self.rl_model.train_on_historical_data(limit=500)

            if success:
                self.logger.info("Forced retraining completed successfully")
                self.trades_since_retrain = 0
                return True
            else:
                self.logger.error("Forced retraining failed")
                return False

        except Exception as e:
            self.logger.error(f"Error in forced retraining: {e}")
            return False

    def save_model(self):
        """Save the RL model"""
        try:
            self.rl_model.save_model()
            self.logger.info("RL model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving RL model: {e}")

    def load_model(self):
        """Load the RL model"""
        try:
            success = self.rl_model.load_model()
            if success:
                self.logger.info("RL model loaded successfully")
            else:
                self.logger.warning("Failed to load RL model")
            return success
        except Exception as e:
            self.logger.error(f"Error loading RL model: {e}")
            return False

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        current_price: float,
        available_capital: float,
        leverage: float = 1.0,
        signal_confidence: float = 0.5,
    ) -> float:
        """
        Calculate position size based on RL confidence and risk management

        Args:
            data: Market data DataFrame
            current_price: Current market price
            available_capital: Available capital
            leverage: Leverage to use
            signal_confidence: Confidence from RL model

        Returns:
            Position size in BTC
        """
        try:
            if available_capital <= 0 or current_price <= 0:
                return 0.001  # Minimum position size

            # Base risk percentage scaled by RL confidence
            base_risk = 0.01  # 1% base risk
            confidence_multiplier = max(0.5, signal_confidence)  # Scale by confidence

            risk_amount = available_capital * base_risk * confidence_multiplier
            position_size = (risk_amount * leverage) / current_price

            # Apply bounds
            min_size = 0.001
            max_size = min(0.1, available_capital * 0.1 / current_price)

            return max(min_size, min(position_size, max_size))

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.001

    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate stop loss based on market volatility

        Args:
            data: Market data DataFrame
            entry_price: Entry price

        Returns:
            Stop loss price
        """
        try:
            if len(data) < 14:
                return entry_price * 0.98  # 2% stop loss

            # Calculate ATR for volatility-based stop loss
            high_low = data["high"] - data["low"]
            high_close = abs(data["high"] - data["close"].shift())
            low_close = abs(data["low"] - data["close"].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            atr = true_range.rolling(14).mean().iloc[-1]

            # Stop loss at 2 * ATR below entry
            stop_loss = entry_price - (2 * atr)

            return max(stop_loss, entry_price * 0.95)  # At least 5% stop loss

        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.98

    def calculate_take_profit(self, data: pd.DataFrame, entry_price: float) -> float:
        """
        Calculate take profit based on market volatility

        Args:
            data: Market data DataFrame
            entry_price: Entry price

        Returns:
            Take profit price
        """
        try:
            if len(data) < 14:
                return entry_price * 1.02  # 2% take profit

            # Calculate ATR for volatility-based take profit
            high_low = data["high"] - data["low"]
            high_close = abs(data["high"] - data["close"].shift())
            low_close = abs(data["low"] - data["close"].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            atr = true_range.rolling(14).mean().iloc[-1]

            # Take profit at 3 * ATR above entry
            take_profit = entry_price + (3 * atr)

            return min(take_profit, entry_price * 1.05)  # At most 5% take profit

        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * 1.02
