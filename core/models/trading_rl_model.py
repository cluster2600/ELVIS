"""
Advanced Reinforcement Learning Model for Bitcoin Trading
This module implements a sophisticated RL system that learns from actual trading data
to improve trading performance by understanding real market patterns and costs.
"""

import json
import logging
import os
import random
from collections import deque, namedtuple
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils.paper_trade_db import get_all_trades, get_conn

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class TradingFeatureExtractor:
    """Extract and normalize features from trading data for RL training"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.feature_scaler = {}
        self.feature_stats = {}

    def extract_features_from_trades(
        self, trades: List[Tuple], market_data: Dict = None
    ) -> np.ndarray:
        """
        Extract features from trading data for RL state representation

        Args:
            trades: List of trade tuples from database
            market_data: Current market data (price, volume, indicators)

        Returns:
            Feature vector for RL state
        """
        try:
            # Base features (12 features)
            if market_data:
                features = [
                    market_data.get("price", 0.0) / 100000.0,  # Normalized price
                    market_data.get("volume", 0.0) / 1000.0,  # Normalized volume
                    market_data.get("rsi", 50.0) / 100.0,  # RSI (0-1)
                    market_data.get("macd", 0.0) / 1000.0,  # MACD normalized
                    market_data.get("bb_upper", 0.0) / 100000.0,  # BB upper
                    market_data.get("bb_lower", 0.0) / 100000.0,  # BB lower
                    market_data.get("atr", 0.0) / 10000.0,  # ATR normalized
                    market_data.get("adx", 0.0) / 100.0,  # ADX (0-1)
                ]
            else:
                features = [0.0] * 8  # Default market features

            # Trading performance features (4 features)
            if trades and len(trades) > 0:
                recent_trades = trades[-20:]  # Last 20 trades

                # Calculate recent performance metrics
                recent_pnl = [
                    float(trade[6]) for trade in recent_trades if trade[6] is not None
                ]
                recent_fees = [
                    float(trade[7]) for trade in recent_trades if trade[7] is not None
                ]

                if recent_pnl:
                    avg_pnl = sum(recent_pnl) / len(recent_pnl)
                    win_rate = len([p for p in recent_pnl if p > 0]) / len(recent_pnl)
                    total_fees = sum(recent_fees) if recent_fees else 0
                    net_profit = sum(recent_pnl) - total_fees

                    features.extend(
                        [
                            avg_pnl / 10.0,  # Average PnL normalized
                            win_rate,  # Win rate (0-1)
                            total_fees / 100.0,  # Total fees normalized
                            net_profit / 100.0,  # Net profit normalized
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Time-based features (4 features)
            now = datetime.now()
            features.extend(
                [
                    now.hour / 24.0,  # Hour of day (0-1)
                    now.weekday() / 7.0,  # Day of week (0-1)
                    now.minute / 60.0,  # Minute of hour (0-1)
                    (now.timestamp() % 3600) / 3600.0,  # Position in hour (0-1)
                ]
            )

            # Ensure exactly 16 features
            while len(features) < 16:
                features.append(0.0)
            features = features[:16]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(16, dtype=np.float32)


class TradingRewardCalculator:
    """Calculate rewards based on actual trading performance"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fee_rate = 0.0004  # 0.04% fee rate

    def calculate_reward(
        self,
        action: int,
        current_price: float,
        last_price: float,
        position: float,
        trade_result: Dict = None,
    ) -> float:
        """
        Calculate reward based on trading action and outcome

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            current_price: Current market price
            last_price: Previous market price
            position: Current position (0=no position, 1=long, -1=short)
            trade_result: Actual trade results if available

        Returns:
            Reward value (can be negative)
        """
        try:
            # Base reward from price movement
            price_change = (current_price - last_price) / last_price

            # Action-specific reward calculation
            if action == 0:  # HOLD
                # Small penalty for holding to encourage trading
                # But reward if we avoided a bad trade
                if abs(price_change) < 0.001:  # Market didn't move much
                    return 0.01  # Small reward for not trading in flat market
                else:
                    return -0.02  # Penalty for missing opportunity

            elif action == 1:  # BUY
                if trade_result:
                    # Use actual trade results
                    pnl = trade_result.get("pnl", 0.0)
                    fees = trade_result.get("fees", 0.0)
                    net_result = pnl - fees

                    # Scale reward based on net result
                    if net_result > 0:
                        reward = min(net_result * 10, 5.0)  # Cap positive reward
                    else:
                        reward = max(net_result * 10, -5.0)  # Cap negative reward

                    # Bonus for profitable trades
                    if pnl > 1.0:  # $1+ profit
                        reward += 2.0

                    return reward
                else:
                    # Simulated reward based on price movement
                    if price_change > 0:
                        return price_change * 100  # Reward proportional to gain
                    else:
                        return price_change * 100 - 0.4  # Penalty including fees

            elif action == 2:  # SELL
                if trade_result:
                    # Use actual trade results (similar to BUY)
                    pnl = trade_result.get("pnl", 0.0)
                    fees = trade_result.get("fees", 0.0)
                    net_result = pnl - fees

                    if net_result > 0:
                        reward = min(net_result * 10, 5.0)
                    else:
                        reward = max(net_result * 10, -5.0)

                    # Bonus for profitable trades
                    if pnl > 1.0:
                        reward += 2.0

                    return reward
                else:
                    # Simulated reward (opposite of BUY)
                    if price_change < 0:
                        return abs(price_change) * 100  # Reward for selling before drop
                    else:
                        return (
                            -price_change * 100 - 0.4
                        )  # Penalty for selling before rise

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return -1.0  # Penalty for errors


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""

    def __init__(
        self, input_size: int = 16, hidden_size: int = 128, output_size: int = 3
    ):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.network(x)


class TradingDQNAgent:
    """DQN Agent for trading decisions"""

    def __init__(
        self,
        state_size: int = 16,
        action_size: int = 3,
        learning_rate: float = 0.001,
        logger: logging.Logger = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.logger = logger or logging.getLogger(__name__)

        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.update_frequency = 4
        self.target_update_frequency = 100

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, 128, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.step_count = 0

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)

        # Convert to numpy arrays first, then to tensors (more efficient)
        states_array = np.array([e.state for e in batch])
        actions_array = np.array([e.action for e in batch])
        rewards_array = np.array([e.reward for e in batch])
        next_states_array = np.array([e.next_state for e in batch])
        dones_array = np.array([e.done for e in batch])

        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.BoolTensor(dones_array).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_model(self, path: str):
        """Load model from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            self.logger.info(f"Model loaded from {path}")
            return True
        return False


class TradingRLModel:
    """Main RL model for trading decisions"""

    def __init__(
        self, logger: logging.Logger, model_path: str = "models/trading_rl_model.pth"
    ):
        self.logger = logger
        self.model_path = model_path

        # Initialize components
        self.feature_extractor = TradingFeatureExtractor(logger)
        self.reward_calculator = TradingRewardCalculator(logger)
        self.agent = TradingDQNAgent(logger=logger)

        # Training state
        self.training_history = []
        self.last_state = None
        self.last_action = None
        self.last_price = None
        self.episode_rewards = []

        # Load existing model if available
        if os.path.exists(model_path):
            self.load_model()

    def train_on_historical_data(self, limit: int = 1000):
        """Train the model on historical trading data"""
        try:
            self.logger.info("Training RL model on historical trading data...")

            # Get historical trades
            conn = get_conn()
            if not conn:
                self.logger.error("Cannot connect to database")
                return False

            c = conn.cursor()
            c.execute(
                """
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE side != 'TEST'
                ORDER BY timestamp ASC
                LIMIT %s
            """,
                (limit,),
            )

            trades = c.fetchall()
            conn.close()

            if not trades:
                self.logger.info(
                    "No historical trades found, creating synthetic training data"
                )
                return self._create_synthetic_training_data()

            self.logger.info(f"Training on {len(trades)} historical trades")

            # Process trades for training
            training_losses = []
            episode_reward = 0

            for i, trade in enumerate(trades[:-1]):
                try:
                    # Extract current state features
                    current_state = self.feature_extractor.extract_features_from_trades(
                        trades[: i + 1], {"price": trade[4], "volume": trade[5] * 1000}
                    )

                    # Determine action from trade
                    action = self._trade_to_action(trade[3])  # side: BUY/SELL

                    # Calculate reward based on next trade
                    if i + 1 < len(trades):
                        next_trade = trades[i + 1]
                        reward = self.reward_calculator.calculate_reward(
                            action,
                            next_trade[4],
                            trade[4],
                            0,
                            {"pnl": trade[6], "fees": trade[7]},
                        )

                        # Next state
                        next_state = (
                            self.feature_extractor.extract_features_from_trades(
                                trades[: i + 2],
                                {
                                    "price": next_trade[4],
                                    "volume": next_trade[5] * 1000,
                                },
                            )
                        )

                        # Store experience
                        done = i == len(trades) - 2
                        self.agent.remember(
                            current_state, action, reward, next_state, done
                        )

                        episode_reward += reward

                        # Train agent
                        if len(self.agent.memory) > self.agent.batch_size:
                            loss = self.agent.replay()
                            training_losses.append(loss)

                            # Update target network periodically
                            if (
                                self.agent.step_count
                                % self.agent.target_update_frequency
                                == 0
                            ):
                                self.agent.update_target_network()

                        self.agent.step_count += 1

                except Exception as e:
                    self.logger.error(f"Error processing trade {i}: {e}")
                    continue

            # Log training results
            avg_loss = (
                sum(training_losses) / len(training_losses) if training_losses else 0
            )
            self.logger.info(
                f"Training completed - Episode reward: {episode_reward:.2f}, Avg loss: {avg_loss:.6f}"
            )
            self.logger.info(f"Final epsilon: {self.agent.epsilon:.4f}")

            self.episode_rewards.append(episode_reward)

            # Save trained model
            self.save_model()

            return True

        except Exception as e:
            self.logger.error(f"Error training RL model: {e}")
            return False

    def _create_synthetic_training_data(self) -> bool:
        """Create synthetic training data when no historical trades exist"""
        try:
            self.logger.info(
                "Creating synthetic training data for RL model initialization..."
            )

            # Generate synthetic market scenarios
            training_losses = []
            episode_reward = 0

            # Create 100 synthetic experiences
            for i in range(100):
                # Random market state
                current_state = np.random.normal(0, 0.5, 16).astype(np.float32)
                current_state = np.clip(current_state, -2, 2)  # Bound values

                # Random action
                action = np.random.randint(0, 3)

                # Simulate reward based on action
                if action == 0:  # HOLD
                    reward = np.random.normal(0, 0.1)  # Small random reward/penalty
                elif action == 1:  # BUY
                    reward = np.random.normal(0.5, 1.0)  # Positive bias with variance
                else:  # SELL
                    reward = np.random.normal(0.5, 1.0)  # Positive bias with variance

                # Next state (slight variation)
                next_state = current_state + np.random.normal(0, 0.1, 16)
                next_state = np.clip(next_state, -2, 2)

                # Store experience
                done = i == 99
                self.agent.remember(current_state, action, reward, next_state, done)
                episode_reward += reward

                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    training_losses.append(loss)

                    # Update target network periodically
                    if self.agent.step_count % self.agent.target_update_frequency == 0:
                        self.agent.update_target_network()

                self.agent.step_count += 1

            # Log training results
            avg_loss = (
                sum(training_losses) / len(training_losses) if training_losses else 0
            )
            self.logger.info(
                f"Synthetic training completed - Episode reward: {episode_reward:.2f}, Avg loss: {avg_loss:.6f}"
            )
            self.logger.info(
                f"RL model initialized with synthetic data - ready for real trading"
            )

            self.episode_rewards.append(episode_reward)

            # Save model
            self.save_model()

            return True

        except Exception as e:
            self.logger.error(f"Error creating synthetic training data: {e}")
            return False

    def _trade_to_action(self, side: str) -> int:
        """Convert trade side to action"""
        if side == "BUY":
            return 1
        elif side == "SELL":
            return 2
        else:
            return 0  # HOLD

    def _action_to_signal(self, action: int) -> str:
        """Convert action to trading signal"""
        if action == 1:
            return "BUY"
        elif action == 2:
            return "SELL"
        else:
            return "HOLD"

    def predict_action(
        self, market_data: Dict, trades: List = None
    ) -> Tuple[str, float]:
        """
        Predict trading action based on current market conditions

        Args:
            market_data: Current market data
            trades: Recent trades for context

        Returns:
            (signal, confidence) tuple
        """
        try:
            # Extract features from current state
            if trades is None:
                trades = self._get_recent_trades(20)

            state = self.feature_extractor.extract_features_from_trades(
                trades, market_data
            )

            # Get action probabilities from network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                q_values = self.agent.q_network(state_tensor)
                probabilities = F.softmax(q_values, dim=1)

            # Convert to action and confidence
            action = q_values.argmax().item()
            confidence = probabilities[0][action].item()

            signal = self._action_to_signal(action)

            self.logger.info(f"RL prediction: {signal} (confidence: {confidence:.3f})")
            self.logger.debug(f"Q-values: {q_values.cpu().numpy()}")

            return signal, confidence

        except Exception as e:
            self.logger.error(f"Error predicting action: {e}")
            return "HOLD", 0.5

    def _get_recent_trades(self, limit: int = 20) -> List:
        """Get recent trades from database"""
        try:
            conn = get_conn()
            if not conn:
                return []

            c = conn.cursor()
            c.execute(
                """
                SELECT id, timestamp, symbol, side, price, quantity, pnl, fee
                FROM trades
                WHERE side != 'TEST'
                ORDER BY timestamp DESC
                LIMIT %s
            """,
                (limit,),
            )

            trades = c.fetchall()
            conn.close()

            return list(reversed(trades))  # Return in chronological order

        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []

    def update_with_trade_result(self, trade_result: Dict):
        """Update model with actual trade results for online learning"""
        try:
            if self.last_state is None or self.last_action is None:
                return

            # Calculate reward from actual trade result
            current_price = trade_result.get("price", 0.0)
            reward = self.reward_calculator.calculate_reward(
                self.last_action, current_price, self.last_price, 0, trade_result
            )

            # Get current state
            trades = self._get_recent_trades(20)
            current_state = self.feature_extractor.extract_features_from_trades(
                trades, {"price": current_price}
            )

            # Store experience
            self.agent.remember(
                self.last_state, self.last_action, reward, current_state, False
            )

            # Train on new experience
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()

                if self.agent.step_count % 10 == 0:  # Log every 10 steps
                    self.logger.info(
                        f"Online learning - Reward: {reward:.3f}, Loss: {loss:.6f}"
                    )

                # Update target network periodically
                if self.agent.step_count % self.agent.target_update_frequency == 0:
                    self.agent.update_target_network()

            self.agent.step_count += 1

            # Update state
            self.last_state = current_state
            self.last_price = current_price

        except Exception as e:
            self.logger.error(f"Error updating with trade result: {e}")

    def save_model(self):
        """Save model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.agent.save_model(self.model_path)

            # Save training history
            history_path = self.model_path.replace(".pth", "_history.json")
            with open(history_path, "w") as f:
                json.dump(
                    {
                        "episode_rewards": self.episode_rewards,
                        "training_history": self.training_history,
                    },
                    f,
                )

            self.logger.info(f"RL model saved to {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load model from disk"""
        try:
            if self.agent.load_model(self.model_path):
                # Load training history
                history_path = self.model_path.replace(".pth", "_history.json")
                if os.path.exists(history_path):
                    with open(history_path, "r") as f:
                        data = json.load(f)
                        self.episode_rewards = data.get("episode_rewards", [])
                        self.training_history = data.get("training_history", [])

                self.logger.info(f"RL model loaded from {self.model_path}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "latest_reward": self.episode_rewards[-1],
            "best_reward": max(self.episode_rewards),
            "worst_reward": min(self.episode_rewards),
            "epsilon": self.agent.epsilon,
            "memory_size": len(self.agent.memory),
        }
