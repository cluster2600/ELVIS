"""
Reinforcement Learning-based Position Sizing Model.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from datetime import datetime

class PositionSizingDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.next_states = torch.FloatTensor(next_states)
        self.dones = torch.FloatTensor(dones)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

class PositionSizingNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic networks."""
        action = self.actor(state)
        value = self.critic(state)
        return action, value

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class PositionSizingAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Environment parameters
        self.state_dim = config.get('state_dim', 32)
        self.action_dim = config.get('action_dim', 1)
        self.max_position_size = config.get('max_position_size', 1.0)
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        self.learning_rate = config.get('learning_rate', 3e-4)
        
        # Initialize networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PositionSizingNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = PositionSizingNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
    def prepare_state(self, market_data: pd.DataFrame, portfolio_state: Dict) -> np.ndarray:
        """Prepare state vector from market data and portfolio state."""
        # Market features
        market_features = []
        
        # Price features
        market_features.append(market_data['close'].pct_change().values[-32:])
        market_features.append(market_data['volume'].pct_change().values[-32:])
        
        # Volatility features
        market_features.append(market_data['close'].pct_change().rolling(20).std().values[-32:])
        
        # Portfolio features
        portfolio_features = [
            portfolio_state['current_position'],
            portfolio_state['unrealized_pnl'],
            portfolio_state['realized_pnl'],
            portfolio_state['available_capital']
        ]
        
        # Combine features
        state = np.concatenate([np.concatenate(market_features), portfolio_features])
        return state
        
    def get_action(self, state: np.ndarray, explore: bool = True) -> float:
        """Get position sizing action from current state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.policy_net(state)
            
        if explore:
            # Add noise for exploration
            noise = torch.randn_like(action) * 0.1
            action = (action + noise).clamp(-1, 1)
            
        # Scale action to position size
        position_size = action.item() * self.max_position_size
        return position_size
        
    def calculate_reward(self, portfolio_state: Dict, new_portfolio_state: Dict) -> float:
        """Calculate reward based on portfolio performance."""
        # PnL component
        pnl_reward = new_portfolio_state['unrealized_pnl'] - portfolio_state['unrealized_pnl']
        
        # Risk-adjusted component
        risk_penalty = -0.1 * abs(new_portfolio_state['current_position'])
        
        # Drawdown penalty
        drawdown_penalty = -0.2 * max(0, portfolio_state['max_drawdown'] - new_portfolio_state['max_drawdown'])
        
        # Combine components
        reward = pnl_reward + risk_penalty + drawdown_penalty
        return reward
        
    def update(self):
        """Update policy and target networks."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q-values
        current_actions, current_values = self.policy_net(states)
        
        # Get target Q-values
        with torch.no_grad():
            next_actions, next_values = self.target_net(next_states)
            target_values = rewards + (1 - dones) * self.gamma * next_values
            
        # Calculate losses
        value_loss = F.mse_loss(current_values, target_values)
        policy_loss = -current_values.mean()
        
        # Update networks
        self.optimizer.zero_grad()
        (value_loss + policy_loss).backward()
        self.optimizer.step()
        
        # Update target network
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def train_step(self, market_data: pd.DataFrame, portfolio_state: Dict) -> float:
        """Perform one training step."""
        # Get current state
        state = self.prepare_state(market_data, portfolio_state)
        
        # Get action
        position_size = self.get_action(state)
        
        # Execute action and get new portfolio state
        new_portfolio_state = self._execute_position_size(position_size, portfolio_state)
        
        # Calculate reward
        reward = self.calculate_reward(portfolio_state, new_portfolio_state)
        
        # Get next state
        next_state = self.prepare_state(market_data, new_portfolio_state)
        
        # Store experience in replay buffer
        self.replay_buffer.push(state, position_size, reward, next_state, False)
        
        # Update networks
        self.update()
        
        return position_size
        
    def _execute_position_size(self, position_size: float, portfolio_state: Dict) -> Dict:
        """Execute position sizing action and return new portfolio state."""
        # Update position
        new_position = position_size * portfolio_state['available_capital']
        
        # Calculate new portfolio state
        new_portfolio_state = {
            'current_position': new_position,
            'unrealized_pnl': portfolio_state['unrealized_pnl'],
            'realized_pnl': portfolio_state['realized_pnl'],
            'available_capital': portfolio_state['available_capital'] - abs(new_position),
            'max_drawdown': max(portfolio_state['max_drawdown'], portfolio_state['unrealized_pnl'])
        }
        
        return new_portfolio_state
        
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 