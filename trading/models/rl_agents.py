"""
Reinforcement learning agents for trading.
Implements PPO, SAC, and DDPG algorithms with custom reward functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TradingEnvironment:
    """Custom trading environment for reinforcement learning."""
    
    def __init__(self, 
                 data: np.ndarray,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        self.trades = []
        return self._get_observation()
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        action: float between -1 and 1 representing position size
        """
        # Calculate new position
        new_position = np.clip(action, -self.max_position, self.max_position)
        
        # Calculate transaction cost
        position_change = abs(new_position - self.position)
        transaction_cost = position_change * self.transaction_cost * self.balance
        
        # Update position and balance
        self.position = new_position
        self.balance -= transaction_cost
        
        # Calculate reward
        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = (next_price - current_price) / current_price
        reward = self.position * price_change * self.balance - transaction_cost
        
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Store trade information
        self.trades.append({
            'step': self.current_step,
            'position': self.position,
            'price': current_price,
            'reward': reward
        })
        
        return self._get_observation(), reward, done, {}
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment."""
        if self.current_step >= len(self.data) - 1:
            return np.zeros(10)  # Return zeros for final step
            
        # Create observation vector
        current_price = self.data[self.current_step]
        price_history = self.data[max(0, self.current_step-9):self.current_step+1]
        returns = np.diff(price_history) / price_history[:-1]
        
        observation = np.concatenate([
            [self.position],
            [self.balance / self.initial_balance],
            returns,
            np.zeros(10 - len(returns) - 2)  # Pad with zeros if needed
        ])
        
        return observation

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for trading data."""
    
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class MultiAgentTradingSystem:
    """Multi-agent system for collaborative trading."""
    
    def __init__(self, 
                 n_agents: int = 3,
                 data: np.ndarray = None,
                 initial_balance: float = 10000.0):
        self.n_agents = n_agents
        self.data = data
        self.initial_balance = initial_balance
        self.agents = []
        self.environments = []
        
        # Initialize agents and environments
        for i in range(n_agents):
            env = TradingEnvironment(data, initial_balance)
            self.environments.append(env)
            
            # Create different agents with different strategies
            if i == 0:
                agent = PPO("MlpPolicy", env, verbose=1)
            elif i == 1:
                agent = SAC("MlpPolicy", env, verbose=1)
            else:
                agent = DDPG("MlpPolicy", env, verbose=1)
                
            self.agents.append(agent)
            
    def train(self, total_timesteps: int = 100000):
        """Train all agents in the system."""
        for i, (agent, env) in enumerate(zip(self.agents, self.environments)):
            print(f"Training agent {i+1}/{self.n_agents}")
            agent.learn(total_timesteps=total_timesteps)
            
    def predict(self, observation: np.ndarray) -> List[float]:
        """Get predictions from all agents."""
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(observation)
            actions.append(action)
        return actions
        
    def get_ensemble_action(self, observation: np.ndarray) -> float:
        """Get ensemble action by averaging predictions."""
        actions = self.predict(observation)
        return np.mean(actions)

class CustomRewardFunction:
    """Custom reward function for trading."""
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 max_drawdown_penalty: float = 0.5,
                 volatility_penalty: float = 0.3):
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_penalty = max_drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.returns = []
        self.max_balance = 0
        
    def calculate_reward(self,
                        current_balance: float,
                        position: float,
                        price_change: float,
                        transaction_cost: float) -> float:
        """Calculate custom reward based on multiple factors."""
        # Basic return
        basic_return = position * price_change * current_balance
        
        # Risk-adjusted return (Sharpe ratio component)
        self.returns.append(basic_return)
        if len(self.returns) > 1:
            sharpe_component = (np.mean(self.returns) - self.risk_free_rate) / (np.std(self.returns) + 1e-6)
        else:
            sharpe_component = 0
            
        # Drawdown penalty
        self.max_balance = max(self.max_balance, current_balance)
        drawdown = (self.max_balance - current_balance) / self.max_balance
        drawdown_penalty = -self.max_drawdown_penalty * drawdown
        
        # Volatility penalty
        volatility_penalty = -self.volatility_penalty * abs(position) * abs(price_change)
        
        # Transaction cost penalty
        transaction_penalty = -transaction_cost
        
        # Combine all components
        reward = (basic_return + 
                 sharpe_component + 
                 drawdown_penalty + 
                 volatility_penalty + 
                 transaction_penalty)
        
        return reward 