"""
Reinforcement Learning model for the ELVIS project.
This module provides a concrete implementation of the BaseModel using reinforcement learning.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gym
from gym import spaces

from core.models.base_model import BaseModel
from config import FILE_PATHS

class Actor(nn.Module):
    """
    Actor network for PPO algorithm.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): The state dimension.
            action_dim (int): The action dimension.
            hidden_dim (int): The hidden dimension.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action mean for continuous actions
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        
        # Action log std for continuous actions
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Action probs for discrete actions
        self.action_probs = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): The state tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action mean and log std for continuous actions,
                                              or action probabilities for discrete actions.
        """
        x = self.network(state)
        
        # For continuous actions
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        
        # For discrete actions
        action_probs = self.action_probs(x)
        
        return action_mean, action_log_std, action_probs
    
    def get_action(self, state: torch.Tensor, continuous: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from state.
        
        Args:
            state (torch.Tensor): The state tensor.
            continuous (bool): Whether the action space is continuous.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action and log probability.
        """
        action_mean, action_log_std, action_probs = self.forward(state)
        
        if continuous:
            # For continuous actions
            action_std = torch.exp(action_log_std)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            # For discrete actions
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).unsqueeze(-1)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, continuous: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate action given state.
        
        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.
            continuous (bool): Whether the action space is continuous.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The log probability and entropy.
        """
        action_mean, action_log_std, action_probs = self.forward(state)
        
        if continuous:
            # For continuous actions
            action_std = torch.exp(action_log_std)
            distribution = Normal(action_mean, action_std)
            log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        else:
            # For discrete actions
            distribution = Categorical(action_probs)
            log_prob = distribution.log_prob(action).unsqueeze(-1)
            entropy = distribution.entropy().unsqueeze(-1)
        
        return log_prob, entropy

class Critic(nn.Module):
    """
    Critic network for PPO algorithm.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): The state dimension.
            hidden_dim (int): The hidden dimension.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): The state tensor.
            
        Returns:
            torch.Tensor: The value.
        """
        return self.network(state)

class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        K_epochs: int = 10,
        eps_clip: float = 0.2,
        continuous: bool = True,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the PPO algorithm.
        
        Args:
            state_dim (int): The state dimension.
            action_dim (int): The action dimension.
            hidden_dim (int): The hidden dimension.
            lr_actor (float): The learning rate for the actor.
            lr_critic (float): The learning rate for the critic.
            gamma (float): The discount factor.
            K_epochs (int): The number of epochs to update the policy.
            eps_clip (float): The clipping parameter.
            continuous (bool): Whether the action space is continuous.
            device (torch.device): The device to use.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.continuous = continuous
        self.device = device
        
        # Initialize actor and critic
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize old actor for PPO
        self.actor_old = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Initialize loss function
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action from state.
        
        Args:
            state (np.ndarray): The state.
            
        Returns:
            np.ndarray: The action.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, _ = self.actor_old.get_action(state, self.continuous)
        
        return action.cpu().numpy()
    
    def update(self, memory: List[Dict[str, torch.Tensor]]) -> None:
        """
        Update the policy.
        
        Args:
            memory (List[Dict[str, torch.Tensor]]): The memory of transitions.
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.stack(memory['states']).to(self.device).detach()
        old_actions = torch.stack(memory['actions']).to(self.device).detach()
        old_logprobs = torch.stack(memory['logprobs']).to(self.device).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.mean().backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action given state.
        
        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The log probability, value, and entropy.
        """
        logprobs, entropy = self.actor.evaluate(state, action, self.continuous)
        state_values = self.critic(state)
        
        return logprobs, state_values, entropy
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path (str): The path to save the model.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_old': self.actor_old.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path (str): The path to load the model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(checkpoint['actor_old'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])

class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, transaction_fee: float = 0.001):
        """
        Initialize the trading environment.
        
        Args:
            data (pd.DataFrame): The data to use.
            initial_balance (float): The initial balance.
            transaction_fee (float): The transaction fee.
        """
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space: [balance, position, price, technical indicators...]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1] + 2,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            np.ndarray: The initial state.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.done = False
        self.history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action (int): The action to take.
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: The next state, reward, done, and info.
        """
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate quantity to buy
                quantity = self.balance / current_price
                # Apply transaction fee
                quantity *= (1 - self.transaction_fee)
                # Update position and balance
                self.position += quantity
                self.balance = 0
        elif action == 2:  # Sell
            if self.position > 0:
                # Calculate amount to receive
                amount = self.position * current_price
                # Apply transaction fee
                amount *= (1 - self.transaction_fee)
                # Update position and balance
                self.position = 0
                self.balance = amount
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Save history
        self.history.append({
            'step': self.current_step,
            'action': action,
            'balance': self.balance,
            'position': self.position,
            'price': current_price,
            'reward': reward
        })
        
        return next_observation, reward, self.done, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            np.ndarray: The current observation.
        """
        # Get current data
        current_data = self.data.iloc[self.current_step].values
        
        # Create observation
        observation = np.concatenate([
            [self.balance, self.position],
            current_data
        ])
        
        return observation
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward.
        
        Returns:
            float: The reward.
        """
        # Calculate portfolio value
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        # Calculate reward as change in portfolio value
        if self.current_step > 0:
            previous_price = self.data.iloc[self.current_step - 1]['close']
            previous_portfolio_value = self.balance + (self.position * previous_price)
            reward = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            reward = 0.0
        
        return reward

class ReinforcementLearningModel(BaseModel):
    """
    Reinforcement Learning model for trading.
    Uses PPO algorithm for training.
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize the reinforcement learning model.
        
        Args:
            logger (logging.Logger): The logger to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__('reinforcement_learning', logger, **kwargs)
        
        # Model parameters
        self.state_dim = kwargs.get('state_dim', 10)
        self.action_dim = kwargs.get('action_dim', 3)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.lr_actor = kwargs.get('lr_actor', 3e-4)
        self.lr_critic = kwargs.get('lr_critic', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.K_epochs = kwargs.get('K_epochs', 10)
        self.eps_clip = kwargs.get('eps_clip', 0.2)
        self.continuous = kwargs.get('continuous', False)
        self.max_episodes = kwargs.get('max_episodes', 1000)
        self.max_timesteps = kwargs.get('max_timesteps', 1000)
        self.update_timestep = kwargs.get('update_timestep', 100)
        
        # Model path
        self.model_path = kwargs.get('model_path', os.path.join(FILE_PATHS['TRAIN_RESULTS_DIR'], 'rl_model.pt'))
        
        # Initialize model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self) -> PPO:
        """
        Build the reinforcement learning model.
        
        Returns:
            PPO: The built model.
        """
        model = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            gamma=self.gamma,
            K_epochs=self.K_epochs,
            eps_clip=self.eps_clip,
            continuous=self.continuous,
            device=self.device
        )
        
        return model
    
    def load_model(self) -> None:
        """
        Load the model from disk.
        """
        try:
            self.logger.info(f"Loading Reinforcement Learning model from {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                return
            
            # Build model
            self.model = self._build_model()
            
            # Load model
            self.model.load(self.model_path)
            
            self.logger.info("Reinforcement Learning model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Reinforcement Learning model: {e}")
    
    def save_model(self) -> None:
        """
        Save the model to disk.
        """
        try:
            self.logger.info(f"Saving Reinforcement Learning model to {self.model_path}")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            self.logger.info("Reinforcement Learning model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving Reinforcement Learning model: {e}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            self.logger.info("Training Reinforcement Learning model")
            
            # Create environment
            env = TradingEnv(X_train)
            
            # Update state dimension
            self.state_dim = env.observation_space.shape[0]
            
            # Build model
            self.model = self._build_model()
            
            # Training loop
            for episode in range(self.max_episodes):
                state = env.reset()
                episode_reward = 0
                
                # Memory for PPO
                memory = {
                    'states': [],
                    'actions': [],
                    'logprobs': [],
                    'rewards': [],
                    'is_terminals': []
                }
                
                for t in range(self.max_timesteps):
                    # Select action
                    action = self.model.select_action(state)
                    
                    # Execute action
                    next_state, reward, done, _ = env.step(action)
                    
                    # Add to memory
                    memory['states'].append(torch.FloatTensor(state).to(self.device))
                    memory['actions'].append(torch.FloatTensor(action).to(self.device))
                    memory['logprobs'].append(torch.FloatTensor(action_logprob).to(self.device))
                    memory['rewards'].append(reward)
                    memory['is_terminals'].append(done)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    
                    # Update if its time
                    if t % self.update_timestep == 0:
                        self.model.update(memory)
                        memory = {
                            'states': [],
                            'actions': [],
                            'logprobs': [],
                            'rewards': [],
                            'is_terminals': []
                        }
                    
                    if done:
                        break
                
                # Log progress
                if (episode + 1) % 10 == 0:
                    self.logger.info(f"Episode {episode + 1}/{self.max_episodes}, Reward: {episode_reward:.2f}")
            
            self.logger.info("Reinforcement Learning model trained successfully")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error training Reinforcement Learning model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (pd.DataFrame): The features to predict on.
            
        Returns:
            np.ndarray: The predictions.
        """
        try:
            self.logger.info("Making predictions with Reinforcement Learning model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return np.zeros(len(X))
            
            # Create environment
            env = TradingEnv(X)
            
            # Initialize predictions
            predictions = np.zeros(len(X))
            
            # Run model
            state = env.reset()
            done = False
            step = 0
            
            while not done and step < len(X):
                # Select action
                action = self.model.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                # Record prediction
                predictions[step] = action
                
                # Update state
                state = next_state
                step += 1
            
            self.logger.info(f"Made {step} predictions")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with Reinforcement Learning model: {e}")
            return np.zeros(len(X))
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test labels.
            
        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        try:
            self.logger.info("Evaluating Reinforcement Learning model")
            
            # Check if model exists
            if self.model is None:
                self.logger.warning("No model loaded. Loading model...")
                self.load_model()
                
                if self.model is None:
                    self.logger.error("Failed to load model")
                    return {'total_reward': 0.0, 'final_balance': 0.0}
            
            # Create environment
            env = TradingEnv(X_test)
            
            # Run model
            state = env.reset()
            done = False
            total_reward = 0.0
            
            while not done:
                # Select action
                action = self.model.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                # Update state
                state = next_state
                total_reward += reward
            
            # Calculate final balance
            final_balance = env.balance + (env.position * X_test.iloc[-1]['close'])
            
            # Calculate metrics
            metrics = {
                'total_reward': total_reward,
                'final_balance': final_balance,
                'return_pct': (final_balance - env.initial_balance) / env.initial_balance * 100
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating Reinforcement Learning model: {e}")
            return {'total_reward': 0.0, 'final_balance': 0.0}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance.
        Reinforcement learning models don't have direct feature importance.
        
        Returns:
            pd.DataFrame: The feature importance.
        """
        self.logger.warning("Feature importance not available for Reinforcement Learning models")
        return pd.DataFrame()
