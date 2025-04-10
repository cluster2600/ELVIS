# ELVIS Training Documentation

## Overview

The training component of ELVIS (Enhanced Leveraged Virtual Investment System) is designed to train reinforcement learning agents for cryptocurrency trading. It implements various state-of-the-art reinforcement learning algorithms including PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic), TD3 (Twin Delayed DDPG), DDPG (Deep Deterministic Policy Gradient), and A2C (Advantage Actor-Critic).

## Training Architecture

The training system is built on the ElegantRL framework and consists of several key components:

1. **Environment**: A custom trading environment that simulates the cryptocurrency market
2. **Agents**: Reinforcement learning algorithms implemented in the `drl_agents/agents/` directory
3. **Training Pipeline**: Orchestrates the training process in the `train/` directory
4. **Evaluation**: Assesses agent performance and saves models in the `train/evaluator.py` file

## Training Process

The training process follows these steps:

1. **Environment Setup**: The trading environment is initialized with market data and parameters
2. **Agent Initialization**: The selected RL agent is initialized with the specified hyperparameters
3. **Exploration**: The agent explores the environment to collect experience
4. **Learning**: The agent learns from the collected experience by updating its neural networks
5. **Evaluation**: The agent's performance is evaluated periodically
6. **Model Saving**: The best performing models are saved for deployment

## Available Agents

ELVIS supports the following reinforcement learning agents:

### PPO (Proximal Policy Optimization)
- **File**: `drl_agents/agents/AgentPPO.py`
- **Description**: A policy gradient method that uses a clipped surrogate objective function
- **Key Features**:
  - Supports both continuous and discrete action spaces
  - Implements GAE (Generalized Advantage Estimation) for sparse rewards
  - Uses entropy regularization for exploration

### SAC (Soft Actor-Critic)
- **File**: `drl_agents/agents/AgentSAC.py`
- **Description**: An off-policy algorithm that maximizes expected return and entropy
- **Key Features**:
  - Automatically adjusts the temperature parameter
  - Uses a stochastic policy for better exploration
  - Implements twin critics to reduce overestimation bias

### TD3 (Twin Delayed DDPG)
- **File**: `drl_agents/agents/AgentTD3.py`
- **Description**: An improvement over DDPG with twin critics and delayed policy updates
- **Key Features**:
  - Uses twin critics to reduce overestimation bias
  - Delays policy updates to improve stability
  - Adds noise to target actions for regularization

### DDPG (Deep Deterministic Policy Gradient)
- **File**: `drl_agents/agents/AgentDDPG.py`
- **Description**: An actor-critic algorithm for continuous action spaces
- **Key Features**:
  - Uses a deterministic policy
  - Implements experience replay for off-policy learning
  - Uses target networks for stable learning

### A2C (Advantage Actor-Critic)
- **File**: `drl_agents/agents/AgentA2C.py`
- **Description**: A synchronous version of A3C for on-policy learning
- **Key Features**:
  - Uses advantage estimation for reduced variance
  - Supports both continuous and discrete action spaces
  - Implements entropy regularization for exploration

## Training Configuration

The training process is configured using the `Arguments` class in `train/config.py`. Key configuration parameters include:

- **Agent Selection**: Choose the RL algorithm to use
- **Network Architecture**: Configure the neural network dimensions and layers
- **Hyperparameters**: Set learning rates, batch sizes, and other algorithm-specific parameters
- **Environment Settings**: Configure the trading environment parameters
- **Training Schedule**: Set the number of steps, evaluation frequency, and early stopping criteria

## Running Training

To train a model, use the `run_training.sh` script:

```bash
./run_training.sh
```

This script will:
1. Activate the virtual environment
2. Install required packages
3. Run the model training pipeline
4. Save trained models and optimized parameters

## Training Pipeline Components

### Learner (`train/learner.py`)
- Manages the learning process
- Updates the agent's neural networks
- Handles multi-GPU training

### Evaluator (`train/evaluator.py`)
- Evaluates the agent's performance
- Saves the best models
- Generates learning curves and performance metrics

### Replay Buffer (`train/replay_buffer.py`)
- Stores and samples experiences for off-policy learning
- Implements prioritized experience replay (PER)
- Handles trajectory data for on-policy algorithms

### Worker (`train/worker.py`)
- Collects experiences by running the agent in the environment
- Used for parallel data collection in distributed training

## Training Results

Training results are saved in the following formats:

- **Model Checkpoints**: Saved as PyTorch state dictionaries
- **Learning Curves**: Plots of performance metrics over time
- **Evaluation Results**: Detailed performance metrics for each evaluation
- **Configuration**: The hyperparameters used for training

## Advanced Training Features

### Multi-GPU Training
The training system supports distributed training across multiple GPUs:

```python
args.learner_gpus = [0, 1, 2, 3]  # Use GPUs 0, 1, 2, and 3
```

### Early Stopping
Training can be configured to stop early when the agent reaches a target performance:

```python
args.target_return = 1000  # Stop when average return reaches 1000
args.if_allow_break = True  # Allow early stopping
```

### Custom Environments
You can create custom trading environments by implementing the Gym interface:

```python
class CustomTradingEnv(gym.Env):
    def __init__(self):
        # Initialize the environment
        pass
    
    def reset(self):
        # Reset the environment
        pass
    
    def step(self, action):
        # Execute the action and return the next state, reward, done, and info
        pass
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size or network size
   - Use gradient accumulation for larger effective batch sizes

2. **Unstable Training**
   - Adjust learning rates
   - Increase the replay buffer size
   - Use a more stable algorithm like PPO

3. **Poor Performance**
   - Check the reward function design
   - Increase the network capacity
   - Adjust the exploration parameters

### Debugging Tips

1. **Monitor Training Progress**
   - Use the console dashboard to monitor training metrics
   - Check the learning curves for signs of improvement

2. **Validate Environment**
   - Test the environment with random actions
   - Verify that the state and action spaces are correctly defined

3. **Profile Performance**
   - Use PyTorch profilers to identify bottlenecks
   - Optimize data loading and preprocessing

## Future Improvements

1. **Meta-Learning**
   - Implement meta-learning to adapt to changing market conditions
   - Use model-based RL for faster adaptation

2. **Multi-Agent Training**
   - Train multiple agents to compete or cooperate
   - Implement market maker and taker agents

3. **Transfer Learning**
   - Pre-train on historical data
   - Fine-tune on recent market conditions

4. **Automated Hyperparameter Optimization**
   - Implement Bayesian optimization for hyperparameter tuning
   - Use population-based training for adaptive hyperparameters 